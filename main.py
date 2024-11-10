from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
from loguru import logger
from services.ollama_service import OllamaService
from services.vllm_service import VLLMService
from dotenv import load_dotenv
from enum import Enum
from models.database import SessionLocal, Base, engine
import crud.model_crud as model_crud
from fastapi import Depends
from datetime import datetime
from sqlalchemy.orm import Session
import asyncio
from asyncio import Queue

load_dotenv()

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서비스 초기화
ollama_service = OllamaService()
vllm_service = VLLMService()


# 열거형 정의
class ModelEngineType(str, Enum):
    OLLAMA = "ollama"
    VLLM = "vllm"


class ModelUsageType(str, Enum):
    GENERATION = "generation"
    EMBEDDING = "embedding"


# 기본 모델 설정 클래스
class BaseModelConfig(BaseModel):
    name: str
    usageType: ModelUsageType
    gpuId: Optional[str] = None
    image: Optional[str] = None



# Ollama 모델 설정 클래스
class OllamaModelConfig(BaseModelConfig):
    engine: ModelEngineType = ModelEngineType.OLLAMA
    parameters: Optional[Dict[str, Any]] = {"temperature": 0.7}


# VLLM 모델 설정 클래스
class VLLMModelConfig(BaseModelConfig):
    engine: ModelEngineType = ModelEngineType.VLLM
    parameters: Optional[Dict[str, Any]] = None


# 통합 모델 설정 타입
ModelConfig = Union[OllamaModelConfig, VLLMModelConfig]


class ModelResponse(BaseModel):
    id: str
    name: str
    status: str
    message: Optional[str] = None


class ModelInfo(BaseModel):
    id: str
    name: str
    engine: str
    status: str
    container_id: Optional[str] = None


class TestPrompt(BaseModel):
    prompt: str


# 컨테이너 정보 클래스
class ContainerInfo(BaseModel):
    containerId: str
    engine: str
    image: str
    port: int
    gpuId: Optional[str] = None


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 전역 큐와 작업자 설정
model_task_queue: Optional[Queue] = None
MAX_WORKERS = 3  # 동시에 처리할 수 있는 최대 작업 수

# 작업자 함수
async def model_worker(worker_id: int):
    while True:
        try:
            task = await model_task_queue.get()
            config, db, result, task_type = task
            
            logger.info(f"Worker {worker_id}: 모델 {config.name} {task_type} 작업을 처리합니다")
            
            try:
                if task_type == "start":
                    if config.engine == ModelEngineType.OLLAMA:
                        container = await ollama_service.start_model(config.name, config)
                    elif config.engine == ModelEngineType.VLLM:
                        container = await vllm_service.start_model(
                            config.name,
                            {"device_ids": [config.gpu_id] if config.gpu_id else None, "model_args": config.parameters},
                        )
                elif task_type == "restart":
                    if config.engine == ModelEngineType.OLLAMA:
                        container = await ollama_service.restart_model(config.name, config)
                    elif config.engine == ModelEngineType.VLLM:
                        container = await vllm_service.restart_model(
                            config.name,
                            {"device_ids": config.parameters.get("device_ids"), "model_args": config.parameters},
                        )
                
                # 컨테이너 ID와 함께 상태 업데이트
                model_crud.update_model_status(
                    db, 
                    result["id"], 
                    "running",
                    container_id=container["id"]
                )
                logger.info(f"Worker {worker_id}: 모델 {config.name} {task_type}이 완료되었습니다")
            except Exception as e:
                model_crud.update_model_status(db, result["id"], "failed")
                logger.error(f"Worker {worker_id}: 모델 {task_type} 작업 실패: {str(e)}")
            finally:
                model_task_queue.task_done()
                
        except Exception as e:
            logger.error(f"Worker {worker_id} 오류 발생: {str(e)}")
            await asyncio.sleep(1)

# 전역 큐 추가
test_task_queue: Optional[Queue] = None
MAX_TEST_WORKERS = 2  # 테스트 동시 처리 최대 수

# 테스트 작업자 함수
async def test_model_worker(worker_id: int):
    while True:
        try:
            task = await test_task_queue.get()
            model_state, prompt, response_queue = task
            
            logger.info(f"Test Worker {worker_id}: 모델 {model_state.name} 테스트 작업을 처리합니다")
            
            try:
                if model_state.engine_type == "ollama":
                    result = await ollama_service.test_model(
                        model_state.container_id, 
                        model_state.name, 
                        prompt.prompt
                    )
                elif model_state.engine_type == "vllm":
                    result = await vllm_service.test_model(
                        model_state.container_id, 
                        model_state.name, 
                        prompt.prompt
                    )
                
                logger.info(f"Test Worker {worker_id}: 모델 {model_state.name} 테스트가 완료되었습니다")
                response = ModelResponse(
                    id=result["id"],
                    name=result["name"],
                    status=result["status"],
                    message=str(result.get("response", "Test completed successfully")),
                )
                # 결과를 응답 큐에 전달
                await response_queue.put(response)
                
            except Exception as e:
                logger.error(f"Test Worker {worker_id}: 테스트 작업 실패: {str(e)}")
                # 에러도 응답 큐에 전달
                await response_queue.put(
                    ModelResponse(
                        id=model_state.id,
                        name=model_state.name,
                        status="error",
                        message=str(e)
                    )
                )
            finally:
                test_task_queue.task_done()
                
        except Exception as e:
            logger.error(f"Test Worker {worker_id} 오류 발생: {str(e)}")
            await asyncio.sleep(1)

# 시작 이벤트 수정
@app.on_event("startup")
async def startup_event():
    global model_task_queue, test_task_queue
    
    logger.info("서버를 시작합니다.")
    
    # 작업 큐 초기화
    model_task_queue = Queue()
    test_task_queue = Queue(maxsize=MAX_TEST_WORKERS)  # 최대 크기 제한
    
    # 작업자 시작
    for i in range(MAX_WORKERS):
        asyncio.create_task(model_worker(i))
    
    # 테스트 작업자 시작
    for i in range(MAX_TEST_WORKERS):
        asyncio.create_task(test_model_worker(i))

    try:
        db = SessionLocal()
        
        # 현재 실행 중인 컨테이너 목록 조회
        running_containers = set()
        
        # vLLM 컨테이너 확인
        vllm_containers = vllm_service.docker_client.containers.list(
            all=True, 
            filters={"network": vllm_service.network_name, "name": "vllm_gpu_*"}
        )
        for container in vllm_containers:
            running_containers.add(container.id)
            
        # Ollama 컨테이너 확인
        ollama_containers = ollama_service.docker_client.containers.list(
            all=True, 
            filters={"network": ollama_service.network_name, "name": "ollama_*"}
        )
        for container in ollama_containers:
            running_containers.add(container.id)
            
        # DB에서 모든 모델 상태 조회
        all_models = model_crud.get_all_model_states(db)
        
        # 실제 컨테이너가 없는 모델 상태 삭제
        for model in all_models:
            if model.container_id is not None and model.container_id not in running_containers:
                logger.warning(f"컨테이너가 존재하지 않는 모델을 삭제합니다: {model.id}")
                model_crud.delete_model_state(db, model.id)
    except Exception as e:
        logger.error(f"서버 시작 중 오류 발생: {str(e)}")
    finally:
        db.close()

# start_model 엔드포인트 수정
@app.post("/models/start", response_model=ModelResponse)
async def start_model(
    config: Union[OllamaModelConfig, VLLMModelConfig], 
    db: Session = Depends(get_db)
):
    try:
        logger.info(f"Starting model: {config.name} with engine: {config.engine}")
        
        # parameters 처리
        processed_parameters = {}
        if config.parameters:
            processed_parameters = {
                k: v for k, v in config.parameters.items() 
                if not isinstance(v, type)
            }
        
        # 초기 상태로 데이터베이스에 기록
        model_id = f"{config.engine.value}_{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = {
            "id": model_id,
            "status": "starting"
        }
        if config.engine.value == "ollama":
            image = 'ollama/ollama'
        else:
            image = 'vllm/vllm-openai:latest'
        logger.info("config")
        logger.info(config.__dict__)
        model_crud.create_model_state(
            db=db,
            model_id=model_id,
            name=config.name,
            engine_type=config.engine.value,
            status="starting",
            container_id=None,  # 초기에는 컨테이너 ID가 음
            parameters=processed_parameters,
            gpuId=config.gpuId,
            image=image
        )
        # 작업을 큐에 추가
        await model_task_queue.put((config, db, result, "start"))

        return ModelResponse(
            id=result["id"],
            name=model_id,
            status="starting",
            message=f"{config.engine} 모델 {config.name} 시작이 요청되었습니다",
        )
    except Exception as e:
        logger.error(f"모델 시작 요청 중 류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{model_id}/stop", response_model=ModelResponse)
async def stop_model(model_id: str, db: Session = Depends(get_db)):
    try:
        model_state = model_crud.get_model_state(db, model_id)

        if model_state is None:
            raise HTTPException(status_code=404, detail="모델 상태를 찾을 수 없습니다")
        
        if model_state.container_id is None:
            raise HTTPException(status_code=404, detail="컨테이너를 찾을 수 없습니다")
        
        if model_state.engine_type == "ollama":
            result = ollama_service.stop_model(model_state.container_id)
        elif model_state.engine_type == "vllm":
            result = vllm_service.stop_model(model_state.container_id)
        else:
            raise HTTPException(status_code=400, detail="지원하지 않는 모델 타입입니다")

        # 데이터베이스 상태 업데이트
        model_state = model_crud.update_model_status(db, model_id, "stopped")
        if not model_state:
            logger.warning(f"모델 ID {model_id}의 상태 정보를 데이터베이스에서 찾을 수 없습니다")

        return ModelResponse(
            id=result["id"], 
            name=model_state.name,
            status=result["status"], 
            message=f"{model_id} 모델이 성공적으로 중지되었습니다"
        )
    except Exception as e:
        logger.error(f"모델 중지 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_id}", response_model=ModelResponse)
async def remove_model(model_id: str, db: Session = Depends(get_db)):
    try:
        model_state = model_crud.get_model_state(db, model_id)

        if model_state is None:
            raise HTTPException(status_code=404, detail="모델 상태를 찾을 수 없습니다")
        
        if model_state.container_id is None:
            raise HTTPException(status_code=404, detail="컨테이너를 찾을 수 없습니다")
        
        if model_state.engine_type == "ollama":
            result = ollama_service.remove_model(model_state.container_id)
        elif model_state.engine_type == "vllm":
            result = vllm_service.remove_model(model_state.container_id)
        else:
            raise HTTPException(status_code=400, detail="지원하지 않는 모델 타입입니다")

        # 데이터베이스에서 모델 상태 삭제
        model_state = model_crud.delete_model_state(db, model_id)
        if not model_state:
            logger.warning(f"모델 ID {model_id}의 상태 정보를 데이터베이스에서 찾을 수 없습니다")

        return ModelResponse(
            id=result["id"], 
            name=model_state.name,
            status=result["status"], 
            message=f"{model_id} 모델이 성공적으로 제거되었습니다"
        )
    except Exception as e:
        logger.error(f"모델 제거 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_id}/logs")
async def get_logs(model_id: str, db: Session = Depends(get_db)):
    try:
        # 컨테이너 이름으로 타입 확인F
        model_state = model_crud.get_model_state(db, model_id)
        logger.info(model_state.__dict__)
        if not model_state:
            raise HTTPException(status_code=404, detail="모델 상태를 찾을 수 없습니다")
        
        if model_state.container_id is None :
            raise HTTPException(status_code=404, detail="컨테이너를 찾을 수 없습니다")
        container_id = model_state.container_id
        container = None
        try:
            container = ollama_service.docker_client.containers.get(container_id)
            return ollama_service.get_logs(container_id)
        except:
            try:
                container = vllm_service.docker_client.containers.get(container_id)
                return vllm_service.get_logs(container_id)
            except:
                raise HTTPException(status_code=404, detail="컨테이너를 찾을 수 없습니다")
    except Exception as e:
        logger.error(f"로그 조회 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{model_id}/test", response_model=ModelResponse)
async def test_model(model_id: str, prompt: TestPrompt, db: Session = Depends(get_db)):
    try:
        model_state = model_crud.get_model_state(db, model_id)
        if not model_state:
            raise HTTPException(status_code=404, detail="모델 상태를 찾을 수 없습니다")
        
        if model_state.container_id is None:
            raise HTTPException(status_code=404, detail="컨테이너를 찾을 수 없습니다")
        
        logger.info(f"테스트 요청: {model_state.container_id} {model_state.name} {prompt.prompt}")
        
        # 응답을 받을 큐 생성
        response_queue = asyncio.Queue()
        
        try:
            # 테스트 작업을 큐에 추가
            await asyncio.wait_for(
                test_task_queue.put((model_state, prompt, response_queue)), 
                timeout=1.0  # 1초 동안 대기
            )
            
            # 결과 대기
            try:
                response = await asyncio.wait_for(
                    response_queue.get(),
                    timeout=30.0  # 30초 동안 결과 대기
                )
                return response
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=504,
                    detail="테스트 응답 시간이 초과되었습니다"
                )
                
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=503, 
                detail="테스트 큐가 가득 찼습니다. 잠시 후 다시 시도해주세요."
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"테스트 요청 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", response_model=List[ModelInfo])
async def get_models(db: Session = Depends(get_db)):
    try:
        # 데이터베이스에서 모든 모델 상태 조회
        model_states = model_crud.get_all_model_states(db)
        
        # ModelInfo 리스트 생성
        model_list = []
        
        for model_state in model_states:
            try:
                # 컨테이너 상태 확인
                model_status = model_state.status
                if model_state.engine_type == "ollama":
                    container = ollama_service.docker_client.containers.get(model_state.container_id)
                else:  # vllm
                    container = vllm_service.docker_client.containers.get(model_state.container_id)
                
                container_status = model_status
            except Exception as e:
                logger.error(f"컨테이너 상태 조회 중 오류 발생: {str(e)}")
                # 컨테이너를 찾을 수 없는 경우
                container_status = 'stopped'
                logger.warning(f"컨테이너를 찾을 수 없습니다: {model_state.id}")

            model_list.append(
                ModelInfo(
                    id=model_state.id,
                    name=f"{model_state.engine_type}:{model_state.name}",
                    engine=model_state.engine_type,
                    container_id=model_state.container_id,
                    status=container_status
                )
            )

        return model_list

    except Exception as e:
        logger.error(f"모델 목록 조회 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/{model_id}/container", response_model=ContainerInfo)
async def get_container_info(model_id: str, db: Session = Depends(get_db)):
    try:
        # 데이터베이스에서 모델 상태 조회
        model_state = model_crud.get_model_state(db, model_id)
        if not model_state:
            raise HTTPException(status_code=404, detail="모델 상태 정보를 찾을 수 없습니다")
        # model state to dict 
        model_state_dict = model_state.__dict__
        logger.info(model_state_dict)

        # 모델 타입에 따라 테이너 조회
        try:
            container_id = model_state.container_id
            if model_state.engine_type == "ollama":
                container = ollama_service.docker_client.containers.get(container_id)
                port = 11434  # Ollama 기본 포트
                gpu_id = container.attrs["Config"].get("NVIDIA_VISIBLE_DEVICES")
            elif model_state.engine_type == "vllm":
                container = vllm_service.docker_client.containers.get(container_id)
                port = 8000  # vLLM 기본 포트
                gpu_id = next(
                    (
                        env.split("=")[1]
                        for env in container.attrs["Config"]["Env"]
                        if env.startswith("CUDA_VISIBLE_DEVICES=")
                    ),
                    None
                )
            else:
                logger.error(f"지원지 않는 모델 타입입니다")
                raise HTTPException(status_code=400, detail="지원지 않는 모델 타입입니다")
        except Exception as e:
            logger.error(f"컨테이너를 찾을 수 없습니다: {str(e)}")
            raise HTTPException(status_code=404, detail=f"컨테이너를 찾을 수 없습니다: {str(e)}")
        gpu_id = model_state.gpuId
        return ContainerInfo(
            containerId=container.id,
            engine=model_state.engine_type,
            image=container.image.tags[0] if container.image.tags else container.image.id,
            port=port,
            gpuId=gpu_id
        )
            
    except Exception as e:
        logger.error(f"컨테이너 정보 조회 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


class ModelStateResponse(BaseModel):
    id: str
    name: str
    engine_type: str
    status: str
    parameters: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@app.get("/models/{model_id}/state", response_model=ModelStateResponse)
async def get_model_state(model_id: str, db: Session = Depends(get_db)):
    try:
        model_state = model_crud.get_model_state(db, model_id)
        if not model_state:
            raise HTTPException(status_code=404, detail="모델 상태를 찾을 수 없습니다")
        return model_state
    except Exception as e:
        logger.error(f"모델 상태 조회 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/states", response_model=List[ModelStateResponse])
async def get_all_model_states(db: Session = Depends(get_db)):
    try:
        return model_crud.get_all_model_states(db)
    except Exception as e:
        logger.error(f"모델 상태 목록 조 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/init-db")
async def initialize_database():
    try:
        # 기존 테이블 삭제
        Base.metadata.drop_all(bind=engine)
        logger.info("기존 테이블이 삭제되었습니다.")
        
        # 새로운 테이블 생성
        Base.metadata.create_all(bind=engine)
        logger.info("새로운 테이블이 생성되었습니다.")
        
        return {"message": "데이터베이스가 성공적으로 초기화되었습니다."}
    except Exception as e:
        logger.error(f"데이터베이스 초기화 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/{model_id}/restart", response_model=ModelResponse)
async def restart_model(model_id: str, db: Session = Depends(get_db)):
    try:
        # 모델 상태 조회
        model_state = model_crud.get_model_state(db, model_id)
        if not model_state:
            raise HTTPException(status_code=404, detail="모델 상태를 찾을 수 없습니다")

        if model_state.container_id is None:
            raise HTTPException(status_code=404, detail="컨테이너를 찾을 수 없습니다")

        # 상태를 starting으로 업데이트
        model_crud.update_model_status(db, model_id, "starting")

        # 재시작 설정 생성
        config = None
        if model_state.engine_type == "ollama":
            config = OllamaModelConfig(
                name=model_state.name,
                engine=ModelEngineType.OLLAMA,
                parameters=model_state.parameters,
                gpuId=model_state.gpuId,
                usageType=ModelUsageType.GENERATION
            )
        elif model_state.engine_type == "vllm":
            config = VLLMModelConfig(
                name=model_state.name,
                engine=ModelEngineType.VLLM,
                parameters=model_state.parameters,
                gpuId=model_state.gpuId,
                usageType=ModelUsageType.GENERATION
            )
        else:
            raise HTTPException(status_code=400, detail="지원하지 않는 모델 타입입니다")

        # 작업 큐에 재시작 작업 추가
        result = {
            "id": model_id,
            "status": "starting"
        }
        await model_task_queue.put((config, db, result, "restart"))

        return ModelResponse(
            id=model_id,
            name=model_state.name,
            status="starting",
            message=f"{model_state.name} 모델 재시작이 요청되었습니다"
        )

    except Exception as e:
        logger.error(f"모델 재시작 중 오류 발생: {str(e)}")
        return ModelResponse(
            id=model_id,
            name=model_state.name if model_state else "",
            status="error",
            message="모델 재시작 실패",
            error=str(e)
        )

# 모델 상세 정보 응답 클래스 추가
class ModelDetailResponse(BaseModel):
    id: str
    name: str
    type: str
    details: Dict[str, Any]
    status: str

@app.get("/models/{model_id}/info", response_model=ModelDetailResponse)
async def get_model_info(model_id: str, db: Session = Depends(get_db)):
    try:
        # 데이터베이스에서 모델 상태 조회
        model_state = model_crud.get_model_state(db, model_id)
        if not model_state:
            raise HTTPException(status_code=404, detail="모델 상태 정보를 찾을 수 없습니다")

        if model_state.container_id is None:
            raise HTTPException(status_code=404, detail="컨테이너를 찾을 수 없습니다")
        logger.info(model_state.__dict__)
        # 모델 타입에 따라 정보 조회
        try:
            if model_state.engine_type == "ollama":
                result = await ollama_service.get_model_info(model_state.container_id)
            elif model_state.engine_type == "vllm":
                result = await vllm_service.get_model_info(model_state.container_id)
            else:
                raise HTTPException(status_code=400, detail="지원하지 않는 모델 타입입니다")

            return result

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"모델 정보 조회 중 오류 발생: {str(e)}")

    except Exception as e:
        logger.error(f"모델 정보 조회 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
