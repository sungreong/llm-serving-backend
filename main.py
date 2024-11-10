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
from schemas.model_schemas import ModelEngineType, ModelUsageType, BaseModelConfig, OllamaModelConfig, VLLMModelConfig, ModelConfig, ModelResponse, ModelInfo, TestPrompt, ContainerInfo
from routers.model import router as model_router
import crud.model_crud as model_crud
from fastapi import Depends
from datetime import datetime
from sqlalchemy.orm import Session
import asyncio
from asyncio import Queue
import os
import tarfile
import io

load_dotenv()

app = FastAPI()
app.include_router(model_router)
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


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# 시작 이벤트 수정
@app.on_event("startup")
async def startup_event():
    pass


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

# Nginx 설정 응답 클래스 추가
class NginxConfigResponse(BaseModel):
    success: bool
    message: str
    config: Optional[str] = None

@app.post("/admin/update-nginx", response_model=NginxConfigResponse)
async def update_nginx_config(db: Session = Depends(get_db)):
    try:
        # 실행 중인 모든 모델 상태 조회
        model_states = model_crud.get_all_model_states(db)
        running_models = [m for m in model_states if m.status == "running"]

        # Nginx 설정 템플릿 (서버 블록만 포함)
        nginx_config = """
server {
    listen 80;
    server_name localhost;

    location / {
        return 404 "Model not found";
    }
"""

        # 각 실행 중인 모델에 대한 location 블록 생성
        for model in running_models:
            container_name = None
            internal_port = None
            
            if model.engine_type == "ollama":
                container_name = f"ollama_{model.name.replace(':', '_').replace('/', '_')}"
                internal_port = 11434
            elif model.engine_type == "vllm":
                container_name = f"vllm_gpu_{model.name.replace(':', '_').replace('/', '_')}"
                internal_port = 8000

            if container_name and internal_port:
                location_block = f"""
    location /{model.name}/ {{
        proxy_pass http://{container_name}:{internal_port}/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Connection "";
        
        # CORS 설정 추가
        add_header 'Access-Control-Allow-Origin' '*';
        add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS';
        add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range';
    }}"""
                nginx_config += location_block

        # 설정 파일 닫기
        nginx_config += """
}
"""

        try:
            # Docker 네트워크에서 Nginx 컨테이너 찾기
            nginx_containers = ollama_service.docker_client.containers.list(
                filters={
                    "network": ollama_service.network_name,
                    "name": "llm-serving-nginx"
                }
            )
            print(nginx_containers)
            if not nginx_containers:
                raise HTTPException(status_code=404, detail="Nginx 컨테이너를 찾을 수 없습니다")

            nginx_container = nginx_containers[0]

            # 임시 설정 파일 생성
            temp_config_path = "/tmp/default.conf"  # 파일 이름 변경
            with open(temp_config_path, "w") as f:
                f.write(nginx_config)

            # tar 파일 생성
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                tar.add(temp_config_path, arcname='default.conf')  # arcname 변경
            tar_stream.seek(0)

            # 설정 파일을 Nginx 컨테이너로 복사
            nginx_container.put_archive("/etc/nginx/conf.d/", tar_stream.read())
            # Nginx 설정 테스트
            test_result = nginx_container.exec_run("nginx -t")
            if test_result.exit_code != 0:
                raise Exception(f"Nginx 설정 테스트 실패: {test_result.output.decode()}")

            # Nginx 재시작
            reload_result = nginx_container.exec_run("nginx -s reload")
            if reload_result.exit_code != 0:
                raise Exception(f"Nginx 재시작 실패: {reload_result.output.decode()}")

            # 임시 파일 삭제
            os.remove(temp_config_path)

            return NginxConfigResponse(
                success=True,
                message="Nginx 설정이 성공적으로 업데이트되었습니다.",
                config=nginx_config
            )

        except Exception as e:
            logger.error(f"Nginx 설정 업데이트 중 오류 발생: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Nginx 설정 업데이트 실패: {str(e)}"
            )

    except Exception as e:
        logger.error(f"Nginx 설정 업데이트 중 오류 발생: {str(e)}")
        return NginxConfigResponse(
            success=False,
            message=f"Nginx 설정 업데이트 실패: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
