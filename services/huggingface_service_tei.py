import docker
from loguru import logger
from typing import Optional, Dict, Any, List, Union
import os
from fastapi import HTTPException
from dotenv import load_dotenv
import aiohttp
import asyncio
from dataclasses import dataclass

load_dotenv()

@dataclass
class TEIParameters:
    """Text Embeddings Inference 파라미터 설정을 위한 데이터 클래스"""
    # 모델 관련 기본 설정
    # max_client_batch_size: Optional[int] = 32
    # max_concurrent_requests: Optional[int] = 512
    # max_batch_tokens: Optional[int] = 16384
    # max_batch_requests: Optional[int] = None
    # tokenization_workers: Optional[int] = None
    
    # # 모델 최적화 관련 설정
    # dtype: Optional[str] = None  # 'float16', 'float32'
    
    # # 추론 관련 설정
    # pooling: Optional[str] = None  # 'cls', 'mean', 'splade', 'last-token'
    # auto_truncate: Optional[bool] = False
    
    # # 프롬프트 관련 설정
    # default_prompt_name: Optional[str] = None
    # default_prompt: Optional[str] = None
    
    # # API 관련 설정
    # payload_limit: Optional[int] = 2000000  # 2MB
    api_key: Optional[str] = None
    hf_api_token: Optional[str] = os.getenv("HF_API_TOKEN")
    
    def to_dict(self) -> Dict[str, Any]:
        """파라미터를 딕셔너리로 변환 (None이 아닌 값만)"""
        return {k: v for k, v in self.__dict__.items() if v is not None}

class HuggingFaceServiceTEI:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.network_name = os.getenv("DOCKER_NETWORK")
        self.cache_path = os.getenv("HF_CACHE_PATH")
        self.default_hf_token = os.getenv("HF_TOKEN")
        self.internal_port = int(os.getenv("HF_PORT", "11434"))
        self.default_parameters = TEIParameters()

    def _create_safe_container_name(self, model_name: str) -> str:
        return f"tei_{model_name.replace(':', '_').replace('/', '_')}"

    def _prepare_network(self):
        try:
            self.docker_client.networks.get(self.network_name)
        except docker.errors.NotFound:
            logger.info(f"Creating network: {self.network_name}")
            self.docker_client.networks.create(self.network_name, driver="bridge", attachable=True)

    def _build_command(self, model_name: str, parameters: Dict[str, Any]) -> List[str]:
        tei_params = TEIParameters(**parameters)
        command = ["--model-id", model_name , "--port", str(self.internal_port)]
        
        param_dict = tei_params.to_dict()
        for key, value in param_dict.items():
            key = key.replace('_', '-')
            if isinstance(value, bool):
                if value:
                    command.append(f"--{key}")
            else:
                command.extend([f"--{key}", str(value)])
        
        return command

    async def start_model(self, model_name: str, config: Any) -> Dict[str, Any]:
        try:
            self._prepare_network()
            container_name = self._create_safe_container_name(model_name)

            existing_containers = self.docker_client.containers.list(all=True, filters={"name": container_name})

            if existing_containers:
                container = existing_containers[0]
                if container.status != "running":
                    await asyncio.to_thread(container.start)
                    logger.info(f"Started existing container for {model_name}")
            else:
                user_params = config.parameters if hasattr(config, 'parameters') else {}
                merged_params = {**self.default_parameters.to_dict(), **user_params}
                config.parameters = merged_params
                
                environment = {
                    "HF_TOKEN": config.parameters.get("hf_token", self.default_hf_token),
                    "HUGGING_FACE_HUB_TOKEN": config.parameters.get("hf_token", self.default_hf_token),
                }

                gpu_config = config.gpuId
                # device_requests = (
                #     [docker.types.DeviceRequest(device_ids=gpu_config, capabilities=[["gpu"]])] if gpu_config else []
                # )
                device_requests =[]
                volumes = {
                    self.cache_path: {"bind": "/data", "mode": "rw"},
                }
                if len(device_requests) == 0:
                    image = "ghcr.io/huggingface/text-embeddings-inference:cpu-latest"
                else :
                    image = "ghcr.io/huggingface/text-embeddings-inference:1.5"
                    
                container_args = {
                    "image": image,
                    "command": self._build_command(model_name, config.parameters),
                    "detach": True,
                    "name": container_name,
                    "environment": environment,
                    "volumes": volumes,
                    "device_requests": device_requests,
                    "runtime": "nvidia" if device_requests else None,
                    "network": self.network_name,
                    "shm_size": "10g",
                }
                logger.info(f"container_args: {container_args}")
                logger.info(f"image: {image}")  
                container = await asyncio.to_thread(self.docker_client.containers.run, **container_args)
                logger.info(f"Created new container for {model_name}")

            max_retries = 30
            retry_interval = 10

            async with aiohttp.ClientSession() as session:
                for attempt in range(max_retries):
                    try:
                        async with session.get(
                            f"http://{container_name}:{self.internal_port}/health", timeout=20
                        ) as response:
                            if response.status == 200:
                                break
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        if attempt == max_retries - 1:
                            raise HTTPException(
                                status_code=500, detail=f"Failed to start {model_name} after {max_retries} attempts"
                            )
                        await asyncio.sleep(retry_interval)
    
            return {"id": container.id, "name": container_name, "status": "running", "port": self.internal_port}

        except Exception as e:
            logger.error(f"Error starting Text Embeddings model {model_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_embeddings(self, container_id: str, texts: Union[str, List[str]]) -> Dict[str, Any]:
        try:
            container = self.docker_client.containers.get(container_id)
            if container.status != "running":
                raise HTTPException(status_code=400, detail="Model container is not running")

            container_name = container.name
            
            # 단일 텍스트를 리스트로 변환
            if isinstance(texts, str):
                texts = [texts]

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{container_name}:{self.internal_port}/embed",
                    json={"inputs": texts}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "id": container_id,
                            "name": container_name,
                            "status": "success",
                            "embeddings": result
                        }
                    else:
                        raise HTTPException(status_code=response.status, detail="Failed to get embeddings from model")

        except Exception as e:
            logger.error(f"Error getting embeddings from model {container_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_model_info(self, container_id: str) -> Dict[str, Any]:
        try:
            container = self.docker_client.containers.get(container_id)
            if container.status != "running":
                raise HTTPException(status_code=400, detail="Model container is not running")

            container_name = container.name
            async with aiohttp.ClientSession() as session:
                async with session.get(f"http://{container_name}:{self.internal_port}/info", timeout=20) as response:
                    if response.status == 200:
                        model_info = await response.json()
                        return {
                            "id": container_id,
                            "name": container_name,
                            "type": "text-embeddings",
                            "details": model_info,
                            "status": "running",
                        }
                    else:
                        raise HTTPException(status_code=response.status, detail="Failed to get model information")

        except Exception as e:
            logger.error(f"Error getting Text Embeddings model info {container_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def test_model_rerank(self, container_id: str, model_name: str, prompt: str, texts: List[str]) -> Dict[str, Any]:
        try :
            container = self.docker_client.containers.get(container_id)
            if container.status != "running":
                raise HTTPException(status_code=400, detail="Model container is not running")

            container_name = container.name
            
            # 단일 텍스트를 리스트로 변환
            if isinstance(texts, str):
                texts = [texts]

            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{container_name}:{self.internal_port}/rerank",
                    json={"query": prompt, "texts": texts}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "id": container_id,
                            "name": container_name,
                            "status": "success",
                            "rerank": result
                        }
                    else:
                        raise HTTPException(status_code=response.status, detail="Failed to get embeddings from model")
        except Exception as e:
            logger.error(f"Error getting rerank task: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        

    async def remove_model(self, container_id: str) -> Dict[str, Any]:
        try :
            try:
                container = self.docker_client.containers.get(container_id)
                await asyncio.to_thread(container.remove, force=True)
            except docker.errors.NotFound:
                logger.info(f"Container {container_id} already removed")
                # 이미 제거된 경우도 성공으로 처리
                return {"id": container_id, "status": "removed"}
            except Exception as e:
                logger.error(f"Error removing Text Embeddings model {container_id}: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
            return {"id": container_id, "status": "removed"}
        except Exception as e:
            logger.error(f"Error removing Text Embeddings model {container_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_logs(self, container_id: str) -> Dict[str, list]:
        try:
            container = self.docker_client.containers.get(container_id)
            logs = container.logs().decode("utf-8").split("\n")
            return {"logs": logs}
        except Exception as e:
            logger.error(f"Error getting logs for Text Embeddings model {container_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        
    async def stop_model(self, container_id: str) -> Dict[str, str]:
        try:
            container = self.docker_client.containers.get(container_id)
            await asyncio.to_thread(container.stop)
            return {"id": container_id, "status": "stopped"}
        except Exception as e:
            logger.error(f"Error stopping Text Embeddings model {container_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def restart_model(self, model_name: str, parameters: Dict[Any, Any] = None) -> Dict[str, Any]:
        try :
            parameters = parameters.parameters
            container_name = self._create_safe_container_name(model_name)
            container = self.docker_client.containers.get(container_name)
            
            existing_containers = self.docker_client.containers.list(all=True, 
                                                                     filters={"name": container_name,
                                                                              "network": self.network_name})
            if existing_containers:
                container = existing_containers[0]
                await asyncio.to_thread(container.remove, force=True)
                logger.info(f"Removed existing container for {model_name}")

            environment = {
                "HF_TOKEN": parameters.get("hf_token", self.default_hf_token),
                "HUGGING_FACE_HUB_TOKEN": parameters.get("hf_token", self.default_hf_token),
            }
            logger.info(f"environment: {environment}")  
            container = await asyncio.to_thread(self.docker_client.containers.run,
                                                image="latest",
                                                command=self._build_command(model_name, parameters),
                                                environment=environment,
                                                name=container_name,
                                                network=self.network_name)
            return {"id": container.id, "name": container_name, "status": "restarted"}
        except Exception as e:
            logger.error(f"Error restarting Text Embeddings model {model_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))