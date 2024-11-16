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
class TGIParameters:
    """Text Generation Inference 파라미터 설정을 위한 데이터 클래스"""
    # 모델 관련 기본 설정
    # max_input_length: Optional[int] = 2048
    # max_total_tokens: Optional[int] = 4096
    # max_batch_size: Optional[int] = 32
    # max_concurrent_requests: Optional[int] = 128
    
    # # 배치 처리 관련 설정
    # waiting_served_ratio: Optional[float] = 0.3
    # max_batch_total_tokens: Optional[int] = 32768
    # max_waiting_tokens: Optional[int] = 20
    
    # # 하드웨어 관련 설정
    # cuda_memory_fraction: Optional[float] = 1.0
    # disable_custom_kernels: Optional[bool] = False
    
    # # 모델 최적화 관련 설정
    # rope_scaling: Optional[str] = "linear"  # 'linear' 또는 'dynamic'
    # rope_factor: Optional[float] = 1.0
    # quantize: Optional[str] = None  # 'bitsandbytes', 'gptq', 'awq' 등
    # dtype: Optional[str] = None  # 'float16', 'bfloat16'
    
    # # 추론 관련 설정
    # max_best_of: Optional[int] = 2
    # max_stop_sequences: Optional[int] = 4
    # max_top_n_tokens: Optional[int] = 5
    pass 


    def to_dict(self) -> Dict[str, Any]:
        """파라미터를 딕셔너리로 변환 (None이 아닌 값만)"""
        return {k: v for k, v in self.__dict__.items() if v is not None}

class HuggingFaceServiceTGI:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.network_name = os.getenv("DOCKER_NETWORK")
        self.cache_path = os.getenv("HF_CACHE_PATH")
        self.default_hf_token = os.getenv("HF_TOKEN")
        self.internal_port = int(os.getenv("HF_PORT", "8080"))
        self.default_parameters = TGIParameters()

    def _create_safe_container_name(self, model_name: str) -> str:
        return f"tgi_{model_name.replace(':', '_').replace('/', '_')}"

    def _prepare_network(self):
        try:
            self.docker_client.networks.get(self.network_name)
        except docker.errors.NotFound:
            logger.info(f"Creating network: {self.network_name}")
            self.docker_client.networks.create(self.network_name, driver="bridge", attachable=True)

    def _build_command(self, model_name: str, parameters: Dict[str, Any]) -> List[str]:
        # 기본 파라미터 객체 생성
        tgi_params = TGIParameters(**parameters)
        
        # 명령어 리스트 시작
        command = ["--model-id", model_name]
        
        # 파라미터 딕셔너리로 변환
        param_dict = tgi_params.to_dict()
        
        # 파라미터를 명령어로 변환
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
            print(f"container_name: {container_name}")
            # 기존 컨테이너 확인
            existing_containers = self.docker_client.containers.list(all=True, filters={"name": container_name})

            if existing_containers:
                container = existing_containers[0]
                if container.status != "running":
                    await asyncio.to_thread(container.start)
                    logger.info(f"Started existing container for {model_name}")
            else:
                # 사용자 파라미터와 기본 파라미터 병합
                user_params = config.parameters if hasattr(config, 'parameters') else {}
                
                # TGIParameters의 기본값을 사용하면서 사용자 파라미터로 덮어쓰기
                merged_params = {**self.default_parameters.to_dict(), **user_params}
                
                # config 객체 업데이트
                config.parameters = merged_params
                
                # 환경 변수 설정
                environment = {
                    "HF_TOKEN": config.parameters.get("hf_token", self.default_hf_token),
                    "HF_API_TOKEN": config.parameters.get("hf_token", self.default_hf_token),
                    "HUGGING_FACE_HUB_TOKEN": config.parameters.get("hf_token", self.default_hf_token),
                }

                # GPU 설정
                gpu_config = config.gpuId
                # device_requests = (
                #     [docker.types.DeviceRequest(device_ids=gpu_config, capabilities=[["gpu"]])] if gpu_config else []
                # )
                device_requests = []

                # 볼륨 설정
                volumes = {
                    self.cache_path: {"bind": "/data", "mode": "rw"},
                }

                # 컨테이너 생성
                container_args = {
                    "image": "ghcr.io/huggingface/text-generation-inference:2.4.0",
                    "command": self._build_command(model_name, config.parameters),
                    "detach": True,
                    "name": container_name,
                    "environment": environment,
                    "volumes": volumes,
                    "device_requests": device_requests,
                    "runtime": "nvidia" if device_requests else None,
                    "network": self.network_name,
                    # more big shz_size
                    "shm_size": "10g",
                    
                }
                logger.info(f"container_args: {container_args}")
                print(f"container_args: {container_args}"   )
                container = await asyncio.to_thread(self.docker_client.containers.run, **container_args)
                logger.info(f"Created new container for {model_name}")

            # 모델 준비 상태 확인
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
            logger.error(f"Error starting HuggingFace model {model_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def test_model(self, container_id: str, model_name: str, prompt: str) -> Dict[str, Any]:
        try:
            container = self.docker_client.containers.get(container_id)
            if container.status != "running":
                raise HTTPException(status_code=400, detail="Model container is not running")

            container_name = container.name
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{container_name}:{self.internal_port}/generate",
                    json={
                        "inputs": prompt,
                        "parameters": {"max_new_tokens": 100, "temperature": 0.7, "do_sample": True},
                    },
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "id": container_id,
                            "name": container_name,
                            "status": "success",
                            "response": result.get("generated_text", ""),
                        }
                    else:
                        raise HTTPException(status_code=response.status, detail="Failed to get response from model")

        except Exception as e:
            logger.error(f"Error testing HuggingFace model {container_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    # 기타 필요한 메서드들 (stop_model, remove_model, get_logs 등)은
    # vllm_service.py와 유사한 방식으로 구현...

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
                            "type": "huggingface",
                            "details": model_info,
                            "status": "running",
                        }
                    else:
                        raise HTTPException(status_code=response.status, detail="Failed to get model information")

        except Exception as e:
            logger.error(f"Error getting HuggingFace model info {container_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


