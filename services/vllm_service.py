import docker
from loguru import logger
from typing import Optional, Dict, Any, List
import time
import os
from fastapi import HTTPException
from dotenv import load_dotenv
import aiohttp
import asyncio

load_dotenv()


class VLLMService:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.network_name = os.getenv("DOCKER_NETWORK")
        self.hf_cache_path = os.getenv("HF_CACHE_PATH")
        self.vllm_serving_path = os.getenv("VLLM_SERVING_PATH")
        self.weights_path = os.getenv("WEIGHTS_PATH")
        self.default_hf_token = os.getenv("HF_TOKEN")
        self.internal_port = int(os.getenv("VLLM_PORT", "11434"))
        self.default_cuda_devices = os.getenv("DEFAULT_CUDA_DEVICES")

    def _create_safe_container_name(self, model_name: str) -> str:
        return f"vllm_gpu_{model_name.replace(':', '_').replace('/', '_')}"

    def _prepare_network(self):
        try:
            self.docker_client.networks.get(self.network_name)
        except docker.errors.NotFound:
            logger.info(f"Creating network: {self.network_name}")
            self.docker_client.networks.create(self.network_name, driver="bridge", attachable=True)

    def _build_command(self, model_name: str, user_arguments: Dict[str, Any]) -> List[str]:
        command = ["--model", model_name, "--host", "0.0.0.0", "--port", str(self.internal_port)]

        for key, value in user_arguments.items():
            if value is not None:
                if isinstance(value, bool):
                    if value:
                        command.append(key)
                else:
                    command.append(key)
                    command.append(str(value))

        return command

    async def start_model(self, model_name: str, parameters: Dict[Any, Any] = None) -> Dict[str, Any]:
        try:
            self._prepare_network()
            container_name = self._create_safe_container_name(model_name)

            # 기존 컨테이너 확인
            existing_containers = self.docker_client.containers.list(all=True, filters={"name": container_name})

            if existing_containers:
                container = existing_containers[0]
                if container.status != "running":
                    await asyncio.to_thread(container.start)
                    logger.info(f"Started existing container for {model_name}")
            else:
                volumes = {
                    self.hf_cache_path: {"bind": "/root/.cache/huggingface", "mode": "rw"},
                    self.vllm_serving_path: {"bind": "/app/vllm_serving", "mode": "rw"},
                    self.weights_path: {"bind": "/app/weights", "mode": "rw"},
                }

                environment = {
                    "TZ": "Asia/Seoul",
                    "HUGGING_FACE_HUB_TOKEN": parameters.get("hf_token", self.default_hf_token),
                    "CUDA_VISIBLE_DEVICES": parameters.get("device_ids", self.default_cuda_devices),
                }

                command = self._build_command(model_name, parameters.get("model_args", {}))

                container_args = {
                    "image": "vllm/vllm-openai:latest",
                    "command": command,
                    "detach": True,
                    "name": container_name,
                    "volumes": volumes,
                    "environment": environment,
                    "runtime": "nvidia",
                    "ipc_mode": "host",
                    "network": self.network_name,
                }

                container = await asyncio.to_thread(
                    self.docker_client.containers.run,
                    **container_args
                )
                logger.info(f"Created new container for {model_name}")

            # 모델 준비 상태 확인
            max_retries = 30
            retry_interval = 10

            async with aiohttp.ClientSession() as session:
                for attempt in range(max_retries):
                    try:
                        async with session.get(
                            f"http://{container_name}:{self.internal_port}/v1/models",
                            timeout=20
                        ) as response:
                            if response.status == 200:
                                break
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        if attempt == max_retries - 1:
                            raise HTTPException(
                                status_code=500,
                                detail=f"Failed to start {model_name} after {max_retries} attempts"
                            )
                        await asyncio.sleep(retry_interval)

            return {"id": container.id, "name": container_name, "status": "running", "port": self.internal_port}

        except Exception as e:
            logger.error(f"Error starting vLLM model {model_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def stop_model(self, container_id: str) -> Dict[str, str]:
        try:
            container = self.docker_client.containers.get(container_id)
            await asyncio.to_thread(container.stop)
            return {"id": container_id, "status": "stopped"}
        except Exception as e:
            logger.error(f"Error stopping vLLM container {container_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def remove_model(self, container_id: str) -> Dict[str, str]:
        try:
            try:
                container = self.docker_client.containers.get(container_id)
                container.remove(force=True)
            except docker.errors.NotFound:
                logger.info(f"Container {container_id} already removed")
                # 이미 제거된 경우도 성공으로 처리
                return {"id": container_id, "status": "removed"}
            except Exception as e:
                logger.error(f"Error removing VLLM container {container_id}: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
                
            return {"id": container_id, "status": "removed"}
        except Exception as e:
            logger.error(f"Error removing VLLM container {container_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_logs(self, container_id: str) -> Dict[str, list]:
        try:
            container = self.docker_client.containers.get(container_id)
            logs = await asyncio.to_thread(lambda: container.logs().decode("utf-8").split("\n"))
            return {"logs": logs}
        except Exception as e:
            logger.error(f"Error getting logs for vLLM container {container_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def test_model(self, container_id: str,model_name:str, prompt: str) -> Dict[str, Any]:
        try:
            container = self.docker_client.containers.get(container_id)
            if container.status != "running":
                raise HTTPException(status_code=400, detail="Model container is not running")

            container_name = container.name
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{container_name}:{self.internal_port}/v1/completions",
                    json={"prompt": prompt, 
                        "model" : model_name,
                        "max_tokens": 100, 
                        "temperature": 0.7}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {"id": container_id,
                        "name": container_name,                    
                         "status": "success", "response": result['choices'][-1]['text']}
                    else:
                        raise HTTPException(
                            status_code=response.status,
                            detail="Failed to get response from model"
                        )

        except Exception as e:
            logger.error(f"Error testing vLLM model {container_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def test_model_embedding(self, container_id: str, model_name: str, prompt: str) -> Dict[str, Any]:
        try:
            container = self.docker_client.containers.get(container_id)
            if container.status != "running":
                raise HTTPException(status_code=400, detail="Model container is not running")

            container_name = container.name
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{container_name}:{self.internal_port}/v1/embeddings",
                    json={"input": prompt, "model": model_name},
                    timeout=20
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {"id": container_id, "name": container_name, "status": "success", "response": str(result['data'][-1]['embedding'])}
                    else:
                        raise HTTPException(status_code=response.status, detail="Failed to get response from model")
        except Exception as e:
            logger.error(f"Error testing vLLM model embedding {container_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def restart_model(self, model_name: str, parameters: Dict[Any, Any] = None) -> Dict[str, Any]:
        try:
            container_name = self._create_safe_container_name(model_name)
            
            # 기존 컨테이너 찾기 및 제거
            existing_containers = self.docker_client.containers.list(
                all=True,
                filters={"name": container_name}
            )
            
            if existing_containers:
                container = existing_containers[0]
                await asyncio.to_thread(container.remove, force=True)
                logger.info(f"Removed existing container for {model_name}")

            # 새 컨테이너 설정
            volumes = {
                self.hf_cache_path: {"bind": "/root/.cache/huggingface", "mode": "rw"},
                self.vllm_serving_path: {"bind": "/app/vllm_serving", "mode": "rw"},
                self.weights_path: {"bind": "/app/weights", "mode": "rw"},
            }

            environment = {
                "TZ": "Asia/Seoul",
                "HUGGING_FACE_HUB_TOKEN": parameters.get("hf_token", self.default_hf_token),
                "CUDA_VISIBLE_DEVICES": parameters.get("device_ids", self.default_cuda_devices),
            }

            command = self._build_command(model_name, parameters.get("model_args", {}))

            container_args = {
                "image": "vllm/vllm-openai:latest",
                "command": command,
                "detach": True,
                "name": container_name,
                "volumes": volumes,
                "environment": environment,
                "runtime": "nvidia",
                "ipc_mode": "host",
                "network": self.network_name,
            }

            # 새 컨테이너 생성
            container = await asyncio.to_thread(
                self.docker_client.containers.run,
                **container_args
            )
            logger.info(f"Created new container for {model_name}")

            # 모델 준비 상태 확인
            max_retries = 30
            retry_interval = 10

            async with aiohttp.ClientSession() as session:
                for attempt in range(max_retries):
                    try:
                        async with session.get(
                            f"http://{container_name}:{self.internal_port}/v1/models",
                            timeout=20
                        ) as response:
                            if response.status == 200:
                                break
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        if attempt == max_retries - 1:
                            raise HTTPException(
                                status_code=500,
                                detail=f"Failed to restart {model_name} after {max_retries} attempts"
                            )
                        await asyncio.sleep(retry_interval)

            return {"id": container.id, "name": container_name, "status": "running", "port": self.internal_port}

        except Exception as e:
            logger.error(f"Error restarting vLLM model {model_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_model_info(self, container_id: str) -> Dict[str, Any]:
        try:
            container = self.docker_client.containers.get(container_id)
            if container.status != "running":
                raise HTTPException(status_code=400, detail="Model container is not running")

            container_name = container.name
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{container_name}:{self.internal_port}/v1/models",
                    timeout=20
                ) as response:
                    if response.status == 200:
                        model_info = await response.json()
                        return {
                            "id": container_id,
                            "name": container_name,
                            "type": "vllm",
                            "details": model_info,
                            "status": "running"
                        }
                    else:
                        raise HTTPException(
                            status_code=response.status,
                            detail="Failed to get model information"
                        )

        except Exception as e:
            logger.error(f"Error getting vLLM model info {container_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
