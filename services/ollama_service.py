import docker
from loguru import logger
from typing import Optional, Dict, Any, Union
import time
import requests
import os
from fastapi import HTTPException
from dotenv import load_dotenv
import aiohttp
import asyncio
import json

load_dotenv()


class OllamaService:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.network_name = os.getenv("DOCKER_NETWORK")
        self.ollama_path = os.getenv("OLLAMA_PATH")
        self.internal_port = int(os.getenv("OLLAMA_PORT", "11434"))

    def _create_safe_container_name(self, model_name: str) -> str:
        return f"ollama_{model_name.replace(':', '_').replace('/', '_')}"

    def _prepare_network(self):
        try:
            self.docker_client.networks.get(self.network_name)
        except docker.errors.NotFound:
            logger.info(f"Creating network: {self.network_name}")
            self.docker_client.networks.create(self.network_name, driver="bridge", attachable=True)

    async def start_model(self, model_name: str, config: Any) -> Dict[str, Any]:
        try:
            self._prepare_network()
            container_name = self._create_safe_container_name(model_name)

            # 기존 컨테이너 확인
            existing_containers = self.docker_client.containers.list(
                all=True,
                filters={
                    "name": container_name,
                    "network": self.network_name,
                },
            )

            if existing_containers:
                logger.info(f"Found existing container for {model_name}")
                container = existing_containers[0]
                if container.status != "running":
                    container.start()
                    logger.info(f"Started existing container for {model_name}")
            else:
                # 새 컨테이너 생성
                environment = {"OLLAMA_KEEP_ALIVE": "-1"}
                if config.parameters:
                    environment.update(config.parameters)
                
                # GPU 설정
                logger.info(f'start model config : {config.__dict__}')
                gpu_config = config.gpuId  # config['gpuId'] 대신 config.gpuId 사용
                
                # GPU 설정에 따른 device_requests 구성
                if gpu_config:
                    device_requests = [
                        docker.types.DeviceRequest(
                            device_ids=gpu_config,
                            capabilities=[['gpu']]
                        )
                    ]
                else:
                    # CPU 사용
                    logger.info(f"CPU 사용")
                    device_requests = []
                
                logger.info(f"Creating new container for {model_name} with GPU config: {gpu_config}")
                
                container = self.docker_client.containers.run(
                    "ollama/ollama",
                    detach=True,
                    environment=environment,
                    volumes={self.ollama_path: {"bind": "/root/.ollama", "mode": "rw"}},
                    name=container_name,
                    network=self.network_name,
                    device_requests=device_requests,
                    runtime='nvidia' if device_requests else None
                )
                logger.info(f"Created new container for {model_name}")

            # 모델 로드 (비동기로 실행)
            await asyncio.to_thread(container.exec_run, f"ollama run {model_name}")

            # 모델 준비 상태 확인 (비동기로 실행)
            max_retries = 30
            retry_interval = 10

            async with aiohttp.ClientSession() as session:
                for attempt in range(max_retries):
                    try:
                        async with session.get(
                            f"http://{container_name}:{self.internal_port}/api/tags",
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
            logger.error(f"Error starting Ollama model {model_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def stop_model(self, container_id: str) -> Dict[str, str]:
        try:
            container = self.docker_client.containers.get(container_id)
            container.stop()
            return {"id": container_id, "status": "stopped"}
        except Exception as e:
            logger.error(f"Error stopping Ollama container {container_id}: {str(e)}")
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
                logger.error(f"Error removing Ollama container {container_id}: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
                
            return {"id": container_id, "status": "removed"}
        except Exception as e:
            logger.error(f"Error removing Ollama container {container_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def get_logs(self, container_id: str) -> Dict[str, list]:
        try:
            container = self.docker_client.containers.get(container_id)
            logs = container.logs().decode("utf-8").split("\n")
            return {"logs": logs}
        except Exception as e:
            logger.error(f"Error getting logs for Ollama container {container_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def test_model(self, container_id: str, model_name: str, prompt: str) -> Dict[str, Any]:
        try:
            container = self.docker_client.containers.get(container_id)
            if container.status != "running":
                raise HTTPException(status_code=400, detail="Model container is not running")

            container_name = container.name
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{container_name}:{self.internal_port}/api/generate",
                    json={
                        "prompt": prompt,
                        "model": model_name,
                        "stream": False,
                    }
                ) as response:
                    if response.status == 200:
                        # ndjson 응답 처리
                        full_response = ""
                        async for line in response.content:
                            if line:
                                # 각 라인을 디코드하고 응답에 추가
                                decoded_line = line.decode('utf-8').strip()
                                if decoded_line:
                                    full_response += decoded_line
                        # 최종 응답 반환
                        json_response = json.loads(full_response)
                        return {
                            "id": container_id, 
                            "name": container_name,                    
                            "status": "success", 
                            "response": json_response.get("response", "Test completed successfully")
                        }
                    else:
                        raise HTTPException(
                            status_code=response.status,
                            detail="Failed to get response from model"
                        )

        except Exception as e:
            logger.error(f"Error testing Ollama model {container_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def test_model_embedding(self, container_id: str, model_name: str, prompt: str) -> Dict[str, Any]:
        try :
            container = self.docker_client.containers.get(container_id)
            if container.status != "running":
                raise HTTPException(status_code=400, detail="Model container is not running")
            container_name = container.name
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{container_name}:{self.internal_port}/api/embed",
                    json={
                        "input": prompt,
                        "model": model_name,
                    },
                    timeout=20
                ) as response:
                    if response.status == 200:
                        json_response = await response.json()
                        return {
                            "id": container_id, 
                            "name": container_name,                    
                            "status": "success", 
                            "response": str(json_response.get("embeddings"))
                        }
                    else:
                        raise HTTPException(status_code=response.status, detail="Failed to get response from model")    
                    
        except Exception as e:
            logger.error(f"Error testing Ollama model {container_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def restart_model(self, model_name: str, parameters: Dict[Any, Any] = None) -> Dict[str, Any]:
        try:
            container_name = self._create_safe_container_name(model_name)
            
            # 기존 컨테이너 찾기 및 제거
            existing_containers = self.docker_client.containers.list(
                all=True,
                filters={
                    "name": container_name,
                    "network": self.network_name,
                },
            )
            
            if existing_containers:
                container = existing_containers[0]
                await asyncio.to_thread(container.remove, force=True)
                logger.info(f"Removed existing container for {model_name}")

            # 새 컨테이너 생성
            environment = {"OLLAMA_KEEP_ALIVE": "-1"}
            if parameters:
                environment.update(parameters)
            logger.info(f"Creating new container for {model_name}")
            container = await asyncio.to_thread(
                self.docker_client.containers.run,
                "ollama/ollama",
                detach=True,
                environment=environment,
                volumes={self.ollama_path: {"bind": "/root/.ollama", "mode": "rw"}},
                name=container_name,
                network=self.network_name,
            )
            logger.info(f"Created new container for {model_name}")

            # 모델 로드
            await asyncio.to_thread(container.exec_run, f"ollama run {model_name}")

            # 모델 준비 상태 확인
            max_retries = 30
            retry_interval = 10

            async with aiohttp.ClientSession() as session:
                for attempt in range(max_retries):
                    try:
                        async with session.get(
                            f"http://{container_name}:{self.internal_port}/api/tags",
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
            logger.error(f"Error restarting Ollama model {model_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_model_info(self, container_id: str) -> Dict[str, Any]:
        try:
            container = self.docker_client.containers.get(container_id)
            if container.status != "running":
                raise HTTPException(status_code=400, detail="Model container is not running")

            # 컨테이너 이름에서 모델 이름 추출
            container_name = container.name
            # ollama_llama3.2_1b container_name
            model_name = container_name.replace("ollama_", "")
            # count "_"
            underscore_count = model_name.count("_")
            if underscore_count == 1:
                # maybe :1b
                model_name = model_name.replace("_", ":", 1)
            else :
                # maybe vlllm 
                model_name = model_name.replace("_", "/", 1).replace("_", ":", 1)   
            # model_name = llama3.2:1b
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"http://{container_name}:{self.internal_port}/api/show",
                    json={"name": model_name},
                    timeout=20
                ) as response:
                    if response.status == 200:
                        model_info = await response.json()
                        return {
                            "id": container_id,
                            "name": container_name,
                            "type": "ollama",
                            "details": model_info,
                            "status": "running"
                        }
                    else:
                        raise HTTPException(
                            status_code=response.status,
                            detail="Failed to get model information"
                        )

        except Exception as e:
            logger.error(f"Error getting Ollama model info {container_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
