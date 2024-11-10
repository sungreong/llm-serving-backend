from pydantic import BaseModel
from typing import Optional, Dict, Any, Union, List
from enum import Enum
from datetime import datetime


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
    embeddings: Optional[str] = None


class EmbeddingResponse(BaseModel):
    id: str
    name: str
    status: str
    embeddings: str

class ModelInfo(BaseModel):
    id: str
    name: str
    engine: str
    status: str
    container_id: Optional[str] = None
    usageType: str


class TestPrompt(BaseModel):
    prompt: str


# 컨테이너 정보 클래스
class ContainerInfo(BaseModel):
    containerId: str
    engine: str
    image: str
    port: int
    gpuId: Optional[str] = None



class ModelStateResponse(BaseModel):
    id: str
    name: str
    engine_type: str
    status: str
    parameters: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


# 모델 상세 정보 응답 클래스 추가
class ModelDetailResponse(BaseModel):
    id: str
    name: str
    type: str
    details: Dict[str, Any]
    status: str


# 모델 서빙 정보 응답 클래스 추가
class ModelServingInfo(BaseModel):
    id: str
    name: str
    port: Optional[int] = None
    status: str
    engine: str
    usageType: str
    nginxEnabled: Optional[bool] = None
    servingUrl: Optional[str] = None
    