# LLM Serving Backend

FastAPI 기반의 LLM 모델 서빙 백엔드 서버입니다. Ollama와 vLLM 모델을 관리하고 서빙합니다.

## 시스템 요구사항

- Docker
- NVIDIA GPU (vLLM 사용시)
- NVIDIA Container Toolkit

## 설치 및 실행

### 1. 이미지 빌드
```bash
docker build -t llm-serving-backend .
```

### 2. 컨테이너 실행

기본 실행:
```bash
docker run -d \
  --name llm-serving-backend \
  --network llmmodelgate_internal_network \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v C:\Users\leesu\Documents\ProjectCode\tmp:/hosted/workspace/4_trained_model \
  -p 8000:8000 \
  llm-serving-backend
```

- 윈도우

```
docker run -it `
--name llm-serving-backend `
--network llmmodelgate_internal_network `
-v //var/run/docker.sock:/var/run/docker.sock `
-v C:/Users/leesu/Documents/ProjectCode/tmp:/hosted/workspace/4_trained_model `
-v C:\Users\leesu\Documents\ProjectCode\llm-serving\llm-serving-backend:/app
-p 8000:8000 `
llm-serving-backend bash
```


환경 변수 오버라이드:
```bash
docker run -it \
--name llm-serving-backend \
--network llmmodelgate_internal_network \
-v /var/run/docker.sock:/var/run/docker.sock \
-v C:\Users\leesu\Documents\ProjectCode\tmp:/hosted/workspace/4_trained_model \
-p 8000:8000 \
--gpus all \
llm-serving-backend bash
```

### 3. 로그 확인
```bash
docker logs -f llm-serving-backend
```

## 환경 변수 설정

| 환경 변수 | 설명 | 
|-----------|------|
| DOCKER_NETWORK | Docker 네트워크 이름 | 
| HF_CACHE_PATH | Hugging Face 캐시 경로 | 
| VLLM_SERVING_PATH | vLLM 서빙 경로 | 
| WEIGHTS_PATH | 모델 가중치 경로 | 
| OLLAMA_PATH | Ollama 데이터 경로 | 
| HF_TOKEN | Hugging Face API 토큰 | 
| VLLM_PORT | vLLM 서비스 포트 | 
| OLLAMA_PORT | Ollama 서비스 포트 | 
| DEFAULT_CUDA_DEVICES | 기본 GPU 장치 ID |

## 컨테이너 실행 방법

### 1. 기본 실행 (FastAPI 서버)
```bash
docker run -d \
  --name llm-serving-backend \
  --network llmmodelgate_internal_network \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v C:\Users\leesu\Documents\ProjectCode\tmp:/hosted/workspace/4_trained_model \
  -p 8000:8000 \
  --gpus all \
  llm-serving-backend
```

