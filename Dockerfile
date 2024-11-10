FROM python:3.9-slim

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Docker CLI 설치
RUN curl -fsSL https://get.docker.com -o get-docker.sh && \
    sh get-docker.sh && \
    rm get-docker.sh

# Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 스크립트 실행 권한 설정
RUN chmod +x entrypoint.sh

# 환경 변수 설정
ENV HOST=0.0.0.0
ENV PORT=8000

# Entrypoint 설정
ENTRYPOINT ["./entrypoint.sh"]