#!/bin/bash

# 기본 명령어 설정
DEFAULT_COMMAND="uvicorn main:app --host 0.0.0.0 --port 8000"

# 인자가 전달되지 않았다면 기본 명령어 실행
if [ $# -eq 0 ]; then
    echo "Running default command: $DEFAULT_COMMAND"
    $DEFAULT_COMMAND &
    tail -f /dev/null
else
    # 인자가 전달되었다면 해당 명령어 실행
    echo "Running custom command: $@"
    exec "$@"
fi 