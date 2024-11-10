import sys
import os

# 상위 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.database import Base, engine
from loguru import logger

def init_db():
    try:
        # 기존 테이블 삭제
        Base.metadata.drop_all(bind=engine)
        logger.info("기존 테이블이 삭제되었습니다.")
        
        # 새로운 테이블 생성
        Base.metadata.create_all(bind=engine)
        logger.info("새로운 테이블이 생성되었습니다.")
        
        return True
    except Exception as e:
        logger.error(f"데이터베이스 초기화 중 오류 발생: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("데이터베이스 초기화를 시작합니다...")
    success = init_db()
    if success:
        logger.info("데이터베이스 초기화가 완료되었습니다.")
    else:
        logger.error("데이터베이스 초기화에 실패했습니다.") 