from sqlalchemy.orm import Session
from models.database import ModelState
from datetime import datetime
from loguru import logger
from typing import Optional

def create_model_state(db: Session, model_id: str, name: str, engine_type: str, status: str, container_id: str, parameters: dict, gpuId: str, image: str):
    existing_model = db.query(ModelState).filter(ModelState.id == model_id).first()
    if existing_model:
        logger.info(f"모델 ID {model_id}가 이미 존재합니다. 기존 상태를 반환합니다.")
        return existing_model
        
    db_model = ModelState(
        id=model_id,
        name=name,
        engine_type=engine_type,
        status=status,
        container_id=container_id,
        gpuId=gpuId,
        image=image,
        parameters=parameters,
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    return db_model

def get_model_state(db: Session, model_id: str):
    return db.query(ModelState).filter(ModelState.id == model_id).first()

def update_model_status(db: Session, model_id: str, status: str, container_id: Optional[str] = None):
    db_model = db.query(ModelState).filter(ModelState.id == model_id).first()
    if db_model:
        if db_model.status == status and (container_id is None or db_model.container_id == container_id):
            logger.info(f"모델 ID {model_id}의 상태가 이미 동일합니다. 업데이트를 건너뜁니다.")
            return db_model
            
        db_model.status = status
        if container_id is not None:
            db_model.container_id = container_id
        db_model.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(db_model)
    return db_model

def delete_model_state(db: Session, model_id: str):
    db_model = db.query(ModelState).filter(ModelState.id == model_id).first()
    if not db_model:
        logger.info(f"모델 ID {model_id}가 이미 삭제되었거나 존재하지 않습니다.")
        return None
        
    db.delete(db_model)
    db.commit()
    return db_model

def get_all_model_states(db: Session):
    return db.query(ModelState).all()

def update_container_id(db: Session, model_id: str, container_id: str):
    db_model = db.query(ModelState).filter(ModelState.id == model_id).first()
    if db_model:
        db_model.container_id = container_id
        db_model.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(db_model)
    return db_model 