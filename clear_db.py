# clear sqlite db  

import os 

from models.database import SessionLocal, Base, engine 
import crud.model_crud as model_crud

if __name__ == "__main__":
    # select  sqlite db 
    db = SessionLocal()
    for model in model_crud.get_all_model_states(db) : 
        print(model.__dict__)
    
    # Base.metadata.drop_all(bind=engine)
    # Base.metadata.create_all(bind=engine)