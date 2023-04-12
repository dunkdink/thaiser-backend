from fastapi import APIRouter, HTTPException
from fastapi_sqlalchemy import db
from sqlalchemy.exc import SQLAlchemyError
from models import User, Record

router = APIRouter()

@router.get("/history")
def get_History_User(username: str = None):
    if not username:
        raise HTTPException(status_code=400, detail="Missing username parameter")
    
    try:
        user = db.session.query(User).filter(User.username == username).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        records = db.session.query(Record).filter(Record.user_id == user.id).all()
        return records
        
    except SQLAlchemyError as e:
        db.session.rollback()
        raise HTTPException(status_code=500, detail=str(e))
        
    except HTTPException:
        raise
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
