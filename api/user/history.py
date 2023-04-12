from fastapi import APIRouter, HTTPException
from fastapi_sqlalchemy import db
from models import User
from models import Record

router = APIRouter()


@router.get("/items")
def get_all_items(username: str):
    user = db.session.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    records = db.session.query(Record).filter_by(user_id=user.id).all()
    return records
