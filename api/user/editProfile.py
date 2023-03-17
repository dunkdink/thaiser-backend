
from fastapi import APIRouter
from fastapi_sqlalchemy import  db
from models import User
from models import Record

router = APIRouter()

@router.put("/user")
def updateProfile(username: str,name:str,age:int,gender:str,password:str):
    user = db.session.query(User).filter(User.username == username).first()

    user.name = name
    user.password = password
    user.age = age
    user.gender = gender
    
    db.session.commit()
    
    return user