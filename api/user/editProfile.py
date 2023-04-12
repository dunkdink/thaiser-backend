from fastapi import APIRouter, HTTPException
from fastapi_sqlalchemy import db
from models import User
from models import Record
from passlib.context import CryptContext

router = APIRouter()

__pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@router.put("/user")
def updateProfile(username: str, name: str, age: int, gender: str):
    user = db.session.query(User).filter(User.username == username).first()

    user.name = name
    user.age = age
    user.gender = gender

    db.session.commit()

    return user


@router.put("/user/password")
def updatePassword(username: str, old_password: str, new_password: str):
    user = db.session.query(User).filter(User.username == username).first()

    if not __pwd_context.verify(old_password, user.password):
        raise HTTPException(status_code=400, detail="Incorrect old password")

    user.password = __pwd_context.hash(new_password)

    db.session.commit()

    return user
