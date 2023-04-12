from fastapi import APIRouter, HTTPException
from fastapi_sqlalchemy import db
from models import User
from passlib.context import CryptContext

router = APIRouter()

__pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@router.put("/user")
def updateProfile(username: str, name: str, age: int, gender: str):
    user = db.session.query(User).filter(User.username == username).first()

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if age < 0 or age > 120:
        raise HTTPException(status_code=400, detail="Invalid age")

    if not name:
        raise HTTPException(status_code=400, detail="Name is required")

    user.name = name
    user.age = age
    user.gender = gender

    db.session.commit()

    return user


@router.put("/user/password")
def updatePassword(username: str, old_password: str, new_password: str):
    user = db.session.query(User).filter(User.username == username).first()

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    if not old_password:
        raise HTTPException(status_code=400, detail="Old password is required")

    if not new_password:
        raise HTTPException(status_code=400, detail="New password is required")

    if not __pwd_context.verify(old_password, user.password):
        raise HTTPException(status_code=400, detail="Incorrect old password")

    user.password = __pwd_context.hash(new_password)

    db.session.commit()

    return user
