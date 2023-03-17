import jwt
from fastapi import APIRouter
from fastapi_sqlalchemy import  db
from passlib.context import CryptContext
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import Depends, HTTPException
from models import User
from models import Record

router = APIRouter()

__pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT config    
__SECRET_KEY = "dunkdink"
__ALGORITHM = "HS256"
    # 1 hour
__ACCESS_TOKEN_EXPIRE_MINUTES = 60 

def __authenticate_user( username: str, password: str) -> User:
    user = db.session.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    user = User(id=user.id,username=user.username, email=user.email, password=user.password,name=user.name, age=user.age, gender=user.gender)

    if not __pwd_context.verify(password, user.password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    return user

def __create_access_token( data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=__ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "sub": data["username"]})
    encoded_jwt = jwt.encode(to_encode, __SECRET_KEY, algorithm=__ALGORITHM)
    return encoded_jwt

@router.post("/signin")
def login( form_data: OAuth2PasswordRequestForm = Depends()):
    auth = __authenticate_user(form_data.username, form_data.password)
    access_token = __create_access_token(auth.to_dict())
    return {"access_token": access_token, "token_type": "bearer","user" : auth}


def __is_valid_password( password: str) -> bool:
    '''
    The password must be at least 8 characters long
    The password must contain at least one uppercase letter
    The password must contain at least one lowercase letter
    The password must contain at least one digit
    '''

    # Check if password is at least 8 characters long
    if len(password) < 8:
        return False

    # Check if password contains at least one uppercase letter
    if not any(char.isupper() for char in password):
        return False

    # Check if password contains at least one lowercase letter
    if not any(char.islower() for char in password):
        return False

    # Check if password contains at least one digit
    if not any(char.isdigit() for char in password):
        return False

    # Password meets all criteria
    return True

@router.post("/validateToken")
def validateToken(token:str):
    payload = jwt.decode(token, "dunkdink", algorithms=["HS256"])
    if not payload['sub']:
            raise HTTPException(status_code=401, detail="Invalid token")
    return payload



@router.post("/signup")
def create_user( username: str, email: str, password: str, name: str,age: int,gender: str):

    # if __authClient.find_one({"username": username}):
    #     raise HTTPException(status_code=400, detail="Username already exists")        

    # if __authClient.find_one({"client_id": client_id}):
    #     raise HTTPException(status_code=400, detail="Client already have a Authorization account")   

    # if not __is_valid_password(password):
    #     raise HTTPException(status_code=400, detail="Password must contain at least one uppercase letter, one lowercase letter, one digit")

    hashed_password = __pwd_context.hash(password)
    user = User(username=username, email=email, password=hashed_password,name=name, age=age, gender=gender)
    db.session.add(user)
    db.session.commit()
    return user
