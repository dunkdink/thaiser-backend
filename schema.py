from pydantic import BaseModel


class User(BaseModel):
    username: str
    email: str
    password: str
    name: str
    age: int
    gender: int

    class Config:
        orm_mode = True


class Record(BaseModel):
    relative_path: str
    output: str
    emotion: str
    user_id: str

    class Config:
        orm_mode = True
