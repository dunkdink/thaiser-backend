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
    recorder_id: str
    record_file: str
    emotion_label_by_human: str
    emotion_label_by_machine: str
    
    class Config:
        orm_mode = True