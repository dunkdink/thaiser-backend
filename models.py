from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

    
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password = Column(String(128), nullable=False)
    name = Column(String)
    age = Column(Integer)
    gender = Column(String)
    
    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
    
class Record(Base):
    __tablename__ = 'records'
    index = Column(Integer,primary_key=True)
    relative_path = Column(String)
    output = Column(Integer)
    emotion = Column(String)
    user_id = Column(Integer)
    
    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}
    

