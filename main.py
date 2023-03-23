import os

import uvicorn
import jwt
from dotenv import load_dotenv
from fastapi import FastAPI,Request
from fastapi_sqlalchemy import DBSessionMiddleware
from JWTMiddleware import jwt_middleware
from fastapi.middleware.cors import CORSMiddleware
from api.user import signInsignUp
from api.user import editProfile
from api.user import upload


load_dotenv(".env")

app = FastAPI()

app.add_middleware(DBSessionMiddleware, db_url=os.environ["DATABASE_URL"])
app.include_router(signInsignUp.router)
app.include_router(editProfile.router)
app.include_router(upload.router)

@app.get("/")
async def root():
    return {"message": "Hello World"}



# Set up allowed origins
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "https://example.com"
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# unrestricted_routes = ["/signin", "/signup","/docs","/validateToken", "/openapi.json"]
# @app.middleware("http")
# async def add_jwt_middleware(request: Request, call_next):
#     if request.url.path in unrestricted_routes:
#         return await call_next(request)
#     return await jwt_middleware(request, call_next)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
 
'''
axios.get("/something", {
    headers : {
    "Authorization": "Bearer ${access_token}"
    },
    body: {
        ...
    }
})

'''