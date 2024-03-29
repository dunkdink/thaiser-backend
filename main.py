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
from api.user import history

load_dotenv(".env")

app = FastAPI()

app.add_middleware(DBSessionMiddleware, db_url=os.environ["DATABASE_URL"])
app.include_router(signInsignUp.router)
app.include_router(editProfile.router)
app.include_router(upload.router)
app.include_router(history.router)

def custom_openapi():
    """Generate a custom OpenAPI schema"""
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="My API",
        version="1.0.0",
        description="This is a custom OpenAPI schema",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/swagger.json")
async def get_swagger_json():
    return custom_openapi()

# Set up allowed origins
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "https://example.com",
    "https://ce32.ce.kmitl.cloud",
    "https://api.ce32.ce.kmitl.cloud"
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