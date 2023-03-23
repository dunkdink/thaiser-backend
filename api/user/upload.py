from fastapi import APIRouter,FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import json
import asyncio

router = APIRouter()


@router.post('/upload')
async def upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename}