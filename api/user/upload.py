from uuid import uuid4
from loguru import logger
from fastapi import APIRouter, HTTPException, File, UploadFile, status
from fastapi.responses import StreamingResponse
import os
import magic
import boto3
import json
import asyncio


router = APIRouter()

ACCESS_KEY = 'AKIA6OROF7LYVCXDAAEO'
SECRET_KEY = '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'
REGION = 'ap-southeast-2'

s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=REGION)

@router.post('/upload')
async def create_upload_file(file: bytes = File(...)):

    response = s3.put_object(Bucket='thaiser-file-storage', Key='my-object-key', Body=file)
    return {"file_size": len(file)}