from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi_sqlalchemy import db
import boto3
from io import BytesIO
from models import User
import dataPipeline as dp

router = APIRouter()


@router.post('/upload')
async def upload_to_s3(file: UploadFile = File(...)):
    print('Uploading to S3...')
    bucket = "thaiser-file-storage"
    region = "ap-southeast-2"
    access_key = "AKIA6OROF7LYVCXDAAEO"
    secret_key = "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"
    client = boto3.client('s3', aws_access_key_id=access_key,
                          aws_secret_access_key=secret_key, region_name=region)

    # Read file content
    file_content = await file.read()

    # Upload file to S3
    upload_file_key = 'buffer_input/' + file.filename
    client.upload_fileobj(BytesIO(file_content), bucket, upload_file_key)

    print('Upload to S3 successfully')
    return {'file_name': upload_file_key}

@router.post("/upload_result_to_postgres")
def upload_to_pq():
    return dp.upload_res()


@router.get("/summary/{id}")
def summary(id: int):
    dp.splitter()
    res = dp.classify(id)
    dp.upload_res()
    return res
