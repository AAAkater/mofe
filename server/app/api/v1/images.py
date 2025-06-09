import uuid

from fastapi import APIRouter, File, UploadFile

from app.core.config import settings
from app.db.minio import minio_client
from app.models.response import FileItem, ResponseBase

router = APIRouter(tags=["images"])


@router.post("/upload/images")
async def upload_images(files: list[UploadFile] = File(...)):
    uploaded_file_infos: list[FileItem] = []
    errors = []

    for file in files:
        if not file.content_type or not file.filename:
            continue

        if not file.content_type.startswith("image/"):
            errors.append(f"{file.filename} 不是图片文件")
            continue

        try:
            file_extension = file.filename.split(".")[-1]
            file_id = uuid.uuid4()
            unique_filename = f"{file_id}.{file_extension}"

            content = await file.read()

            minio_client.upload_file(
                object_name=unique_filename, file_data=content
            )

            uploaded_file_infos.append(
                FileItem(
                    id=file_id,
                    filename=unique_filename,
                    bucket=settings.MINIO_BUCKET_NAME,
                )
            )

        except Exception as e:
            errors.append(f"上传 {file.filename} 失败: {str(e)}")
        finally:
            await file.close()

    return ResponseBase[list[FileItem]](data=uploaded_file_infos)
