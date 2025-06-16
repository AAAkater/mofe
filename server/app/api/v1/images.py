import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.db.minio import minio_client
from app.models.response import FileItem, ResponseBase
from app.utils.logger import logger

router = APIRouter(tags=["images"])


@router.post(
    path="/upload/image",
    summary="上传图片文件",
)
async def upload_image(file: UploadFile = File(...)) -> ResponseBase[FileItem]:
    if not file.content_type or not file.filename:
        logger.error(f"无效文件:{file.content_type=},{file.filename}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="无效文件"
        )

    if not file.content_type.startswith("image/"):
        logger.error(f"非图片格式:{file.filename}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="请上传图片文件"
        )
    # 存入minio
    try:
        file_extension = file.filename.split(".")[-1]
        file_id = uuid.uuid4()
        unique_filename = f"{file_id}.{file_extension}"

        content = await file.read()

        minio_client.upload_file(object_name=unique_filename, file_data=content)

        return ResponseBase[FileItem](
            data=FileItem(
                id=file_id,
                filename=unique_filename,
                bucket=settings.MINIO_BUCKET_NAME,
            )
        )
    except Exception as e:
        logger.error(f"存储文件失败:\n{e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="存储文件失败",
        )
    finally:
        await file.close()
