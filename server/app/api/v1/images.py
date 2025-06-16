import io
from datetime import datetime, timezone
from uuid import UUID

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.dao import dao_images
from app.db.main import SessionDep
from app.db.minio import minio_client
from app.models.response import FileItem, ResponseBase
from app.utils.logger import logger

router = APIRouter(tags=["images"])


@router.post(
    path="/upload/image",
    summary="上传图片文件",
)
async def upload_image(
    session: SessionDep,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> ResponseBase[FileItem]:
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
    try:
        file_extension = file.filename.split(".")[-1]
        content = await file.read()
        # 旧图片信息存入pg
        db_image = dao_images.create_new_images(
            session=session,
            filename=file.filename,
            image_type="original",
        )
        # 图片文件存入minio
        minio_client.upload_file(
            object_name=str(db_image.id), file_data=content
        )
        logger.success(f"图片存储成功:{db_image.id=} {db_image.filename=}")

        # TODO 图片修复逻辑
        # background_tasks.add_task()

        # 将新图片存入pg
        background_tasks.add_task(
            dao_images.create_new_images,
            session=session,
            filename=f"{datetime.now(timezone.utc)}.{file_extension}",
            image_type="restored",
            original_image_id=db_image.id,
            restored_at=datetime.now(timezone.utc),
        )

        return ResponseBase[FileItem](
            data=FileItem(
                id=db_image.id,
                filename=db_image.filename,
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


@router.get("/download/image", summary="下载图片")
async def download_image(
    session: SessionDep,
    file_id: UUID,
):
    # 从数据库获取图片信息
    try:
        db_image = dao_images.get_image_by_id(session=session, id=file_id)
    except Exception as e:
        logger.error(f"图片未找到:{file_id=}\n{e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="图片未找到",
        )

    # 从minio获取图片内容
    try:
        file_data = minio_client.download_file(object_name=str(file_id))
        # 返回文件内容
        # URL编码文件名
        from urllib.parse import quote

        encoded_filename = quote(db_image.filename)

        # 返回文件内容
        return StreamingResponse(
            io.BytesIO(file_data),
            media_type="image/*",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{encoded_filename}"
            },
        )

    except Exception as e:
        logger.error(f"下载文件失败:\n{e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="下载文件失败",
        )


@router.get("/history/images")
async def get_images_history(
    session: SessionDep,
    original_image_id: UUID | None = None,
    skip: int = 0,
    limit: int = 100,
) -> ResponseBase[list[FileItem]]:
    """
    获取图片修复历史
    """
    try:
        db_images = dao_images.get_images_history(
            session=session,
            original_image_id=original_image_id,
            skip=skip,
            limit=limit,
        )

        return ResponseBase[list[FileItem]](
            data=[
                FileItem(
                    id=image.id,
                    filename=image.filename,
                    bucket=settings.MINIO_BUCKET_NAME,
                )
                for image in db_images
            ]
        )
    except Exception as e:
        logger.error(f"获取图片历史失败:\n{e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取图片历史失败",
        )
