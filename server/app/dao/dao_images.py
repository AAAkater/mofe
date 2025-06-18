import uuid
from datetime import datetime
from uuid import UUID

from sqlmodel import Session, desc, select

from app.models.db_models import Images


def get_image_by_id(
    *,
    session: Session,
    id: UUID,
) -> Images:
    stmt = select(Images).where(Images.id == id)
    result = session.exec(stmt).one()
    return result


def create_new_images(
    *,
    session: Session,
    filename: str,
    new_id: UUID | None = None,
    image_type: str = "original",
    original_image_id: UUID | None = None,
    restored_at: datetime | None = None,
) -> Images:
    """
    创建新的图片记录

    Args:
        session: 数据库会话
        filename: 文件名
        width: 图片宽度
        height: 图片高度
        image_type: 图片类型 (original/restored)
        original_image_id: 原图ID (仅修复图需要)
        restored_at: 修复完成时间 (仅修复图需要)

    Returns:
        Images: 创建的图片记录
    """
    if new_id is None:
        new_id = uuid.uuid4()
    image = Images(
        id=new_id,
        filename=filename,
        image_type=image_type,
        original_image_id=original_image_id,
        restored_at=restored_at,
    )
    session.add(image)
    session.commit()
    session.refresh(image)
    return image


def get_images_history(
    *,
    session: Session,
    original_image_id: UUID | None = None,
    skip: int = 0,
    limit: int = 100,
):
    """
    获取图像修复历史

    Args:
        session: 数据库会话
        original_image_id: 原图ID，如果提供则只返回该原图的修复历史
        skip: 分页跳过条数
        limit: 分页限制条数

    Returns:
        list[Images]: 修复历史记录列表
    """
    statement = select(Images).where(
        not Images.is_deleted, Images.image_type == "restored"
    )

    if original_image_id:
        statement = statement.where(
            Images.original_image_id == original_image_id
        )

    statement = (
        statement.order_by(desc(Images.restored_at)).offset(skip).limit(limit)
    )

    result = session.exec(statement).all()
    return result
