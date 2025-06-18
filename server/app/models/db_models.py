import uuid
from datetime import datetime, timezone
from uuid import UUID

from sqlmodel import Field, SQLModel


class Images(SQLModel, table=True):
    id: UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    filename: str
    is_deleted: bool = Field(default=False, nullable=False)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"onupdate": lambda: datetime.now(timezone.utc)},
    )
    status: str = Field(default="active", max_length=50, nullable=False)
    image_type: str = Field(
        default="original", max_length=50, nullable=False
    )  # 取值: original(原图) 或 restored(修复图)
    original_image_id: UUID | None = Field(
        default=None, foreign_key="images.id"
    )  # 如果是修复图,指向原图的ID
    restored_at: datetime | None = Field(default=None)  # 修复完成时间
