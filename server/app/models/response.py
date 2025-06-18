from typing import Generic, TypeVar
from uuid import UUID

from pydantic import BaseModel, Field

T = TypeVar("T")


class ResponseBase(BaseModel, Generic[T]):
    code: str = Field(default="0", description="Business Code")
    msg: str = Field(default="ok")
    data: T | None = None


class FileItem(BaseModel):
    id: UUID
    filename: str
    bucket: str
