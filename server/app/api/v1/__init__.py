from fastapi import APIRouter

from app.api.v1 import images

v1_router = APIRouter(prefix="/v1")

v1_router.include_router(images.router)
