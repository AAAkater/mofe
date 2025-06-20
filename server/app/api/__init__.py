from fastapi import APIRouter

from app.api.v1 import v1_router
from app.api.v2 import v2_router

api_router = APIRouter(prefix="/api")

api_router.include_router(router=v1_router)
api_router.include_router(router=v2_router)
