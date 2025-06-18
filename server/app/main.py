from fastapi import FastAPI

from app.api import api_router
from app.db.main import init_db

app = FastAPI(
    title="MoFe API",
    description="MoFe Backend API Service",
    lifespan=init_db,
)

app.include_router(api_router)
