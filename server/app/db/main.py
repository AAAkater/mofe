from contextlib import asynccontextmanager
from typing import Annotated, AsyncGenerator

from fastapi import Depends, FastAPI
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Session, SQLModel

from app.db.minio import MinioClient
from app.db.postgres import get_db_session, pg_engine
from app.utils.logger import logger


async def init_postgres() -> bool:
    """Initialize PostgreSQL database"""
    try:
        SQLModel.metadata.create_all(pg_engine)
        return True
    except SQLAlchemyError as e:
        logger.error(f"PostgreSQL initialization failed: {e}")
        return False


async def init_minio() -> bool:
    """Initialize MinIO storage"""
    try:
        test_bucket = "system-test"
        minio_client = MinioClient(test_bucket)
        # Test connection by creating a test bucket
        minio_client.create_bucket()
        return True
    except Exception as e:
        logger.error(f"MinIO initialization failed: {e}")
        return False


@asynccontextmanager
async def init_db(_: FastAPI) -> AsyncGenerator:
    """Initialize all database connections"""
    try:
        # Initialize PostgreSQL
        if not await init_postgres():
            raise Exception("PostgreSQL initialization failed")

        # Initialize MinIO
        if not await init_minio():
            raise Exception("MinIO initialization failed")

        logger.success("All database services initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        exit(1)

    yield

    # Cleanup
    try:
        pg_engine.dispose()
        logger.success("PostgreSQL connection closed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")


SessionDep = Annotated[Session, Depends(get_db_session)]
