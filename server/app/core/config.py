import secrets

from pydantic import PostgresDsn, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file="./.env",
        env_ignore_empty=True,
        extra="ignore",
    )

    PROJECT_NAME: str = "mofe"

    # MINIO
    MINIO_ROOT_USER: str = ""
    MINIO_ROOT_PASSWORD: str = ""
    MINIO_SERVER_URL: str = "localhost:9000"
    MINIO_CONSOLE_URL: str = "localhost:9001"
    MINIO_BUCKET_NAME: str = "mofe"
    MINIO_SECURE: bool = False

    # POSTGRESQL
    POSTGRESQL_USER: str = "postgres"
    POSTGRESQL_PASSWORD: str = ""
    POSTGRESQL_PROT: int = 5432
    POSTGRESQL_SERVER: str = "127.0.0.1"
    POSTGRESQL_DB: str = ""

    @computed_field
    @property
    def POSTGRESQL_URI(self) -> PostgresDsn:
        return PostgresDsn.build(
            scheme="postgresql+psycopg2",
            username=self.POSTGRESQL_USER,
            password=self.POSTGRESQL_PASSWORD,
            host=self.POSTGRESQL_SERVER,
            port=self.POSTGRESQL_PROT,
            path=self.POSTGRESQL_DB,
        )

    # TOKEN
    SECRET_KEY: str = secrets.token_urlsafe(nbytes=32)
    ALGORITHM: str = "HS256"
    # 60 minutes * 24 hours * 8 days = 8 days 八天有效期
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    API_VER_STR: str = "/api/v1"


settings = Settings()
