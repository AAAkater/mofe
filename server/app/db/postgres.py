from sqlalchemy import Engine
from sqlmodel import Session, create_engine

from app.core.config import settings

pg_engine: Engine = create_engine(url=str(settings.POSTGRESQL_URI))


def get_db_session():
    """
    Generator function to yield a database session.

    This function creates a context-managed database session using the provided PostgreSQL engine.
    The session is yielded for use in database operations, ensuring proper cleanup after use.

    Yields:
        sqlalchemy.orm.Session: A SQLAlchemy database session object.

    Example:
        >>> for session in get_db_session():
        ...     # perform database operations
        ...     pass
    """
    with Session(pg_engine) as session:
        yield session
