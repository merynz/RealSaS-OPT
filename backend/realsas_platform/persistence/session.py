from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def create_platform_engine(database_url: str, *, echo: bool = False) -> Engine:
    if not database_url.startswith(("postgresql+psycopg://", "postgresql://")):
        raise ValueError("RealSaS platform metadata requires PostgreSQL/psycopg")
    return create_engine(
        database_url,
        echo=echo,
        pool_pre_ping=True,
        isolation_level="SERIALIZABLE",
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
