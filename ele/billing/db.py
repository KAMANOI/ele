"""SQLAlchemy engine and session factory for ele billing."""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from ele.billing.config import get_config


class Base(DeclarativeBase):
    pass


def _make_engine():
    cfg = get_config()
    url = cfg.database_url
    kwargs: dict = {}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


engine = _make_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Create all tables (idempotent).  Call on application startup."""
    from ele.billing import models  # noqa: F401 — registers ORM models with Base
    Base.metadata.create_all(engine)


def get_db():
    """FastAPI dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
