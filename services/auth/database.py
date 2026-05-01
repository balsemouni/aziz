"""
database.py — SQLAlchemy async engine for Auth service
"""
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv(
    "AUTH_DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/micro_auth",
)

engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

Base = declarative_base()


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
