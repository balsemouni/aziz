"""
models.py — SQLAlchemy ORM models for Auth service
"""
import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, String, Text
from sqlalchemy.dialects.postgresql import UUID

from database import Base


class User(Base):
    __tablename__ = "users"

    id              = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email           = Column(String(320), unique=True, nullable=False, index=True)
    password_hash   = Column(Text, nullable=False)
    is_active       = Column(Boolean, default=True, nullable=False)
    # Store a hash of the current refresh token so old tokens are invalidated on logout
    refresh_token_hash = Column(Text, nullable=True)
    created_at      = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
