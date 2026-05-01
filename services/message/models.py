"""
models.py — SQLAlchemy ORM models for Message service
"""
import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, LargeBinary, String, Text
from sqlalchemy.dialects.postgresql import UUID

from database import Base


class Message(Base):
    __tablename__ = "messages"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id  = Column(String(64), nullable=False, index=True)
    user_id     = Column(String(64), nullable=False, index=True)
    role        = Column(String(16), nullable=False)       # "user" | "assistant"
    # AES-GCM encrypted content — stored as raw bytes
    ciphertext  = Column(LargeBinary, nullable=False)
    iv          = Column(LargeBinary, nullable=False)      # 12-byte GCM nonce
    created_at  = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
