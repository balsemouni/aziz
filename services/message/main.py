"""
main.py — Message Service  v1.0
================================
Port 8003

All message content is encrypted at rest with AES-256-GCM.
Each message has its own random IV; decryption requires the same
MESSAGE_ENCRYPTION_KEY that was active at write time.

Endpoints
─────────
  POST   /messages                    Store a message
  GET    /messages/{session_id}       Retrieve all messages for a session
  DELETE /messages/{session_id}       Delete all messages for a session

Environment variables
─────────────────────
  MESSAGE_DATABASE_URL       postgresql+asyncpg://user:pass@host:5432/db
  MESSAGE_ENCRYPTION_KEY     64-char hex string (32 bytes = AES-256)
                             Generate: python -c "import secrets; print(secrets.token_hex(32))"
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from crypto import decrypt, encrypt
from database import Base, engine, get_db
from models import Message

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("message.service")

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Message DB tables ensured.")
    yield
    await engine.dispose()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Message Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────

class StoreMessageRequest(BaseModel):
    session_id: str = Field(..., max_length=64)
    user_id:    str = Field(..., max_length=64)
    role:       str = Field(..., pattern="^(user|assistant)$")
    content:    str = Field(..., min_length=1, max_length=32_768)


class MessageOut(BaseModel):
    id:         str
    session_id: str
    user_id:    str
    role:       str
    content:    str          # decrypted plaintext
    created_at: datetime


class BulkDeleteResponse(BaseModel):
    deleted: int

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "message", "version": "1.0.0"}


@app.post(
    "/messages",
    response_model=MessageOut,
    status_code=status.HTTP_201_CREATED,
    tags=["messages"],
)
async def store_message(req: StoreMessageRequest, db: AsyncSession = Depends(get_db)):
    ciphertext, iv = encrypt(req.content)

    msg = Message(
        session_id = req.session_id,
        user_id    = req.user_id,
        role       = req.role,
        ciphertext = ciphertext,
        iv         = iv,
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)

    log.info(f"Stored message {msg.id} session={req.session_id} role={req.role}")
    return MessageOut(
        id         = str(msg.id),
        session_id = msg.session_id,
        user_id    = msg.user_id,
        role       = msg.role,
        content    = req.content,   # return plaintext in response
        created_at = msg.created_at,
    )


@app.get(
    "/messages/{session_id}",
    response_model=List[MessageOut],
    tags=["messages"],
)
async def get_messages(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
    )
    rows = result.scalars().all()

    out = []
    for row in rows:
        try:
            content = decrypt(row.ciphertext, row.iv)
        except Exception:
            content = "[decryption error]"
        out.append(
            MessageOut(
                id         = str(row.id),
                session_id = row.session_id,
                user_id    = row.user_id,
                role       = row.role,
                content    = content,
                created_at = row.created_at,
            )
        )
    return out


@app.delete(
    "/messages/{session_id}",
    response_model=BulkDeleteResponse,
    tags=["messages"],
)
async def delete_messages(session_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        delete(Message).where(Message.session_id == session_id)
    )
    await db.commit()
    deleted = result.rowcount
    log.info(f"Deleted {deleted} messages for session {session_id}")
    return BulkDeleteResponse(deleted=deleted)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=False)
