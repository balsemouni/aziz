"""
main.py — Session Service  v1.0
================================
Port 8005

Stores session metadata in Redis with a 24-hour TTL.

Endpoints
─────────
  POST   /sessions              Create a new session
  GET    /sessions/{id}         Get session by ID
  PATCH  /sessions/{id}         Update session metadata
  DELETE /sessions/{id}         Delete session
  GET    /sessions?user_id=...  List sessions for a user (scans Redis — dev only)

Session payload stored in Redis
───────────────────────────────
  {
    "session_id":  "<uuid>",
    "user_id":     "<uuid>",
    "created_at":  "<iso8601>",
    "updated_at":  "<iso8601>",
    "metadata":    { ... }       ← arbitrary JSON, e.g. voice, language, tags
  }

Environment variables
─────────────────────
  REDIS_URL           redis://localhost:6379/0
  SESSION_TTL_HOURS   Default 24
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Config ────────────────────────────────────────────────────────────────────

REDIS_URL         = os.getenv("REDIS_URL",          "redis://localhost:6379/0")
SESSION_TTL_S     = int(os.getenv("SESSION_TTL_HOURS", "24")) * 3600
SESSION_KEY_PREFIX = "session:"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("session.service")

# ── Redis client ──────────────────────────────────────────────────────────────

redis_client: Optional[aioredis.Redis] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    redis_client = aioredis.from_url(REDIS_URL, decode_responses=True)
    try:
        await redis_client.ping()
        log.info(f"Redis connected: {REDIS_URL}")
    except Exception as exc:
        log.error(f"Redis connection failed: {exc}")
        raise
    yield
    await redis_client.aclose()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Session Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    user_id:  str = Field(..., description="UUID of the authenticated user")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UpdateSessionRequest(BaseModel):
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SessionResponse(BaseModel):
    session_id: str
    user_id:    str
    created_at: str
    updated_at: str
    metadata:   Dict[str, Any]
    ttl_seconds: Optional[int] = None

# ── Helpers ───────────────────────────────────────────────────────────────────

def _key(session_id: str) -> str:
    return f"{SESSION_KEY_PREFIX}{session_id}"


async def _get_session(session_id: str) -> dict:
    raw = await redis_client.get(_key(session_id))
    if raw is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found or expired",
        )
    return json.loads(raw)


async def _save_session(data: dict, reset_ttl: bool = True) -> None:
    raw = json.dumps(data)
    await redis_client.set(_key(data["session_id"]), raw, ex=SESSION_TTL_S if reset_ttl else None, keepttl=not reset_ttl)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "session", "version": "1.0.0"}


@app.post(
    "/sessions",
    response_model=SessionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["sessions"],
)
async def create_session(req: CreateSessionRequest):
    now = datetime.now(timezone.utc).isoformat()
    session = {
        "session_id": str(uuid.uuid4()),
        "user_id":    req.user_id,
        "created_at": now,
        "updated_at": now,
        "metadata":   req.metadata,
    }
    await _save_session(session)
    log.info(f"Created session {session['session_id']} for user {req.user_id}")
    return SessionResponse(**session, ttl_seconds=SESSION_TTL_S)


@app.get("/sessions/{session_id}", response_model=SessionResponse, tags=["sessions"])
async def get_session(session_id: str):
    session = await _get_session(session_id)
    ttl     = await redis_client.ttl(_key(session_id))
    return SessionResponse(**session, ttl_seconds=ttl)


@app.patch("/sessions/{session_id}", response_model=SessionResponse, tags=["sessions"])
async def update_session(session_id: str, req: UpdateSessionRequest):
    session = await _get_session(session_id)
    session["metadata"].update(req.metadata)
    session["updated_at"] = datetime.now(timezone.utc).isoformat()
    # Preserve remaining TTL — don't reset it on every metadata update
    await _save_session(session, reset_ttl=False)
    ttl = await redis_client.ttl(_key(session_id))
    return SessionResponse(**session, ttl_seconds=ttl)


@app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["sessions"])
async def delete_session(session_id: str):
    deleted = await redis_client.delete(_key(session_id))
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    log.info(f"Deleted session {session_id}")


@app.get("/sessions", response_model=list[SessionResponse], tags=["sessions"])
async def list_sessions(user_id: str = Query(..., description="Filter by user UUID")):
    """
    Scan Redis for all sessions belonging to a user.
    NOTE: SCAN is O(n) over keyspace — suitable for dev/small deployments.
    For production use a secondary index or store session IDs per-user in a Redis Set.
    """
    results = []
    async for key in redis_client.scan_iter(f"{SESSION_KEY_PREFIX}*"):
        raw = await redis_client.get(key)
        if raw:
            data = json.loads(raw)
            if data.get("user_id") == user_id:
                ttl = await redis_client.ttl(key)
                results.append(SessionResponse(**data, ttl_seconds=ttl))
    return results


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8005, reload=False)
