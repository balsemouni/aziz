"""
main.py — Auth Service  v1.0
=============================
Port 8006

Endpoints
─────────
  POST /auth/register      Create account
  POST /auth/login         Authenticate → access + refresh tokens
  POST /auth/logout        Invalidate refresh token
  POST /auth/refresh       Exchange refresh token → new access token
  GET  /auth/me            Return current user profile (requires access token)

Auth header format (protected endpoints):
  Authorization: Bearer <access_token>

Environment variables
─────────────────────
  AUTH_DATABASE_URL       postgresql+asyncpg://user:pass@host:5432/db
  JWT_SECRET              HS256 signing secret (min 32 chars)
  ACCESS_TOKEN_TTL_MIN    Default 15 (minutes)
  REFRESH_TOKEN_TTL_DAYS  Default 7  (days)
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth_utils import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)
from database import Base, engine, get_db
from models import User

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("auth.service")

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    log.info("Auth DB tables ensured.")
    yield
    await engine.dispose()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Auth Service", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer_scheme = HTTPBearer(auto_error=False)

# ── Schemas ───────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email:    EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email:    EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"


class AccessTokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    id:         str
    email:      str
    is_active:  bool
    created_at: datetime


class MessageResponse(BaseModel):
    message: str

# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Dependency: decode Bearer token and return the User row."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = decode_token(credentials.credentials)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if payload.get("type") != "access":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Wrong token type")

    result = await db.execute(select(User).where(User.id == uuid.UUID(payload["sub"])))
    user   = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "auth", "version": "1.0.0"}


@app.post(
    "/auth/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["auth"],
)
async def register(req: RegisterRequest, db: AsyncSession = Depends(get_db)):
    # Reject duplicate email
    existing = await db.execute(select(User).where(User.email == req.email))
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    user = User(
        email         = req.email,
        password_hash = hash_password(req.password),
    )
    db.add(user)
    await db.flush()   # get user.id before commit

    access  = create_access_token(str(user.id), user.email)
    refresh = create_refresh_token(str(user.id))
    user.refresh_token_hash = hash_refresh_token(refresh)

    await db.commit()
    log.info(f"Registered user {user.email} id={user.id}")
    return TokenResponse(access_token=access, refresh_token=refresh)


@app.post("/auth/login", response_model=TokenResponse, tags=["auth"])
async def login(req: LoginRequest, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == req.email))
    user   = result.scalar_one_or_none()

    # Constant-time comparison — always verify even if user is None to prevent timing attacks
    dummy_hash = "$2b$12$" + "x" * 53
    stored     = user.password_hash if user else dummy_hash
    valid      = verify_password(req.password, stored)

    if user is None or not valid or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    access  = create_access_token(str(user.id), user.email)
    refresh = create_refresh_token(str(user.id))
    user.refresh_token_hash = hash_refresh_token(refresh)
    await db.commit()

    log.info(f"Login: {user.email}")
    return TokenResponse(access_token=access, refresh_token=refresh)


@app.post("/auth/logout", response_model=MessageResponse, tags=["auth"])
async def logout(
    req: RefreshRequest,
    db:  AsyncSession = Depends(get_db),
):
    """Invalidate the supplied refresh token (stored hash is cleared)."""
    try:
        payload = decode_token(req.refresh_token)
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Not a refresh token")

    result = await db.execute(select(User).where(User.id == uuid.UUID(payload["sub"])))
    user   = result.scalar_one_or_none()
    if user:
        expected = hash_refresh_token(req.refresh_token)
        if user.refresh_token_hash != expected:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token already invalidated")
        user.refresh_token_hash = None
        await db.commit()

    log.info(f"Logout: user {payload.get('sub')}")
    return MessageResponse(message="Logged out")


@app.post("/auth/refresh", response_model=AccessTokenResponse, tags=["auth"])
async def refresh_access(req: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """Exchange a valid refresh token for a new access token."""
    try:
        payload = decode_token(req.refresh_token)
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Not a refresh token")

    result = await db.execute(select(User).where(User.id == uuid.UUID(payload["sub"])))
    user   = result.scalar_one_or_none()
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    expected = hash_refresh_token(req.refresh_token)
    if user.refresh_token_hash != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token revoked")

    access = create_access_token(str(user.id), user.email)
    return AccessTokenResponse(access_token=access)


@app.get("/auth/me", response_model=UserResponse, tags=["auth"])
async def get_me(current_user: User = Depends(_get_current_user)):
    return UserResponse(
        id         = str(current_user.id),
        email      = current_user.email,
        is_active  = current_user.is_active,
        created_at = current_user.created_at,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8006, reload=False)
