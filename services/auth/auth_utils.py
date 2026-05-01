"""
auth_utils.py — JWT creation/verification + bcrypt password helpers
"""
import hashlib
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import JWTError, jwt

# ── Config from environment ────────────────────────────────────────────────────
JWT_SECRET          = os.getenv("JWT_SECRET",          "CHANGE_ME_IN_PRODUCTION_32_CHARS_MIN")
JWT_ALGORITHM       = "HS256"
ACCESS_TOKEN_TTL    = int(os.getenv("ACCESS_TOKEN_TTL_MIN",  "15"))    # minutes
REFRESH_TOKEN_TTL   = int(os.getenv("REFRESH_TOKEN_TTL_DAYS", "7"))    # days


# ── Password helpers ───────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    """Hash a plain-text password with bcrypt (cost factor 12)."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches the bcrypt hash."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ── Token helpers ──────────────────────────────────────────────────────────────

def create_access_token(user_id: str, email: str) -> str:
    """Return a signed JWT access token valid for ACCESS_TOKEN_TTL minutes."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub":   user_id,
        "email": email,
        "type":  "access",
        "iat":   now,
        "exp":   now + timedelta(minutes=ACCESS_TOKEN_TTL),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_refresh_token(user_id: str) -> str:
    """Return a signed JWT refresh token valid for REFRESH_TOKEN_TTL days."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub":  user_id,
        "type": "refresh",
        "iat":  now,
        "exp":  now + timedelta(days=REFRESH_TOKEN_TTL),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """
    Decode and verify a JWT.  Raises JWTError on invalid/expired token.
    Returns the full payload dict.
    """
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


def hash_refresh_token(token: str) -> str:
    """SHA-256 digest of the raw refresh token — stored in DB for invalidation."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
