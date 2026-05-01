"""
crypto.py — AES-256-GCM encryption for message content
========================================================
Each message gets its own random 96-bit IV (nonce).
The same MESSAGE_ENCRYPTION_KEY is used for all messages, but the unique
IV per message ensures ciphertexts are never reused.

Key derivation
──────────────
Set MESSAGE_ENCRYPTION_KEY to a 64-char hex string (32 raw bytes = 256-bit key).
Generate one with:  python -c "import secrets; print(secrets.token_hex(32))"

If the env var is missing, a deterministic fallback is used and a warning is logged.
DO NOT use the fallback in production.
"""

import logging
import os
import secrets

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

log = logging.getLogger("message.crypto")

_HEX_KEY = os.getenv("MESSAGE_ENCRYPTION_KEY", "")

if not _HEX_KEY:
    log.warning(
        "MESSAGE_ENCRYPTION_KEY not set — using insecure dev key. "
        "Set a 64-char hex string in production."
    )
    _HEX_KEY = "0" * 64   # all-zero key — dev only

try:
    _KEY_BYTES = bytes.fromhex(_HEX_KEY)
    if len(_KEY_BYTES) != 32:
        raise ValueError(f"Key must be 32 bytes (64 hex chars), got {len(_KEY_BYTES)}")
except ValueError as exc:
    raise RuntimeError(f"Invalid MESSAGE_ENCRYPTION_KEY: {exc}") from exc

_aesgcm = AESGCM(_KEY_BYTES)


def encrypt(plaintext: str) -> tuple[bytes, bytes]:
    """
    Encrypt a UTF-8 string with AES-256-GCM.

    Returns:
        (ciphertext, iv)  — both raw bytes.
        iv is 12 bytes (96-bit nonce); ciphertext includes the 16-byte GCM auth tag.
    """
    iv         = secrets.token_bytes(12)   # 96-bit nonce — unique per message
    ciphertext = _aesgcm.encrypt(iv, plaintext.encode("utf-8"), None)
    return ciphertext, iv


def decrypt(ciphertext: bytes, iv: bytes) -> str:
    """
    Decrypt AES-256-GCM ciphertext.

    Returns:
        Decrypted plaintext string.

    Raises:
        cryptography.exceptions.InvalidTag if the ciphertext or IV is tampered.
    """
    return _aesgcm.decrypt(iv, ciphertext, None).decode("utf-8")
