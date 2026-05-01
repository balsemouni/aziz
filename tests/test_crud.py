"""
test_crud.py — CRUD smoke-tests for Auth, Session & Message services
=====================================================================
Ports
  Auth    : 8006
  Session : 8005
  Message : 8003

Run
  python tests/test_crud.py
  python tests/test_crud.py --auth-url http://host:8006 \
                             --session-url http://host:8005 \
                             --message-url http://host:8003

Exit code 0 = all tests passed.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
import datetime

try:
    import requests
except ImportError:
    print("Missing: pip install requests")
    sys.exit(1)

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--auth-url",    default="http://127.0.0.1:8006")
parser.add_argument("--session-url", default="http://127.0.0.1:8005")
parser.add_argument("--message-url", default="http://127.0.0.1:8003")
parser.add_argument("--stop-on-fail", action="store_true",
                    help="Stop on first failure instead of running all tests")
args = parser.parse_args()

AUTH    = args.auth_url.rstrip("/")
SESSION = args.session_url.rstrip("/")
MESSAGE = args.message_url.rstrip("/")

# ── Pretty printer ────────────────────────────────────────────────────────────

W = 70
PASS = "\033[92m PASS \033[0m"
FAIL = "\033[91m FAIL \033[0m"
SKIP = "\033[93m SKIP \033[0m"
BOLD = "\033[1m"
DIM  = "\033[2m"
RST  = "\033[0m"

_results: list[tuple[str, bool, str]] = []


def _hdr(title: str):
    print(f"\n{BOLD}{'─'*W}{RST}")
    print(f"{BOLD}  {title}{RST}")
    print(f"{BOLD}{'─'*W}{RST}")


def ok(name: str, detail: str = ""):
    tag = f"[{PASS}]"
    msg = f"  {tag}  {name}"
    if detail:
        msg += f"  {DIM}{detail}{RST}"
    print(msg)
    _results.append((name, True, detail))


def fail(name: str, detail: str = ""):
    tag = f"[{FAIL}]"
    print(f"  {tag}  {name}  {detail}")
    _results.append((name, False, detail))
    if args.stop_on_fail:
        _summary()
        sys.exit(1)


def check(name: str, cond: bool, detail: str = ""):
    if cond:
        ok(name, detail)
    else:
        fail(name, detail)


def _req(method: str, url: str, **kwargs):
    """Thin wrapper — always returns (response | None, error_str | None)."""
    try:
        r = requests.request(method, url, timeout=10, **kwargs)
        return r, None
    except requests.exceptions.ConnectionError:
        return None, "Connection refused"
    except Exception as e:
        return None, str(e)


def _summary():
    total  = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed
    print(f"\n{'═'*W}")
    print(f"  RESULTS  {passed}/{total} passed", end="")
    if failed:
        print(f"  —  {failed} FAILED \033[91m✗\033[0m")
    else:
        print(f"  —  all green \033[92m✓\033[0m")
    print(f"{'═'*W}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  State shared across test sections
# ─────────────────────────────────────────────────────────────────────────────

state: dict = {}      # access_token, refresh_token, user_id, session_id


# ─────────────────────────────────────────────────────────────────────────────
#  AUTH TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_auth():
    _hdr("AUTH SERVICE  " + AUTH)

    # ── Health ────────────────────────────────────────────────────────────────
    r, err = _req("GET", f"{AUTH}/health")
    if err:
        fail("GET /health", err); return
    check("GET /health → 200", r.status_code == 200, str(r.json()))

    # ── Register ──────────────────────────────────────────────────────────────
    email = f"test_{uuid.uuid4().hex[:8]}@example.com"
    password = "Test1234!"
    r, err = _req("POST", f"{AUTH}/auth/register",
                  json={"email": email, "password": password})
    if err:
        fail("POST /auth/register", err); return
    check("POST /auth/register → 201", r.status_code == 201, f"email={email}")
    if r.status_code != 201:
        fail("POST /auth/register body", str(r.text)); return
    data = r.json()
    check("register: access_token present",  "access_token"  in data)
    check("register: refresh_token present", "refresh_token" in data)
    state["access_token"]  = data["access_token"]
    state["refresh_token"] = data["refresh_token"]
    state["email"]         = email
    state["password"]      = password

    # ── Duplicate register → 409 ──────────────────────────────────────────────
    r2, _ = _req("POST", f"{AUTH}/auth/register",
                 json={"email": email, "password": password})
    check("POST /auth/register duplicate → 409", r2 is not None and r2.status_code == 409)

    # ── Me (protected) ────────────────────────────────────────────────────────
    r, err = _req("GET", f"{AUTH}/auth/me",
                  headers={"Authorization": f"Bearer {state['access_token']}"})
    if err:
        fail("GET /auth/me", err)
    else:
        check("GET /auth/me → 200", r.status_code == 200)
        if r.status_code == 200:
            me = r.json()
            check("GET /auth/me: email matches", me.get("email") == email)
            state["user_id"] = me.get("id", str(uuid.uuid4()))

    # ── Me without token → 401 ───────────────────────────────────────────────
    r, _ = _req("GET", f"{AUTH}/auth/me")
    check("GET /auth/me no token → 401", r is not None and r.status_code == 401)

    # ── Login ─────────────────────────────────────────────────────────────────
    r, err = _req("POST", f"{AUTH}/auth/login",
                  json={"email": email, "password": password})
    if err:
        fail("POST /auth/login", err)
    else:
        check("POST /auth/login → 200", r.status_code == 200)
        if r.status_code == 200:
            data = r.json()
            check("login: access_token present",  "access_token"  in data)
            check("login: refresh_token present", "refresh_token" in data)
            # Use freshly-issued tokens for the rest of the tests
            state["access_token"]  = data["access_token"]
            state["refresh_token"] = data["refresh_token"]

    # ── Login wrong password → 401 ────────────────────────────────────────────
    r, _ = _req("POST", f"{AUTH}/auth/login",
                json={"email": email, "password": "WrongPass999!"})
    check("POST /auth/login bad password → 401", r is not None and r.status_code == 401)

    # ── Refresh token ─────────────────────────────────────────────────────────
    r, err = _req("POST", f"{AUTH}/auth/refresh",
                  json={"refresh_token": state.get("refresh_token", "")})
    if err:
        fail("POST /auth/refresh", err)
    else:
        check("POST /auth/refresh → 200", r.status_code == 200)
        if r.status_code == 200:
            data = r.json()
            check("refresh: new access_token present", "access_token" in data)
            state["access_token"] = data["access_token"]   # keep updated

    # ── Logout ────────────────────────────────────────────────────────────────
    # (Logout but re-login so later tests can still use a valid token)
    old_refresh = state.get("refresh_token", "")
    r, err = _req("POST", f"{AUTH}/auth/logout",
                  json={"refresh_token": old_refresh})
    if err:
        fail("POST /auth/logout", err)
    else:
        check("POST /auth/logout → 200", r.status_code == 200)

    # Re-login to refresh tokens for downstream tests
    r2, _ = _req("POST", f"{AUTH}/auth/login",
                 json={"email": email, "password": password})
    if r2 and r2.status_code == 200:
        d = r2.json()
        state["access_token"]  = d["access_token"]
        state["refresh_token"] = d["refresh_token"]


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_session():
    _hdr("SESSION SERVICE  " + SESSION)

    user_id = state.get("user_id", str(uuid.uuid4()))

    # ── Health ────────────────────────────────────────────────────────────────
    r, err = _req("GET", f"{SESSION}/health")
    if err:
        fail("GET /health", err); return
    check("GET /health → 200", r.status_code == 200, str(r.json()))

    # ── Create session ────────────────────────────────────────────────────────
    r, err = _req("POST", f"{SESSION}/sessions",
                  json={"user_id": user_id,
                        "metadata": {"voice": "tara", "language": "en"}})
    if err:
        fail("POST /sessions", err); return
    check("POST /sessions → 201", r.status_code == 201)
    if r.status_code != 201:
        fail("POST /sessions body", str(r.text)); return
    sess = r.json()
    sid  = sess["session_id"]
    check("create: session_id present",    bool(sid))
    check("create: user_id matches",       sess.get("user_id") == user_id)
    check("create: metadata voice=tara",   sess.get("metadata", {}).get("voice") == "tara")
    check("create: ttl_seconds > 0",       (sess.get("ttl_seconds") or 0) > 0)
    state["session_id"] = sid

    # ── Get session ───────────────────────────────────────────────────────────
    r, err = _req("GET", f"{SESSION}/sessions/{sid}")
    if err:
        fail("GET /sessions/{id}", err)
    else:
        check("GET /sessions/{id} → 200",         r.status_code == 200)
        check("GET /sessions/{id}: id matches",   r.json().get("session_id") == sid)

    # ── Get non-existent → 404 ────────────────────────────────────────────────
    r, _ = _req("GET", f"{SESSION}/sessions/{uuid.uuid4()}")
    check("GET /sessions/missing → 404", r is not None and r.status_code == 404)

    # ── Update session ────────────────────────────────────────────────────────
    r, err = _req("PATCH", f"{SESSION}/sessions/{sid}",
                  json={"metadata": {"language": "fr", "tag": "updated"}})
    if err:
        fail("PATCH /sessions/{id}", err)
    else:
        check("PATCH /sessions/{id} → 200", r.status_code == 200)
        if r.status_code == 200:
            updated = r.json()
            check("PATCH: language updated to fr",
                  updated.get("metadata", {}).get("language") == "fr")
            check("PATCH: original voice key preserved",
                  updated.get("metadata", {}).get("voice") == "tara")
            check("PATCH: updated_at changed",
                  updated.get("updated_at") != sess.get("created_at"))

    # ── List sessions by user ─────────────────────────────────────────────────
    r, err = _req("GET", f"{SESSION}/sessions", params={"user_id": user_id})
    if err:
        fail("GET /sessions?user_id=...", err)
    else:
        check("GET /sessions?user_id=... → 200", r.status_code == 200)
        if r.status_code == 200:
            items = r.json()
            check("list: at least 1 session returned", len(items) >= 1)
            found = any(s["session_id"] == sid for s in items)
            check("list: created session is in results", found)

    # ── Delete session ────────────────────────────────────────────────────────
    r, err = _req("DELETE", f"{SESSION}/sessions/{sid}")
    if err:
        fail("DELETE /sessions/{id}", err)
    else:
        check("DELETE /sessions/{id} → 204", r.status_code == 204)

    # ── Verify gone → 404 ────────────────────────────────────────────────────
    r, _ = _req("GET", f"{SESSION}/sessions/{sid}")
    check("GET after DELETE → 404", r is not None and r.status_code == 404)

    # ── Delete non-existent → 404 ────────────────────────────────────────────
    r, _ = _req("DELETE", f"{SESSION}/sessions/{uuid.uuid4()}")
    check("DELETE /sessions/missing → 404", r is not None and r.status_code == 404)


# ─────────────────────────────────────────────────────────────────────────────
#  MESSAGE TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_message():
    _hdr("MESSAGE SERVICE  " + MESSAGE)

    user_id    = state.get("user_id",    str(uuid.uuid4()))
    session_id = state.get("session_id", f"sess_{uuid.uuid4().hex}")

    # ── Health ────────────────────────────────────────────────────────────────
    r, err = _req("GET", f"{MESSAGE}/health")
    if err:
        fail("GET /health", err); return
    check("GET /health → 200", r.status_code == 200, str(r.json()))

    # ── Store user message ────────────────────────────────────────────────────
    r, err = _req("POST", f"{MESSAGE}/messages",
                  json={"session_id": session_id,
                        "user_id":    user_id,
                        "role":       "user",
                        "content":    "Hello, this is a test message."})
    if err:
        fail("POST /messages (user)", err); return
    check("POST /messages user → 201", r.status_code == 201)
    if r.status_code != 201:
        fail("POST /messages body", str(r.text)); return
    msg1 = r.json()
    check("store user: id present",           bool(msg1.get("id")))
    check("store user: role=user",            msg1.get("role") == "user")
    check("store user: content decrypted",    msg1.get("content") == "Hello, this is a test message.")
    check("store user: session_id matches",   msg1.get("session_id") == session_id)

    # ── Store assistant message ───────────────────────────────────────────────
    r, err = _req("POST", f"{MESSAGE}/messages",
                  json={"session_id": session_id,
                        "user_id":    user_id,
                        "role":       "assistant",
                        "content":    "Hi! I am the AI assistant."})
    if err:
        fail("POST /messages (assistant)", err)
    else:
        check("POST /messages assistant → 201", r.status_code == 201)
        if r.status_code == 201:
            msg2 = r.json()
            check("store assistant: role=assistant", msg2.get("role") == "assistant")

    # ── Invalid role → 422 ───────────────────────────────────────────────────
    r, _ = _req("POST", f"{MESSAGE}/messages",
                json={"session_id": session_id,
                      "user_id":    user_id,
                      "role":       "admin",
                      "content":    "should be rejected"})
    check("POST /messages invalid role → 422", r is not None and r.status_code == 422)

    # ── Empty content → 422 ──────────────────────────────────────────────────
    r, _ = _req("POST", f"{MESSAGE}/messages",
                json={"session_id": session_id,
                      "user_id":    user_id,
                      "role":       "user",
                      "content":    ""})
    check("POST /messages empty content → 422", r is not None and r.status_code == 422)

    # ── Get messages for session ──────────────────────────────────────────────
    r, err = _req("GET", f"{MESSAGE}/messages/{session_id}")
    if err:
        fail("GET /messages/{session_id}", err)
    else:
        check("GET /messages/{session_id} → 200", r.status_code == 200)
        if r.status_code == 200:
            msgs = r.json()
            check("get: at least 2 messages returned", len(msgs) >= 2)
            roles = [m["role"] for m in msgs]
            check("get: user role present",      "user"      in roles)
            check("get: assistant role present", "assistant" in roles)
            # Verify AES-256-GCM decryption worked (content plaintext is back)
            contents = [m["content"] for m in msgs]
            check("get: plaintext decrypted (user msg)",
                  "Hello, this is a test message." in contents)

    # ── Get messages for non-existent session → empty list ───────────────────
    r, _ = _req("GET", f"{MESSAGE}/messages/nonexistent_{uuid.uuid4().hex}")
    if r is not None and r.status_code == 200:
        check("GET /messages/missing → empty list", r.json() == [])
    else:
        check("GET /messages/missing → 200 or 404",
              r is not None and r.status_code in (200, 404))

    # ── Delete messages for session ───────────────────────────────────────────
    r, err = _req("DELETE", f"{MESSAGE}/messages/{session_id}")
    if err:
        fail("DELETE /messages/{session_id}", err)
    else:
        check("DELETE /messages/{session_id} → 200", r.status_code == 200)
        if r.status_code == 200:
            body = r.json()
            check("delete: deleted count >= 2", (body.get("deleted") or 0) >= 2)

    # ── Verify deleted ────────────────────────────────────────────────────────
    r, _ = _req("GET", f"{MESSAGE}/messages/{session_id}")
    if r and r.status_code == 200:
        check("GET after DELETE → empty list", r.json() == [])


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n{'═'*W}")
    print(f"{BOLD}  CRUD TEST SUITE{RST}  —  {datetime.datetime.now().strftime('%H:%M:%S')}")
    print(f"  Auth    {AUTH}")
    print(f"  Session {SESSION}")
    print(f"  Message {MESSAGE}")
    print(f"{'═'*W}")

    test_auth()
    test_session()
    test_message()

    _summary()

    failed = sum(1 for _, ok, _ in _results if not ok)
    sys.exit(0 if failed == 0 else 1)
