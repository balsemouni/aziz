"""
CAG Microservice — Gateway-Ready Entry Point  v5.0
==================================================

NEW in v5.0
───────────
  WebSocket endpoint  /chat/ws
  ─────────────────────────────
  Adds a persistent WebSocket stream alongside the existing HTTP SSE endpoint.
  The gateway prefers the WS endpoint for minimal framing overhead.

  Protocol:
    → {"type": "query", "turn_id": "...", "message": "...", "reset": bool}
    ← {"type": "turn_id",  "turn_id": "..."}        (first frame — routing confirm)
    ← {"type": "token",    "token": "...", "turn_id": "..."}
    ← {"type": "done",     "turn_id": "..."}
    ← {"type": "error",    "detail": "...", "turn_id": "..."}
    ← {"type": "timeout",  "turn_id": "..."}

  Multiple concurrent sessions are each tracked by their own turn_id.
  The GPU lock serializes inference; if another turn arrives while inference
  is running, it waits in a per-connection queue.
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from statistics import mean, quantiles
from typing import Any, AsyncGenerator, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from cag_config import CAGConfig, get_config_preset
from cag_system import CAGSystemFreshSession

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger("cag.service")

# ── Tunables from env ─────────────────────────────────────────────────────────

DEDUP_WINDOW_S   = float(os.getenv("DEDUP_WINDOW_S",  "2.0"))
TOKEN_TIMEOUT_S  = float(os.getenv("TOKEN_TIMEOUT_S", "30.0"))
DEDUP_CACHE_SIZE = 4

# Max queries to hold per WebSocket connection before dropping
WS_QUERY_QUEUE_MAX = int(os.getenv("WS_QUERY_QUEUE_MAX", "4"))


# ── Metrics ───────────────────────────────────────────────────────────────────

class _Metrics:
    def __init__(self):
        self._total_queries = 0
        self._total_errors  = 0
        self._latencies_ms  : collections.deque = collections.deque(maxlen=200)
        self._lock          = threading.Lock()

    def record(self, latency_ms: float, error: bool = False):
        with self._lock:
            self._total_queries += 1
            if error:
                self._total_errors += 1
            self._latencies_ms.append(latency_ms)

    def snapshot(self) -> dict:
        with self._lock:
            lats = list(self._latencies_ms)
        avg = round(mean(lats), 1) if lats else 0.0
        p95 = round(quantiles(lats, n=20)[18], 1) if len(lats) >= 20 else avg
        return {
            "total_queries":  self._total_queries,
            "total_errors":   self._total_errors,
            "avg_latency_ms": avg,
            "p95_latency_ms": p95,
        }


metrics = _Metrics()


# ── Dedup guard ───────────────────────────────────────────────────────────────

class _DedupGuard:
    def __init__(self, window_s: float = DEDUP_WINDOW_S, maxsize: int = DEDUP_CACHE_SIZE):
        self._window  = window_s
        self._maxsize = maxsize
        self._cache   : collections.OrderedDict[str, float] = collections.OrderedDict()
        self._lock    = threading.Lock()

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def is_duplicate(self, query: str) -> bool:
        key = self._key(query)
        now = time.monotonic()
        with self._lock:
            if key in self._cache:
                if now - self._cache[key] < self._window:
                    return True
            self._cache[key] = now
            self._cache.move_to_end(key)
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)
        return False

    def clear(self):
        with self._lock:
            self._cache.clear()


dedup = _DedupGuard()


# ── Service state ─────────────────────────────────────────────────────────────

class ServiceState:
    def __init__(self):
        self.config    : Optional[CAGConfig]             = None
        self.cag       : Optional[CAGSystemFreshSession] = None
        self.ready     = False
        self.boot_time : Optional[datetime]              = None
        self._gpu_lock = asyncio.Lock()

    async def startup(self):
        log.info("=== CAG SERVICE STARTUP ===")
        preset = os.getenv("CAG_PRESET", "").strip()
        self.config = get_config_preset(preset) if preset else CAGConfig.from_env()
        _validate_config(self.config)
        self.cag       = CAGSystemFreshSession(self.config)
        await asyncio.get_event_loop().run_in_executor(None, self.cag.initialize)
        self.ready     = True
        self.boot_time = datetime.utcnow()
        log.info("=== CAG SERVICE READY ===")

    async def teardown(self):
        log.info("Shutting down…")
        if self.cag:
            self.cag.cleanup()
        torch.cuda.empty_cache()
        log.info("Cleanup done.")

    def reset_session(self):
        if self.cag is None:
            return
        try:
            self.cag._fast_reset()
        except Exception as e:
            log.warning(f"Fast reset failed ({e}), trying full reset")
            try:
                self.cag.reset_conversation()
            except Exception as e2:
                log.error(f"Full reset also failed: {e2}")


def _validate_config(cfg: CAGConfig):
    errors: List[str] = []
    if getattr(cfg, "max_new_tokens", 1) < 1:
        errors.append("max_new_tokens must be >= 1")
    if getattr(cfg, "max_context_tokens", 1) < 64:
        errors.append("max_context_tokens must be >= 64")
    if errors:
        raise RuntimeError("CAGConfig validation failed: " + "; ".join(errors))


def _make_turn_id(hint: Optional[str] = None) -> str:
    return hint or str(uuid.uuid4())


def _assert_ready():
    if not svc.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="CAG not ready",
        )


svc = ServiceState()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    await svc.startup()
    yield
    await svc.teardown()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Ask Novation — CAG Inference Service",
    version="5.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message:       str           = Field(..., min_length=1, max_length=4096)
    reset_session: bool          = Field(default=True)
    turn_id:       Optional[str] = Field(default=None)


class ChatResponse(BaseModel):
    answer       : str
    user_name    : Optional[str] = None
    query_number : int
    turn_id      : str
    success      : bool


class HealthResponse(BaseModel):
    status         : str
    uptime_seconds : Optional[float]
    gpu_free_mb    : Optional[int]
    version        : str


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    uptime = None
    if svc.boot_time:
        uptime = round((datetime.utcnow() - svc.boot_time).total_seconds(), 1)
    gpu_free = None
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        gpu_free = free // (1024 * 1024)
    return HealthResponse(
        status="ok" if svc.ready else "starting",
        uptime_seconds=uptime,
        gpu_free_mb=gpu_free,
        version="5.0.0",
    )


@app.get("/metrics", response_class=PlainTextResponse, tags=["system"])
async def prometheus_metrics():
    snap  = metrics.snapshot()
    lines = [
        "# HELP cag_total_queries Total number of chat queries received",
        "# TYPE cag_total_queries counter",
        f"cag_total_queries {snap['total_queries']}",
        "# HELP cag_total_errors Total number of failed requests",
        "# TYPE cag_total_errors counter",
        f"cag_total_errors {snap['total_errors']}",
        "# HELP cag_avg_latency_ms Average inference latency (ms, last 200 samples)",
        "# TYPE cag_avg_latency_ms gauge",
        f"cag_avg_latency_ms {snap['avg_latency_ms']}",
        "# HELP cag_p95_latency_ms P95 inference latency (ms, last 200 samples)",
        "# TYPE cag_p95_latency_ms gauge",
        f"cag_p95_latency_ms {snap['p95_latency_ms']}",
    ]
    return "\n".join(lines) + "\n"


@app.post("/reset", tags=["chat"])
async def reset_session():
    _assert_ready()
    async with svc._gpu_lock:
        await asyncio.get_event_loop().run_in_executor(None, svc.reset_session)
    dedup.clear()
    log.info("CAG session reset via /reset")
    return {"reset": True}


# ── Chat (batch) ──────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse, tags=["chat"])
async def chat(req: ChatRequest):
    _assert_ready()
    turn_id = _make_turn_id(req.turn_id)

    if dedup.is_duplicate(req.message):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"error": "duplicate_query", "turn_id": turn_id},
        )

    t0         = time.monotonic()
    error_flag = False
    try:
        async with svc._gpu_lock:
            if req.reset_session:
                await asyncio.get_event_loop().run_in_executor(None, svc.reset_session)
            result = await asyncio.get_event_loop().run_in_executor(
                None, svc.cag.query, req.message
            )

        if not result.get("success"):
            error_flag = True
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("error", "Inference error"),
            )

        log.info(
            f"[turn:{turn_id}] chat OK "
            f"q={len(req.message)}ch "
            f"lat={round((time.monotonic()-t0)*1000)}ms"
        )
        return ChatResponse(
            answer       = result["answer"],
            user_name    = result.get("user_name"),
            query_number = result["query_number"],
            turn_id      = turn_id,
            success      = True,
        )
    finally:
        metrics.record((time.monotonic() - t0) * 1000, error=error_flag)


# ── Chat (streaming SSE) ──────────────────────────────────────────────────────

@app.post("/chat/stream", tags=["chat"])
async def chat_stream(req: ChatRequest):
    """
    Single-turn streaming chat via Server-Sent Events.
    Kept for backward compatibility. Prefer the /chat/ws WebSocket endpoint
    for lower latency.

    Token format:    data: <token>\\n\\n
    Turn header:     data: [TURN_ID] <uuid>\\n\\n
    Done signal:     data: [DONE]\\n\\n
    Error signal:    data: [ERROR] <message>\\n\\n
    Timeout signal:  data: [TIMEOUT]\\n\\n
    """
    _assert_ready()
    turn_id = _make_turn_id(req.turn_id)

    if dedup.is_duplicate(req.message):
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"error": "duplicate_query", "turn_id": turn_id},
        )

    async def _event_generator() -> AsyncGenerator[str, None]:
        loop         = asyncio.get_event_loop()
        q            : asyncio.Queue = asyncio.Queue()
        cancel_event = threading.Event()
        t0           = time.monotonic()
        token_count  = [0]
        error_flag   = [False]

        yield f"data: [TURN_ID] {turn_id}\n\n"

        def _producer():
            try:
                if req.reset_session:
                    svc.reset_session()
                for chunk in svc.cag.stream_query(req.message):
                    if cancel_event.is_set():
                        log.info(f"[turn:{turn_id}] stream cancelled by client")
                        break
                    if chunk:
                        token_count[0] += 1
                        loop.call_soon_threadsafe(q.put_nowait, ("token", chunk))
            except Exception as exc:
                error_flag[0] = True
                loop.call_soon_threadsafe(q.put_nowait, ("error", str(exc)))
            finally:
                loop.call_soon_threadsafe(q.put_nowait, ("done", None))

        try:
            async with svc._gpu_lock:
                producer_future = loop.run_in_executor(None, _producer)
                try:
                    while True:
                        try:
                            kind, value = await asyncio.wait_for(q.get(), timeout=TOKEN_TIMEOUT_S)
                        except asyncio.TimeoutError:
                            log.warning(f"[turn:{turn_id}] token timeout after {TOKEN_TIMEOUT_S}s")
                            cancel_event.set()
                            error_flag[0] = True
                            yield f"data: [TIMEOUT]\n\n"
                            break
                        if kind == "token":
                            yield f"data: {value}\n\n"
                        elif kind == "error":
                            log.error(f"[turn:{turn_id}] producer error: {value}")
                            yield f"data: [ERROR] {value}\n\n"
                            break
                        else:
                            break
                finally:
                    cancel_event.set()
                    try:
                        await asyncio.wait_for(asyncio.wrap_future(producer_future), timeout=10.0)
                    except Exception:
                        pass
        except asyncio.CancelledError:
            cancel_event.set()
            log.info(f"[turn:{turn_id}] stream cancelled (client disconnect)")
            return
        finally:
            lat_ms = (time.monotonic() - t0) * 1000
            metrics.record(lat_ms, error=error_flag[0])
            log.info(
                f"[turn:{turn_id}] stream done "
                f"tokens={token_count[0]} "
                f"lat={round(lat_ms)}ms"
            )

        yield f"data: [DONE]\n\n"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Turn-Id":         turn_id,
        },
    )


# ── Chat (WebSocket streaming) ────────────────────────────────────────────────

@app.websocket("/chat/ws")
async def chat_ws(ws: WebSocket):
    """
    Persistent WebSocket endpoint for low-latency streaming inference.

    Each message from the gateway is a JSON query frame:
      {"type": "query", "turn_id": "...", "message": "...", "reset": bool}

    Each response token is a JSON frame:
      {"type": "turn_id", "turn_id": "..."}     ← first frame, routing confirm
      {"type": "token",   "token":  "...", "turn_id": "..."}
      {"type": "done",    "turn_id": "..."}
      {"type": "error",   "detail": "...", "turn_id": "..."}
      {"type": "timeout", "turn_id": "..."}

    Advantages over SSE:
      • Full-duplex: gateway can send barge-in cancel while tokens are flowing
      • Lower per-frame overhead: no HTTP chunked encoding / SSE data prefix
      • Single TCP connection per session vs one per query with SSE
    """
    await ws.accept()
    conn_id = str(uuid.uuid4())[:8]
    log.info(f"[ws:{conn_id}] CAG WebSocket connected")

    # Per-connection query queue — allows pipelining when GPU is busy
    query_q: asyncio.Queue = asyncio.Queue(maxsize=WS_QUERY_QUEUE_MAX)

    async def _receiver():
        """Read query frames from the gateway and enqueue them."""
        try:
            async for raw in ws.iter_text():
                try:
                    frame = json_loads(raw)
                except Exception:
                    continue
                if frame.get("type") != "query":
                    continue
                msg = frame.get("message", "").strip()
                if not msg:
                    continue
                if query_q.full():
                    log.warning(f"[ws:{conn_id}] query queue full — dropping oldest")
                    try:
                        query_q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                await query_q.put(frame)
        except WebSocketDisconnect:
            pass
        finally:
            # Sentinel to stop processor
            await query_q.put(None)

    async def _processor():
        """Process queries one at a time (serialized by GPU lock), stream tokens back."""
        while True:
            frame = await query_q.get()
            if frame is None:
                break

            turn_id  = frame.get("turn_id") or str(uuid.uuid4())
            message  = frame.get("message", "").strip()
            do_reset = frame.get("reset", False)

            if not message:
                continue

            if dedup.is_duplicate(message):
                try:
                    await ws.send_json({"type": "error", "detail": "duplicate_query", "turn_id": turn_id})
                except Exception:
                    pass
                continue

            log.info(f"[ws:{conn_id}] turn:{turn_id} query={message[:60]!r}")

            # Confirm turn routing
            try:
                await ws.send_json({"type": "turn_id", "turn_id": turn_id})
            except Exception:
                break

            loop         = asyncio.get_event_loop()
            q            : asyncio.Queue = asyncio.Queue()
            cancel_event = threading.Event()
            t0           = time.monotonic()
            token_count  = [0]
            error_flag   = [False]

            def _producer():
                try:
                    if do_reset:
                        svc.reset_session()
                    for chunk in svc.cag.stream_query(message):
                        if cancel_event.is_set():
                            break
                        if chunk:
                            token_count[0] += 1
                            loop.call_soon_threadsafe(q.put_nowait, ("token", chunk))
                except Exception as exc:
                    error_flag[0] = True
                    loop.call_soon_threadsafe(q.put_nowait, ("error", str(exc)))
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, ("done", None))

            try:
                async with svc._gpu_lock:
                    producer_future = loop.run_in_executor(None, _producer)
                    try:
                        while True:
                            try:
                                kind, value = await asyncio.wait_for(
                                    q.get(), timeout=TOKEN_TIMEOUT_S
                                )
                            except asyncio.TimeoutError:
                                log.warning(f"[ws:{conn_id}] turn:{turn_id} token timeout")
                                cancel_event.set()
                                error_flag[0] = True
                                await ws.send_json({"type": "timeout", "turn_id": turn_id})
                                break

                            if kind == "token":
                                await ws.send_json({"type": "token", "token": value, "turn_id": turn_id})
                            elif kind == "error":
                                log.error(f"[ws:{conn_id}] turn:{turn_id} producer error: {value}")
                                await ws.send_json({"type": "error", "detail": value, "turn_id": turn_id})
                                error_flag[0] = True
                                break
                            else:  # done
                                break

                    finally:
                        cancel_event.set()
                        try:
                            await asyncio.wait_for(asyncio.wrap_future(producer_future), timeout=10.0)
                        except Exception:
                            pass

            except WebSocketDisconnect:
                cancel_event.set()
                break
            except Exception as e:
                log.error(f"[ws:{conn_id}] turn:{turn_id} error: {e}")
                error_flag[0] = True
                try:
                    await ws.send_json({"type": "error", "detail": str(e), "turn_id": turn_id})
                except Exception:
                    break
            finally:
                lat_ms = (time.monotonic() - t0) * 1000
                metrics.record(lat_ms, error=error_flag[0])
                log.info(
                    f"[ws:{conn_id}] turn:{turn_id} done "
                    f"tokens={token_count[0]} lat={round(lat_ms)}ms"
                )

            # Send done frame
            try:
                await ws.send_json({"type": "done", "turn_id": turn_id})
            except Exception:
                break

    try:
        await asyncio.gather(_receiver(), _processor())
    except Exception as e:
        log.error(f"[ws:{conn_id}] fatal: {e}")
    finally:
        log.info(f"[ws:{conn_id}] disconnected")


# ── JSON helper ───────────────────────────────────────────────────────────────

import json as _json


def json_loads(raw) -> dict:
    if isinstance(raw, (bytes, bytearray)):
        return _json.loads(raw.decode())
    return _json.loads(raw)


# ── Global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception(f"Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Check service logs."},
    )


# ── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info",
        reload=False,
    )