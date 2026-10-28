"""
backend/app/main.py — FastAPI app (WF4).

Run from backend/:  uvicorn app.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
import warnings
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.deps import APIError, load_flags, settings, state
from app.events import bus, current_request_id
from app.routers import analyze, events, metrics, mitigation, network, rag, scenarios, simulate, stream
from app.schemas import FeaturesResponse, HealthResponse

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
log = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from ml.infer import load_models
    from app.services.simulation_service import build_network
    t0 = time.perf_counter()
    try:
        state.bundle = load_models(settings.model_dir, settings.baseline_path)
    except Exception as e:                    # API still starts; ML routes answer 503
        log.warning(json.dumps({"event": "model_load_failed", "error": str(e)}))
        state.bundle = None
    state.network = build_network()
    bus.bind_loop(asyncio.get_running_loop())
    ms = round((time.perf_counter() - t0) * 1000)
    log.info(json.dumps({"event": "startup", "ms": ms, "models": _loaded()}))
    loaded = [k for k, v in _loaded().items() if v]
    bus.publish("system", "info" if state.bundle else "error",
                f"API started in {ms} ms — models loaded: {', '.join(loaded) or 'none'}",
                {"models": _loaded(), "version": getattr(state.bundle, "version", None)})
    yield
    bus.publish("system", "warn", "API shutting down")


app = FastAPI(
    title="Sharavathi Corridor — Suspected Extraction Detection API",
    version="1.0.0",
    description="Simulation-only research API. Outputs indicate *suspected unauthorized "
                "extraction / unexplained anomalies* and require field verification.",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                   allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["X-Request-ID", "X-Latency-ms"])


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    token = current_request_id.set(rid)
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    finally:
        current_request_id.reset(token)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Latency-ms"] = str(ms)
    path = request.url.path
    log.info(json.dumps({"request_id": rid, "method": request.method, "path": path,
                         "status": response.status_code, "latency_ms": ms}))
    if path.startswith("/api/"):
        st = response.status_code
        noisy = path in ("/api/health", "/api/events", "/api/events/meta") or request.method == "OPTIONS"
        level = "error" if st >= 500 else "warn" if st >= 400 else "debug" if noisy else "info"
        bus.publish("api", level, f"{request.method} {path} → {st} ({ms:.0f} ms)",
                    {"method": request.method, "path": path, "status": st, "latency_ms": ms}, request_id=rid)
    return response


def _error(status: int, code: str, message: str, **extra) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, **extra}})


@app.exception_handler(APIError)
async def api_error(_: Request, e: APIError):
    bus.publish("system", "error" if e.status >= 500 else "warn", f"{e.code}: {e.message}",
                {"status": e.status, "code": e.code})
    return _error(e.status, e.code, e.message)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, e: RequestValidationError):
    details = [{"loc": list(err["loc"]), "msg": err["msg"]} for err in e.errors()]
    return _error(422, "validation_error", "Request body failed validation", details=details)


@app.exception_handler(StarletteHTTPException)
async def http_error(_: Request, e: StarletteHTTPException):
    return _error(e.status_code, "http_error", str(e.detail))


@app.exception_handler(Exception)
async def unhandled(_: Request, e: Exception):
    log.exception("unhandled error")
    bus.publish("system", "error", f"Unhandled {type(e).__name__}: {e}")
    return _error(500, "internal_error", f"{type(e).__name__}: {e}")


def _loaded() -> dict[str, bool]:
    b = state.bundle
    names = ["baseline_detector", "xgb_classifier", "localizer_reg", "localizer_rank", "gnn"]
    return {n: b is not None and getattr(b, n, None) is not None for n in names}


@app.get("/api/health", response_model=HealthResponse, tags=["meta"])
def health():
    b = state.bundle
    ok = _loaded()
    return {"status": "ok" if ok["xgb_classifier"] and ok["localizer_reg"] else "degraded",
            "model_version": getattr(b, "version", None), "dataset_hash": getattr(b, "dataset_hash", None),
            "models_loaded": ok, "selected_localizer": getattr(b, "selected_localizer", None)}


@app.get("/api/features", response_model=FeaturesResponse, tags=["meta"])
def features():
    return {**load_flags(), "gnn_available": _loaded()["gnn"]}


for r in (network, simulate, analyze, metrics, scenarios, stream, mitigation, rag, events):
    app.include_router(r.router)
