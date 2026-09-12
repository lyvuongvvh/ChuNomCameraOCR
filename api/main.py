"""FastAPI app entrypoint. Run with: uvicorn api.main:app --host 0.0.0.0 --port 8000

All heavy dependencies (recognizer weights, LM, reading dict, Anthropic client, detection model)
are constructed once during startup (see api/deps.py) and torn down on shutdown - never
lazily/per-request. A missing required file fails startup loudly, per the Phase 4 plan's §4.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.config import settings
from api.deps import build_dependencies
from api.routers import health, ocr


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.deps = build_dependencies(settings)
    yield
    app.state.deps.translate_pool.shutdown(wait=True)


app = FastAPI(title="ChuNomCameraOCR API", lifespan=lifespan)
app.include_router(health.router)
app.include_router(ocr.router)
