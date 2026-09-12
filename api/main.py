"""FastAPI app entrypoint. Run with: uvicorn api.main:app --host 0.0.0.0 --port 8000
(--host 0.0.0.0 so the Phase 5 frontend is reachable from a phone on the same LAN, not just
localhost).

All heavy dependencies (recognizer weights, LM, reading dict, Anthropic client, detection model)
are constructed once during startup (see api/deps.py) and torn down on shutdown - never
lazily/per-request. A missing required file fails startup loudly, per the Phase 4 plan's §4.
"""
from __future__ import annotations

import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.config import settings
from api.deps import build_dependencies
from api.routers import health, ocr

_WEB_DIR = pathlib.Path(__file__).resolve().parent.parent / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.deps = build_dependencies(settings)
    yield
    app.state.deps.translate_pool.shutdown(wait=True)


app = FastAPI(title="ChuNomCameraOCR API", lifespan=lifespan)

# Only matters when the frontend is served from a different origin than this API (e.g. web/ on
# Cloudflare Pages, this API behind a Cloudflare Tunnel) - the default same-origin deployment
# (this app serving web/ itself, via the mount below) never triggers a browser CORS check at all,
# so this is a no-op there. No credentials/cookies are used by this API, so a wildcard default is
# safe; set ALLOWED_ORIGINS to your real Pages URL for a split deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ocr.router)

# Registered last and mounted at "/": Starlette resolves routes in registration order, so the
# specific /healthz and /v1/ocr paths above must come first or this catch-all static mount would
# shadow them. Serves the Phase 5 frontend (web/) from this same process/port when deployed
# together (same-origin, no CORS needed); harmless if you instead host web/ elsewhere (e.g.
# Cloudflare Pages) and never hit "/" on this server at all.
app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")
