"""ele FastAPI application.

Includes the web UI router, billing router, session middleware, and static files.

Start locally:
  uvicorn ele.api.app:app --reload
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from ele.billing.config import get_config
from ele.billing.db import init_db
from ele.billing.routes import router as billing_router
from ele.config import APP_NAME, APP_VERSION
from ele.web.routes import router as web_router
from ele.web.services import ensure_storage

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE         = Path(__file__).parent           # ele/api/
_ELE_PKG      = _HERE.parent                    # ele/
_PROJECT      = _ELE_PKG.parent                 # project root
_STATIC_DIR   = _ELE_PKG / "web" / "static"
_PREVIEWS_DIR = _PROJECT / "storage" / "previews"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

cfg = get_config()

app = FastAPI(
    title=APP_NAME,
    description="Pseudo-RAW preprocessing engine",
    version=APP_VERSION,
)

# Session middleware — must be added before routes that use request.session
app.add_middleware(SessionMiddleware, secret_key=cfg.session_secret)

# Ensure storage directories and DB tables exist on startup
ensure_storage()
init_db()

# Static assets (CSS etc.)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# Browser preview JPEGs served directly
app.mount(
    "/previews",
    StaticFiles(directory=str(_PREVIEWS_DIR)),
    name="previews",
)

# Routes — billing first so /login, /pricing etc. are registered before web /
app.include_router(billing_router)
app.include_router(web_router)
