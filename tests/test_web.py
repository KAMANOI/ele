"""Web app smoke tests.

Tests that exercise the full upload/export/download pipeline run with an
ambassador-plan user so the billing access check always passes.
Tests that are purely UI / health checks run anonymously.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ele.api.app import app
from ele.billing.auth import create_user
from ele.billing.db import Base, get_db


# ---------------------------------------------------------------------------
# Client fixtures
# ---------------------------------------------------------------------------

def _make_jpeg_bytes(size: tuple[int, int] = (64, 64)) -> bytes:
    arr = (np.random.default_rng(7).random((*size, 3)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _make_test_engine():
    return create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


@pytest.fixture(scope="module")
def authed_client():
    """TestClient with an ambassador-plan user logged in.

    Scope is module-level for speed; the in-memory DB and session are shared
    across all tests in this module.
    """
    engine = _make_test_engine()
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    # Create an ambassador user for unblocked exports
    user = create_user(db, "webtest@example.com", "password123")
    user.plan_type = "ambassador"
    db.commit()

    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app, follow_redirects=True) as c:
        # Log in so session cookie is set
        c.post("/login", data={"email": "webtest@example.com", "password": "password123"})
        yield c, db

    app.dependency_overrides.clear()
    db.close()
    engine.dispose()


# Anonymous client (no auth) for tests that don't need it
anon_client = TestClient(app)


# ---------------------------------------------------------------------------
# Health / home (anonymous OK)
# ---------------------------------------------------------------------------

def test_health() -> None:
    r = anon_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_home_returns_html() -> None:
    r = anon_client.get("/")
    assert r.status_code == 200
    assert "ele" in r.text.lower()
    assert "text/html" in r.headers["content-type"]


def test_result_page_not_found() -> None:
    r = anon_client.get("/result/doesnotexist")
    assert r.status_code == 404


def test_download_not_found() -> None:
    r = anon_client.get("/download/doesnotexist")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Upload validation (extension check runs before auth check)
# ---------------------------------------------------------------------------

def test_upload_invalid_extension_rejected(authed_client) -> None:
    c, _ = authed_client
    r = c.post(
        "/upload",
        files={"file": ("photo.bmp", b"not an image", "image/bmp")},
        data={"mode": "creator", "flow": "quick"},
    )
    assert r.status_code == 400
    assert "unsupported" in r.text.lower()


# ---------------------------------------------------------------------------
# Upload + pipeline + result (require auth)
# ---------------------------------------------------------------------------

def test_upload_valid_jpeg_quick_export(authed_client) -> None:
    c, _ = authed_client
    jpeg = _make_jpeg_bytes()
    r = c.post(
        "/upload",
        files={"file": ("test.jpg", jpeg, "image/jpeg")},
        data={"mode": "free", "flow": "quick"},
    )
    assert r.status_code == 200
    assert "pseudo-raw master ready" in r.text.lower()


def test_upload_valid_jpeg_preview_flow(authed_client) -> None:
    c, _ = authed_client
    jpeg = _make_jpeg_bytes()
    r = c.post(
        "/upload",
        files={"file": ("test.jpg", jpeg, "image/jpeg")},
        data={"mode": "creator", "flow": "preview"},
    )
    assert r.status_code == 200
    assert "export target" in r.text.lower()


def test_export_target_lightroom(authed_client) -> None:
    """Upload in preview flow then export to Lightroom target."""
    c, _ = authed_client
    jpeg = _make_jpeg_bytes()

    # Step 1: upload in preview flow (no redirect follow so we get job_id)
    r1 = c.post(
        "/upload",
        files={"file": ("test.jpg", jpeg, "image/jpeg")},
        data={"mode": "creator", "flow": "preview"},
        follow_redirects=False,
    )
    assert r1.status_code == 303
    job_id = r1.headers["location"].split("/")[-1]

    # Step 2: export to Lightroom
    r2 = c.post(
        f"/export/{job_id}",
        data={"target": "lightroom"},
        follow_redirects=True,
    )
    assert r2.status_code == 200
    assert "pseudo-raw master ready" in r2.text.lower()


def test_export_dng_shows_not_implemented(authed_client) -> None:
    """DNG export target should show a clear not-implemented message."""
    c, _ = authed_client
    jpeg = _make_jpeg_bytes()

    r1 = c.post(
        "/upload",
        files={"file": ("test.jpg", jpeg, "image/jpeg")},
        data={"mode": "creator", "flow": "preview"},
        follow_redirects=False,
    )
    job_id = r1.headers["location"].split("/")[-1]

    r2 = c.post(
        f"/export/{job_id}",
        data={"target": "adobe_dng"},
        follow_redirects=True,
    )
    assert r2.status_code == 200
    assert "planned" in r2.text.lower() or "not available" in r2.text.lower()


def test_download_after_quick_export(authed_client) -> None:
    """Download endpoint should return a TIFF file."""
    c, _ = authed_client
    jpeg = _make_jpeg_bytes()

    r1 = c.post(
        "/upload",
        files={"file": ("test.jpg", jpeg, "image/jpeg")},
        data={"mode": "free", "flow": "quick"},
        follow_redirects=False,
    )
    assert r1.status_code == 303
    job_id = r1.headers["location"].split("/")[-1]

    r2 = c.get(f"/download/{job_id}")
    assert r2.status_code == 200
    assert r2.headers["content-type"] == "image/tiff"
    assert len(r2.content) > 0
