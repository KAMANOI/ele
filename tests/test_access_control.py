"""Integration tests for web access control and auth routes.

Tests are split into:
  1. Direct access-control logic (can_export / consume) — no HTTP stack
  2. HTTP route smoke tests — auth pages render, anonymous upload is blocked
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from sqlalchemy.pool import StaticPool

from ele.billing.db import Base, get_db
from ele.billing.auth import create_user
from ele.billing.models import User
from ele.billing import services as svc
from ele.api.app import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="function")
def test_db_session():
    # StaticPool keeps the same in-memory connection across the entire test,
    # which is required because SQLite :memory: databases are per-connection.
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture(scope="function")
def client(test_db_session):
    """TestClient with overridden DB dependency and cookie persistence."""
    def override_get_db():
        yield test_db_session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app, raise_server_exceptions=True, follow_redirects=True) as c:
        yield c, test_db_session

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg_bytes() -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (100, 100), color=(128, 128, 128)).save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Access control logic (no HTTP stack — already tested in test_billing.py,
# but kept here as a cross-check with the same fixture approach)
# ---------------------------------------------------------------------------

def test_can_export_anonymous_blocked(test_db_session):
    can, reason = svc.can_export(None, "standard")
    assert not can
    assert reason == "login_required"


def test_can_export_no_plan_blocked(test_db_session):
    user = create_user(test_db_session, "noPlan@test.com", "password123")
    can, reason = svc.can_export(user, "standard")
    assert not can
    assert reason == "no_plan"


def test_can_export_creator_enough_credits(test_db_session):
    user = create_user(test_db_session, "creator@test.com", "password123")
    user.plan_type = "creator"
    user.credits   = 5
    test_db_session.commit()
    can, reason = svc.can_export(user, "standard")
    assert can


def test_can_export_creator_no_credits(test_db_session):
    user = create_user(test_db_session, "broke@test.com", "password123")
    user.plan_type = "creator"
    user.credits   = 0
    test_db_session.commit()
    can, reason = svc.can_export(user, "standard")
    assert not can
    assert reason == "insufficient_credits"


def test_can_export_pro_active(test_db_session):
    user = create_user(test_db_session, "pro@test.com", "password123")
    user.plan_type          = "pro"
    user.subscription_status = "active"
    test_db_session.commit()
    can, reason = svc.can_export(user, "print")
    assert can


def test_can_export_ambassador(test_db_session):
    user = create_user(test_db_session, "amb@test.com", "password123")
    user.plan_type = "ambassador"
    test_db_session.commit()
    can, reason = svc.can_export(user, "print")
    assert can


def test_consume_credit_on_export(test_db_session):
    user = create_user(test_db_session, "spend@test.com", "password123")
    user.plan_type = "creator"
    user.credits   = 10
    test_db_session.commit()

    svc.consume_export_credit(test_db_session, user, "standard", "job_abc")
    test_db_session.refresh(user)
    assert user.credits == 9

    ledger = svc.get_recent_ledger(test_db_session, user)
    assert ledger[0].delta == -1
    assert ledger[0].reason == "standard_export"


def test_print_costs_3_credits(test_db_session):
    user = create_user(test_db_session, "print@test.com", "password123")
    user.plan_type = "creator"
    user.credits   = 10
    test_db_session.commit()

    svc.consume_export_credit(test_db_session, user, "print", "job_xyz")
    test_db_session.refresh(user)
    assert user.credits == 7


def test_pro_no_credits_consumed(test_db_session):
    user = create_user(test_db_session, "pronospend@test.com", "password123")
    user.plan_type          = "pro"
    user.subscription_status = "active"
    user.credits            = 0
    test_db_session.commit()

    svc.consume_export_credit(test_db_session, user, "standard")
    test_db_session.refresh(user)
    assert user.credits == 0


# ---------------------------------------------------------------------------
# HTTP route smoke tests
# ---------------------------------------------------------------------------

def test_home_page_renders(client):
    c, db = client
    resp = c.get("/")
    assert resp.status_code == 200
    assert "ele" in resp.text


def test_pricing_page_renders(client):
    c, db = client
    resp = c.get("/pricing")
    assert resp.status_code == 200
    assert "Creator" in resp.text
    assert "Pro" in resp.text
    assert "Ambassador" in resp.text


def test_login_page_renders(client):
    c, db = client
    resp = c.get("/login")
    assert resp.status_code == 200
    assert "Log in" in resp.text


def test_signup_page_renders(client):
    c, db = client
    resp = c.get("/signup")
    assert resp.status_code == 200
    assert "Sign up" in resp.text or "Create" in resp.text


def test_account_redirect_when_not_logged_in(client):
    c, db = client
    resp = c.get("/account")
    # Should redirect to login (follow_redirects=True, so final page is login)
    assert resp.status_code == 200
    assert "Log in" in resp.text or "login" in resp.url


def test_anonymous_upload_blocked(client):
    c, db = client
    resp = c.post(
        "/upload",
        files={"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")},
        data={"mode": "creator", "flow": "quick"},
        # follow_redirects=True by default for this client
    )
    # Either shown login page or redirected there
    assert resp.status_code == 200
    assert "log in" in resp.text.lower() or "login" in resp.url or "pricing" in resp.url


def test_signup_creates_account(client):
    c, db = client
    resp = c.post(
        "/signup",
        data={"email": "newsignup@test.com", "password": "password123"},
    )
    assert resp.status_code == 200  # after redirect, lands on pricing
    user = svc.get_user_by_email(db, "newsignup@test.com")
    assert user is not None
    assert user.plan_type == "none"


def test_signup_duplicate_email_rejected(client):
    c, db = client
    create_user(db, "dup@test.com", "password123")

    resp = c.post(
        "/signup",
        data={"email": "dup@test.com", "password": "password123"},
    )
    assert resp.status_code == 400


def test_login_wrong_password_rejected(client):
    c, db = client
    create_user(db, "logintest@test.com", "correctpassword")

    resp = c.post(
        "/login",
        data={"email": "logintest@test.com", "password": "wrongpassword"},
    )
    assert resp.status_code == 401


def test_short_password_rejected_on_signup(client):
    c, db = client
    resp = c.post(
        "/signup",
        data={"email": "short@test.com", "password": "abc"},
    )
    assert resp.status_code == 400


def test_add_credits_and_check_balance(test_db_session):
    user = create_user(test_db_session, "addtest@test.com", "password123")
    svc.add_credits(test_db_session, user, 50, "credit_purchase")
    test_db_session.refresh(user)
    assert user.credits == 50
    assert user.plan_type == "none"  # plan is not changed by adding credits alone


def test_ambassador_key_redemption_flow(test_db_session):
    from ele.billing.models import AmbassadorKey
    from datetime import datetime

    user = create_user(test_db_session, "redeem@test.com", "password123")
    key  = AmbassadorKey(key_value="AMB-REDEMPTIONTEST", is_active=True)
    test_db_session.add(key)
    test_db_session.commit()

    # Simulate redemption
    key.redeemed_by_user_id = user.id
    key.redeemed_at         = datetime.utcnow()
    user.plan_type          = "ambassador"
    user.ambassador_key_id  = key.id
    test_db_session.commit()

    test_db_session.refresh(user)
    can, reason = svc.can_export(user, "print")
    assert can
    assert user.plan_type == "ambassador"
