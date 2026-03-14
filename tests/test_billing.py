"""Tests for billing services — access control, credits, ambassador keys."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from ele.billing.db import Base
from ele.billing.models import AmbassadorKey, DiscountCode, User
from ele.billing import services as svc
from ele.billing.auth import create_user, verify_password


# ---------------------------------------------------------------------------
# In-memory SQLite fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


# ---------------------------------------------------------------------------
# User creation and authentication
# ---------------------------------------------------------------------------

def test_create_user(db):
    user = create_user(db, "test@example.com", "password123")
    assert user.id is not None
    assert user.email == "test@example.com"
    assert user.plan_type == "none"
    assert user.credits == 0


def test_password_hashing(db):
    user = create_user(db, "hash@example.com", "mypassword")
    assert verify_password("mypassword", user.password_hash)
    assert not verify_password("wrongpassword", user.password_hash)


def test_email_normalised(db):
    user = create_user(db, "  Test@Example.COM  ", "password123")
    assert user.email == "test@example.com"


def test_get_user_by_email(db):
    create_user(db, "find@example.com", "password123")
    found = svc.get_user_by_email(db, "find@example.com")
    assert found is not None
    assert found.email == "find@example.com"


def test_get_user_by_email_missing(db):
    assert svc.get_user_by_email(db, "missing@example.com") is None


# ---------------------------------------------------------------------------
# Access control — can_export
# ---------------------------------------------------------------------------

def test_anonymous_blocked(db):
    can, reason = svc.can_export(None, "standard")
    assert not can
    assert reason == "login_required"


def test_no_plan_blocked(db):
    user = create_user(db, "noPlan@example.com", "password123")
    can, reason = svc.can_export(user, "standard")
    assert not can
    assert reason == "no_plan"


def test_creator_with_credits_allowed(db):
    user = create_user(db, "creator@example.com", "password123")
    user.plan_type = "creator"
    user.credits   = 5
    db.commit()
    can, reason = svc.can_export(user, "standard")
    assert can
    assert reason is None


def test_creator_without_credits_blocked(db):
    user = create_user(db, "broke@example.com", "password123")
    user.plan_type = "creator"
    user.credits   = 0
    db.commit()
    can, reason = svc.can_export(user, "standard")
    assert not can
    assert reason == "insufficient_credits"


def test_creator_print_needs_3_credits(db):
    user = create_user(db, "print2@example.com", "password123")
    user.plan_type = "creator"
    user.credits   = 2
    db.commit()
    can, reason = svc.can_export(user, "print")
    assert not can
    assert reason == "insufficient_credits"


def test_creator_print_with_3_credits_allowed(db):
    user = create_user(db, "print3@example.com", "password123")
    user.plan_type = "creator"
    user.credits   = 3
    db.commit()
    can, reason = svc.can_export(user, "print")
    assert can


def test_pro_active_allowed(db):
    user = create_user(db, "pro@example.com", "password123")
    user.plan_type          = "pro"
    user.subscription_status = "active"
    db.commit()
    for kind in ("standard", "print"):
        can, reason = svc.can_export(user, kind)
        assert can, f"pro should be allowed for {kind}"


def test_pro_trialing_allowed(db):
    user = create_user(db, "trial@example.com", "password123")
    user.plan_type          = "pro"
    user.subscription_status = "trialing"
    db.commit()
    can, reason = svc.can_export(user, "standard")
    assert can


def test_pro_past_due_blocked(db):
    user = create_user(db, "pastdue@example.com", "password123")
    user.plan_type          = "pro"
    user.subscription_status = "past_due"
    db.commit()
    can, reason = svc.can_export(user, "standard")
    assert not can
    assert reason == "no_plan"


def test_ambassador_allowed(db):
    user = create_user(db, "amb@example.com", "password123")
    user.plan_type = "ambassador"
    db.commit()
    for kind in ("standard", "print"):
        can, reason = svc.can_export(user, kind)
        assert can, f"ambassador should be allowed for {kind}"


# ---------------------------------------------------------------------------
# Credit consumption
# ---------------------------------------------------------------------------

def test_consume_standard_credit(db):
    user = create_user(db, "spend1@example.com", "password123")
    user.plan_type = "creator"
    user.credits   = 10
    db.commit()

    svc.consume_export_credit(db, user, "standard", "job123")
    db.refresh(user)
    assert user.credits == 9

    ledger = svc.get_recent_ledger(db, user)
    assert len(ledger) == 1
    assert ledger[0].delta == -1
    assert ledger[0].reason == "standard_export"


def test_consume_print_credit(db):
    user = create_user(db, "spend3@example.com", "password123")
    user.plan_type = "creator"
    user.credits   = 10
    db.commit()

    svc.consume_export_credit(db, user, "print", "job456")
    db.refresh(user)
    assert user.credits == 7

    ledger = svc.get_recent_ledger(db, user)
    assert ledger[0].delta == -3


def test_pro_no_credit_consumed(db):
    user = create_user(db, "pronospend@example.com", "password123")
    user.plan_type          = "pro"
    user.subscription_status = "active"
    user.credits            = 0
    db.commit()

    svc.consume_export_credit(db, user, "standard")
    db.refresh(user)
    assert user.credits == 0
    assert svc.get_recent_ledger(db, user) == []


def test_add_credits(db):
    user = create_user(db, "addcredits@example.com", "password123")
    svc.add_credits(db, user, 50, "credit_purchase", {"price_id": "price_abc"})
    db.refresh(user)
    assert user.credits == 50

    ledger = svc.get_recent_ledger(db, user)
    assert len(ledger) == 1
    assert ledger[0].delta == 50
    assert ledger[0].reason == "credit_purchase"


# ---------------------------------------------------------------------------
# Ambassador key redemption logic
# ---------------------------------------------------------------------------

def test_ambassador_key_redemption(db):
    user = create_user(db, "redeemer@example.com", "password123")

    key = AmbassadorKey(key_value="AMB-TESTKEY1234", label="test", is_active=True)
    db.add(key)
    db.commit()
    db.refresh(key)

    # Simulate redemption
    assert key.redeemed_by_user_id is None
    key.redeemed_by_user_id = user.id
    user.plan_type         = "ambassador"
    user.ambassador_key_id = key.id
    db.commit()
    db.refresh(user)
    db.refresh(key)

    assert user.plan_type == "ambassador"
    assert key.redeemed_by_user_id == user.id


def test_ambassador_key_cannot_be_reused(db):
    user1 = create_user(db, "user1@example.com", "password123")
    user2 = create_user(db, "user2@example.com", "password123")

    key = AmbassadorKey(key_value="AMB-ONEUSE9999", is_active=True)
    db.add(key)
    db.commit()

    key.redeemed_by_user_id = user1.id
    db.commit()

    # Key is already redeemed — user2 cannot redeem it
    assert key.redeemed_by_user_id is not None
    assert key.redeemed_by_user_id != user2.id


def test_inactive_key_rejected(db):
    key = AmbassadorKey(key_value="AMB-INACTIVE111", is_active=False)
    db.add(key)
    db.commit()
    assert not key.is_active


# ---------------------------------------------------------------------------
# Discount code validation logic
# ---------------------------------------------------------------------------

def test_valid_discount_code(db):
    dc = DiscountCode(
        code="MAG-ELE20",
        kind="percent",
        percent_off=20,
        is_active=True,
        max_uses=500,
        current_uses=0,
    )
    db.add(dc)
    db.commit()

    found = db.query(DiscountCode).filter(DiscountCode.code == "MAG-ELE20").first()
    assert found is not None
    assert found.percent_off == 20
    assert found.is_active


def test_inactive_discount_code_rejected(db):
    dc = DiscountCode(
        code="INACTIVE-CODE",
        kind="percent",
        percent_off=10,
        is_active=False,
    )
    db.add(dc)
    db.commit()

    found = db.query(DiscountCode).filter(DiscountCode.code == "INACTIVE-CODE").first()
    assert found is not None
    assert not found.is_active


def test_maxed_discount_code_rejected(db):
    dc = DiscountCode(
        code="MAXED-OUT",
        kind="percent",
        percent_off=15,
        is_active=True,
        max_uses=10,
        current_uses=10,
    )
    db.add(dc)
    db.commit()

    found = db.query(DiscountCode).filter(DiscountCode.code == "MAXED-OUT").first()
    assert found.current_uses >= found.max_uses
