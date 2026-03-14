"""Business logic for access control and credit management."""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from ele.billing.models import CreditLedger, User

# Credits consumed per export kind
EXPORT_COSTS: dict[str, int] = {
    "standard": 1,
    "print":    3,
}

_ACTIVE_SUB_STATUSES = {"active", "trialing"}


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------

def can_export(user: User | None, export_kind: str) -> tuple[bool, str | None]:
    """Check whether a user is allowed to perform an export.

    Returns (allowed, reason_if_blocked).

    Possible block reasons:
      "login_required"      — no authenticated user
      "no_plan"             — user has no active plan
      "insufficient_credits"— creator plan but not enough credits
    """
    if user is None:
        return False, "login_required"

    if user.plan_type == "ambassador":
        return True, None

    if user.plan_type == "pro" and user.subscription_status in _ACTIVE_SUB_STATUSES:
        return True, None

    if user.plan_type == "creator":
        cost = EXPORT_COSTS.get(export_kind, 1)
        if user.credits >= cost:
            return True, None
        return False, "insufficient_credits"

    return False, "no_plan"


# ---------------------------------------------------------------------------
# Credit management
# ---------------------------------------------------------------------------

def consume_export_credit(
    db: Session,
    user: User,
    export_kind: str,
    job_id: str = "",
) -> None:
    """Deduct credits from a creator-plan user and write a ledger entry.

    No-op for pro / ambassador users.
    """
    if user.plan_type != "creator":
        return

    cost = EXPORT_COSTS.get(export_kind, 1)
    user.credits -= cost

    entry = CreditLedger(
        user_id=user.id,
        delta=-cost,
        reason=f"{export_kind}_export",
        metadata_json=json.dumps({"job_id": job_id}) if job_id else None,
    )
    db.add(entry)
    db.commit()
    db.refresh(user)


def add_credits(
    db: Session,
    user: User,
    amount: int,
    reason: str,
    metadata: dict | None = None,
) -> None:
    """Add credits and write a ledger entry."""
    user.credits += amount

    entry = CreditLedger(
        user_id=user.id,
        delta=amount,
        reason=reason,
        metadata_json=json.dumps(metadata) if metadata else None,
    )
    db.add(entry)
    db.commit()
    db.refresh(user)


# ---------------------------------------------------------------------------
# User lookups
# ---------------------------------------------------------------------------

def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_email(db: Session, email: str) -> User | None:
    return db.query(User).filter(User.email == email.lower().strip()).first()


def get_user_by_stripe_customer_id(db: Session, stripe_customer_id: str) -> User | None:
    return db.query(User).filter(User.stripe_customer_id == stripe_customer_id).first()


def get_user_by_stripe_subscription_id(db: Session, sub_id: str) -> User | None:
    return db.query(User).filter(User.stripe_subscription_id == sub_id).first()


def get_recent_ledger(db: Session, user: User, limit: int = 20) -> list[CreditLedger]:
    return (
        db.query(CreditLedger)
        .filter(CreditLedger.user_id == user.id)
        .order_by(CreditLedger.created_at.desc())
        .limit(limit)
        .all()
    )
