"""Authentication helpers — password hashing and session management."""

from __future__ import annotations

import bcrypt
from fastapi import Request
from sqlalchemy.orm import Session

from ele.billing import services as svc
from ele.billing.models import User


# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# User creation / authentication
# ---------------------------------------------------------------------------

def create_user(db: Session, email: str, password: str) -> User:
    user = User(
        email=email.lower().strip(),
        password_hash=hash_password(password),
        plan_type="none",
        credits=0,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    user = svc.get_user_by_email(db, email)
    if user and verify_password(password, user.password_hash):
        return user
    return None


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def login_session(request: Request, user: User) -> None:
    request.session["user_id"] = user.id


def logout_session(request: Request) -> None:
    request.session.clear()


def get_current_user_from_session(request: Request, db: Session) -> User | None:
    user_id = request.session.get("user_id")
    if not user_id:
        return None
    return svc.get_user_by_id(db, int(user_id))
