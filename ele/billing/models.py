"""SQLAlchemy ORM models for ele billing."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ele.billing.db import Base


class User(Base):
    __tablename__ = "users"

    id:                     Mapped[int]        = mapped_column(Integer, primary_key=True, index=True)
    email:                  Mapped[str]        = mapped_column(String(255), unique=True, index=True, nullable=False)
    password_hash:          Mapped[str]        = mapped_column(String(255), nullable=False)
    plan_type:              Mapped[str]        = mapped_column(String(32),  default="none", nullable=False)
    credits:                Mapped[int]        = mapped_column(Integer,     default=0,      nullable=False)
    ambassador_key_id:      Mapped[int | None] = mapped_column(Integer,     ForeignKey("ambassador_keys.id"), nullable=True)
    stripe_customer_id:     Mapped[str | None] = mapped_column(String(255), nullable=True)
    stripe_subscription_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    subscription_status:    Mapped[str | None] = mapped_column(String(64),  nullable=True)
    created_at:             Mapped[datetime]   = mapped_column(DateTime,    server_default=func.now())
    updated_at:             Mapped[datetime]   = mapped_column(DateTime,    server_default=func.now(), onupdate=func.now())

    ledger_entries: Mapped[list[CreditLedger]] = relationship("CreditLedger", back_populates="user")


class AmbassadorKey(Base):
    __tablename__ = "ambassador_keys"

    id:                  Mapped[int]            = mapped_column(Integer, primary_key=True, index=True)
    key_value:           Mapped[str]            = mapped_column(String(128), unique=True, index=True, nullable=False)
    label:               Mapped[str | None]     = mapped_column(String(255), nullable=True)
    is_active:           Mapped[bool]           = mapped_column(Boolean,     default=True, nullable=False)
    created_at:          Mapped[datetime]       = mapped_column(DateTime,    server_default=func.now())
    redeemed_by_user_id: Mapped[int | None]     = mapped_column(Integer,     ForeignKey("users.id"), nullable=True)
    redeemed_at:         Mapped[datetime | None] = mapped_column(DateTime,   nullable=True)


class DiscountCode(Base):
    __tablename__ = "discount_codes"

    id:               Mapped[int]            = mapped_column(Integer,     primary_key=True, index=True)
    code:             Mapped[str]            = mapped_column(String(64),  unique=True, index=True, nullable=False)
    kind:             Mapped[str]            = mapped_column(String(32),  default="percent", nullable=False)
    percent_off:      Mapped[int]            = mapped_column(Integer,     nullable=False)
    is_active:        Mapped[bool]           = mapped_column(Boolean,     default=True, nullable=False)
    max_uses:         Mapped[int | None]     = mapped_column(Integer,     nullable=True)
    current_uses:     Mapped[int]            = mapped_column(Integer,     default=0, nullable=False)
    expires_at:       Mapped[datetime | None] = mapped_column(DateTime,   nullable=True)
    stripe_coupon_id: Mapped[str | None]     = mapped_column(String(255), nullable=True)
    created_at:       Mapped[datetime]       = mapped_column(DateTime,    server_default=func.now())


class CreditLedger(Base):
    __tablename__ = "credit_ledger"

    id:            Mapped[int]        = mapped_column(Integer,  primary_key=True, index=True)
    user_id:       Mapped[int]        = mapped_column(Integer,  ForeignKey("users.id"), nullable=False)
    delta:         Mapped[int]        = mapped_column(Integer,  nullable=False)
    reason:        Mapped[str]        = mapped_column(String(128), nullable=False)
    metadata_json: Mapped[str | None] = mapped_column(Text,     nullable=True)
    created_at:    Mapped[datetime]   = mapped_column(DateTime, server_default=func.now())

    user: Mapped[User] = relationship("User", back_populates="ledger_entries")


class CheckoutSessionLog(Base):
    __tablename__ = "checkout_session_logs"

    id:                    Mapped[int]        = mapped_column(Integer,     primary_key=True, index=True)
    user_id:               Mapped[int | None] = mapped_column(Integer,     ForeignKey("users.id"), nullable=True)
    stripe_session_id:     Mapped[str]        = mapped_column(String(255), unique=True, nullable=False)
    checkout_type:         Mapped[str]        = mapped_column(String(64),  nullable=False)
    price_lookup_key:      Mapped[str]        = mapped_column(String(255), nullable=False)
    discount_code_entered: Mapped[str | None] = mapped_column(String(64),  nullable=True)
    status:                Mapped[str]        = mapped_column(String(32),  default="pending", nullable=False)
    created_at:            Mapped[datetime]   = mapped_column(DateTime,    server_default=func.now())
