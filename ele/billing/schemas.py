"""Pydantic schemas for billing API responses (minimal set)."""

from __future__ import annotations

from pydantic import BaseModel


class UserPublic(BaseModel):
    id:                  int
    email:               str
    plan_type:           str
    credits:             int
    subscription_status: str | None = None

    model_config = {"from_attributes": True}


class LedgerEntry(BaseModel):
    id:            int
    delta:         int
    reason:        str
    created_at:    str

    model_config = {"from_attributes": True}
