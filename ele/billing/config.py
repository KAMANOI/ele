"""Billing configuration loaded from environment variables."""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

# Load .env file if present (no-op in production if already set via environment)
load_dotenv()


class BillingConfig:
    """Runtime billing configuration.  All values come from environment variables."""

    # Stripe
    stripe_secret_key:      str = os.environ.get("STRIPE_SECRET_KEY", "")
    stripe_publishable_key: str = os.environ.get("STRIPE_PUBLISHABLE_KEY", "")
    stripe_webhook_secret:  str = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

    # Stripe Price IDs for credit packs
    stripe_price_creator_10:  str = os.environ.get("STRIPE_PRICE_CREATOR_10", "")
    stripe_price_creator_50:  str = os.environ.get("STRIPE_PRICE_CREATOR_50", "")
    stripe_price_creator_200: str = os.environ.get("STRIPE_PRICE_CREATOR_200", "")

    # Stripe Price ID for Pro monthly subscription
    stripe_price_pro_monthly: str = os.environ.get("STRIPE_PRICE_PRO_MONTHLY", "")

    # App
    app_base_url:   str = os.environ.get("APP_BASE_URL", "http://localhost:8000")
    session_secret: str = os.environ.get("SESSION_SECRET", "change-this-in-production-please")
    database_url:   str = os.environ.get("DATABASE_URL", "sqlite:///./storage/ele.db")

    # Admin — 2FA login (password + TOTP).
    # ADMIN_TOKEN is kept only for the one-time /admin/setup bootstrap page.
    admin_token:       str = os.environ.get("ADMIN_TOKEN", "")
    admin_password:    str = os.environ.get("ADMIN_PASSWORD", "")
    admin_totp_secret: str = os.environ.get("ADMIN_TOTP_SECRET", "")


@lru_cache(maxsize=1)
def get_config() -> BillingConfig:
    return BillingConfig()
