"""Billing HTTP routes — auth, checkout, account, webhooks, admin."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Optional

import stripe
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ele.billing import services as svc
from ele.billing import stripe_service
from ele.billing.auth import (
    authenticate_user,
    create_user,
    get_current_user_from_session,
    login_session,
    logout_session,
)
from ele.billing.config import get_config
from ele.billing.db import get_db
from ele.billing.models import AmbassadorKey, DiscountCode

log = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Login rate limiting — in-memory, per IP address
# Max 10 failures per 15-minute window. Satisfies Stripe security requirement #6.
# ---------------------------------------------------------------------------

_RATE_LIMIT_MAX      = 10          # max failed attempts
_RATE_LIMIT_WINDOW   = timedelta(minutes=15)
_RATE_LIMIT_LOCKOUT  = timedelta(minutes=15)

# { ip: [datetime, ...] }  — stores timestamps of failed attempts
_failed_attempts: dict[str, list[datetime]] = defaultdict(list)


def _client_ip(request: Request) -> str:
    """Return the client IP, respecting X-Forwarded-For from Render's proxy."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _is_rate_limited(ip: str) -> bool:
    """Return True if this IP has exceeded the failed-login threshold."""
    now = datetime.utcnow()
    cutoff = now - _RATE_LIMIT_WINDOW
    # Evict old entries
    _failed_attempts[ip] = [t for t in _failed_attempts[ip] if t > cutoff]
    return len(_failed_attempts[ip]) >= _RATE_LIMIT_MAX


def _record_failure(ip: str) -> None:
    _failed_attempts[ip].append(datetime.utcnow())


def _clear_failures(ip: str) -> None:
    _failed_attempts.pop(ip, None)

_TEMPLATES_DIR = Path(__file__).parent.parent / "web" / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _render(
    request: Request,
    template: str,
    ctx: dict,
    status_code: int = 200,
) -> HTMLResponse:
    return templates.TemplateResponse(request, template, ctx, status_code=status_code)


def _prices() -> dict:
    cfg = get_config()
    return {
        "creator_10":  cfg.stripe_price_creator_10,
        "creator_50":  cfg.stripe_price_creator_50,
        "creator_200": cfg.stripe_price_creator_200,
        "pro_monthly": cfg.stripe_price_pro_monthly,
    }


# ---------------------------------------------------------------------------
# Auth — login / signup / logout
# ---------------------------------------------------------------------------

@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    user = get_current_user_from_session(request, db)
    if user:
        return RedirectResponse("/account", status_code=302)
    return _render(request, "login.html", {
        "user":     None,
        "next_url": request.query_params.get("next", ""),
    })


@router.post("/login", response_model=None)
async def login_post(
    request:  Request,
    email:    Annotated[str, Form()],
    password: Annotated[str, Form()],
    next_url: Annotated[str, Form()] = "",
    db:       Session = Depends(get_db),
) -> RedirectResponse | HTMLResponse:
    ip = _client_ip(request)

    if _is_rate_limited(ip):
        log.warning("Login rate limit exceeded for IP %s", ip)
        return _render(request, "login.html", {
            "user":     None,
            "error":    "ログイン試行回数の上限に達しました。15分後に再試行してください。",
            "email":    email,
            "next_url": next_url,
        }, status_code=429)

    user = authenticate_user(db, email, password)
    if not user:
        _record_failure(ip)
        return _render(request, "login.html", {
            "user":     None,
            "error":    "Invalid email or password.",
            "email":    email,
            "next_url": next_url,
        }, status_code=401)

    _clear_failures(ip)
    login_session(request, user)
    return RedirectResponse(next_url or "/account", status_code=303)


@router.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    user = get_current_user_from_session(request, db)
    if user:
        return RedirectResponse("/account", status_code=302)
    return _render(request, "signup.html", {"user": None})


@router.post("/signup", response_model=None)
async def signup_post(
    request:  Request,
    email:    Annotated[str, Form()],
    password: Annotated[str, Form()],
    db:       Session = Depends(get_db),
) -> RedirectResponse | HTMLResponse:
    ip = _client_ip(request)

    if _is_rate_limited(ip):
        return _render(request, "signup.html", {
            "user":  None,
            "error": "試行回数の上限に達しました。15分後に再試行してください。",
            "email": email,
        }, status_code=429)

    if len(password) < 8:
        _record_failure(ip)
        return _render(request, "signup.html", {
            "user":  None,
            "error": "Password must be at least 8 characters.",
            "email": email,
        }, status_code=400)

    if svc.get_user_by_email(db, email):
        return _render(request, "signup.html", {
            "user":  None,
            "error": "An account with that email already exists.",
            "email": email,
        }, status_code=400)

    user = create_user(db, email, password)
    login_session(request, user)
    return RedirectResponse("/pricing", status_code=303)


@router.post("/logout")
async def logout_post(request: Request) -> RedirectResponse:
    logout_session(request)
    return RedirectResponse("/", status_code=303)


# ---------------------------------------------------------------------------
# Ambassador key redemption
# ---------------------------------------------------------------------------

@router.post("/redeem-ambassador-key", response_model=None)
async def redeem_ambassador_key(
    request:   Request,
    key_value: Annotated[str, Form()],
    db:        Session = Depends(get_db),
) -> RedirectResponse | HTMLResponse:
    user = get_current_user_from_session(request, db)
    if not user:
        return RedirectResponse("/login?next=/account", status_code=303)

    key = (
        db.query(AmbassadorKey)
        .filter(AmbassadorKey.key_value == key_value.strip().upper())
        .first()
    )

    if not key or not key.is_active:
        return _render(request, "account.html", {
            "user":   user,
            "ledger": svc.get_recent_ledger(db, user),
            "error":  "Invalid or inactive ambassador key.",
        }, status_code=400)

    if key.redeemed_by_user_id is not None:
        return _render(request, "account.html", {
            "user":   user,
            "ledger": svc.get_recent_ledger(db, user),
            "error":  "This ambassador key has already been redeemed.",
        }, status_code=400)

    key.redeemed_by_user_id = user.id
    key.redeemed_at         = datetime.utcnow()
    user.plan_type          = "ambassador"
    user.ambassador_key_id  = key.id
    db.commit()
    db.refresh(user)

    return _render(request, "account.html", {
        "user":    user,
        "ledger":  svc.get_recent_ledger(db, user),
        "success": "Ambassador access activated. Unlimited exports enabled.",
    })


# ---------------------------------------------------------------------------
# Account page
# ---------------------------------------------------------------------------

@router.get("/account", response_class=HTMLResponse)
def account_page(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    user = get_current_user_from_session(request, db)
    if not user:
        return RedirectResponse("/login?next=/account", status_code=302)
    return _render(request, "account.html", {
        "user":   user,
        "ledger": svc.get_recent_ledger(db, user),
    })


# ---------------------------------------------------------------------------
# Pricing page
# ---------------------------------------------------------------------------

@router.get("/pricing", response_class=HTMLResponse)
def pricing_page(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    user = get_current_user_from_session(request, db)
    return _render(request, "pricing.html", {
        "user":   user,
        "prices": _prices(),
    })


# ---------------------------------------------------------------------------
# Checkout
# ---------------------------------------------------------------------------

def _allowed_price_ids(checkout_type: str) -> set[str]:
    """Return the set of configured price IDs valid for the given checkout_type."""
    cfg = get_config()
    if checkout_type == "creator_pack":
        return {
            p for p in (
                cfg.stripe_price_creator_10,
                cfg.stripe_price_creator_50,
                cfg.stripe_price_creator_200,
            ) if p
        }
    if checkout_type == "pro_subscription":
        return {p for p in (cfg.stripe_price_pro_monthly,) if p}
    return set()


@router.post("/billing/create-checkout-session", response_model=None)
async def create_checkout_session(
    request:       Request,
    checkout_type: Annotated[str, Form()],
    price_id:      Annotated[str, Form()],
    discount_code: Annotated[str, Form()] = "",
    db:            Session = Depends(get_db),
) -> RedirectResponse | HTMLResponse:
    user = get_current_user_from_session(request, db)
    if not user:
        return RedirectResponse("/login?next=/pricing", status_code=303)

    if checkout_type not in ("creator_pack", "pro_subscription"):
        return _render(request, "pricing.html", {
            "user":   user,
            "error":  "Invalid checkout type.",
            "prices": _prices(),
        }, status_code=400)

    # Validate price_id against the whitelist for this checkout_type
    allowed = _allowed_price_ids(checkout_type)
    if not allowed or price_id not in allowed:
        log.warning(
            "Rejected checkout: price_id %r not in allowed set for %s (user#%d)",
            price_id, checkout_type, user.id,
        )
        return _render(request, "pricing.html", {
            "user":   user,
            "error":  "Invalid price selection. Please try again.",
            "prices": _prices(),
        }, status_code=400)

    # Validate discount code if provided
    dc: DiscountCode | None = None
    if discount_code.strip():
        dc = (
            db.query(DiscountCode)
            .filter(DiscountCode.code == discount_code.strip().upper())
            .first()
        )
        if not dc or not dc.is_active:
            return _render(request, "pricing.html", {
                "user":   user,
                "error":  f"Discount code '{discount_code}' is not valid.",
                "prices": _prices(),
            }, status_code=400)
        if dc.max_uses is not None and dc.current_uses >= dc.max_uses:
            return _render(request, "pricing.html", {
                "user":   user,
                "error":  f"Discount code '{discount_code}' has reached its usage limit.",
                "prices": _prices(),
            }, status_code=400)
        if dc.expires_at and dc.expires_at < datetime.utcnow():
            return _render(request, "pricing.html", {
                "user":   user,
                "error":  f"Discount code '{discount_code}' has expired.",
                "prices": _prices(),
            }, status_code=400)

    if not price_id:
        return _render(request, "pricing.html", {
            "user":   user,
            "error":  "Stripe is not configured. Please contact support.",
            "prices": _prices(),
        }, status_code=400)

    try:
        checkout_url = stripe_service.create_checkout_session(
            db=db,
            user=user,
            checkout_type=checkout_type,
            price_id=price_id,
            discount_code=dc,
        )
    except stripe.StripeError as exc:
        log.error("Stripe error creating checkout session: %s", exc)
        return _render(request, "pricing.html", {
            "user":   user,
            "error":  "Payment system error. Please try again or contact support.",
            "prices": _prices(),
        }, status_code=500)

    return RedirectResponse(checkout_url, status_code=303)


# ---------------------------------------------------------------------------
# Stripe webhook
# ---------------------------------------------------------------------------

@router.post("/billing/webhook")
async def stripe_webhook(
    request: Request,
    db:      Session = Depends(get_db),
):
    cfg     = get_config()
    payload = await request.body()
    sig     = request.headers.get("stripe-signature", "")

    if not cfg.stripe_webhook_secret:
        log.error("STRIPE_WEBHOOK_SECRET not configured — rejecting webhook")
        raise HTTPException(status_code=500, detail="Webhook secret not configured")

    try:
        event = stripe.Webhook.construct_event(payload, sig, cfg.stripe_webhook_secret)
    except stripe.errors.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    event_type = event["type"]
    log.info("Stripe webhook received: %s", event_type)

    try:
        if event_type == "checkout.session.completed":
            stripe_service.handle_checkout_completed(db, event)
        elif event_type in (
            "customer.subscription.created",
            "customer.subscription.updated",
        ):
            stripe_service.handle_subscription_created_or_updated(db, event)
        elif event_type == "customer.subscription.deleted":
            stripe_service.handle_subscription_deleted(db, event)
        elif event_type == "invoice.payment_failed":
            stripe_service.handle_invoice_payment_failed(db, event)
        elif event_type == "invoice.paid":
            pass  # Subscription renewal handled via subscription.updated
    except Exception as exc:
        log.error("Error handling webhook %s: %s", event_type, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Webhook handler error")

    return {"ok": True}


# ---------------------------------------------------------------------------
# Billing success / cancel
# ---------------------------------------------------------------------------

@router.get("/billing/success", response_class=HTMLResponse)
def billing_success(
    request: Request,
    db:      Session = Depends(get_db),
) -> HTMLResponse:
    user = get_current_user_from_session(request, db)
    if not user:
        return RedirectResponse("/login", status_code=302)
    db.refresh(user)
    return _render(request, "billing_success.html", {"user": user})


@router.get("/billing/cancel", response_class=HTMLResponse)
def billing_cancel(
    request: Request,
    db:      Session = Depends(get_db),
) -> HTMLResponse:
    user = get_current_user_from_session(request, db)
    return _render(request, "billing_cancel.html", {"user": user})


# ---------------------------------------------------------------------------
# Admin — 2FA login (password + TOTP) + account lockout
# ---------------------------------------------------------------------------

# Track failed admin login attempts per IP (max 10, then 15-min lockout)
_admin_failures: dict[str, list[datetime]] = defaultdict(list)
_ADMIN_MAX_FAILURES = 10
_ADMIN_LOCKOUT      = timedelta(minutes=15)


def _admin_is_locked(ip: str) -> bool:
    now    = datetime.utcnow()
    cutoff = now - _ADMIN_LOCKOUT
    _admin_failures[ip] = [t for t in _admin_failures[ip] if t > cutoff]
    return len(_admin_failures[ip]) >= _ADMIN_MAX_FAILURES


def _admin_record_failure(ip: str) -> None:
    _admin_failures[ip].append(datetime.utcnow())


def _admin_clear(ip: str) -> None:
    _admin_failures.pop(ip, None)


def _admin_authenticated(request: Request) -> bool:
    return request.session.get("admin_auth") is True


@router.get("/admin/login", response_class=HTMLResponse)
def admin_login_page(request: Request) -> HTMLResponse:
    if _admin_authenticated(request):
        return RedirectResponse("/admin/keys", status_code=302)
    ip = _client_ip(request)
    return _render(request, "admin_login.html", {
        "user":   None,
        "locked": _admin_is_locked(ip),
        "error":  None,
    })


@router.post("/admin/login", response_model=None)
async def admin_login_post(
    request:   Request,
    password:  Annotated[str, Form()],
    totp_code: Annotated[str, Form()],
) -> RedirectResponse | HTMLResponse:
    import pyotp

    cfg = get_config()
    ip  = _client_ip(request)

    if _admin_is_locked(ip):
        return _render(request, "admin_login.html", {
            "user": None, "locked": True, "error": None,
        }, status_code=429)

    # Verify password
    password_ok = cfg.admin_password and password == cfg.admin_password

    # Verify TOTP
    totp_ok = False
    if cfg.admin_totp_secret and totp_code.strip():
        totp = pyotp.TOTP(cfg.admin_totp_secret)
        totp_ok = totp.verify(totp_code.strip(), valid_window=1)

    if not (password_ok and totp_ok):
        _admin_record_failure(ip)
        remaining = _ADMIN_MAX_FAILURES - len(_admin_failures[ip])
        log.warning("Admin login failed for IP %s (remaining attempts: %d)", ip, max(remaining, 0))
        return _render(request, "admin_login.html", {
            "user":   None,
            "locked": _admin_is_locked(ip),
            "error":  f"認証に失敗しました。残り試行回数: {max(remaining, 0)}",
        }, status_code=401)

    _admin_clear(ip)
    request.session["admin_auth"] = True
    log.info("Admin login successful from IP %s", ip)
    return RedirectResponse("/admin/keys", status_code=303)


@router.post("/admin/logout")
async def admin_logout(request: Request) -> RedirectResponse:
    request.session.pop("admin_auth", None)
    return RedirectResponse("/admin/login", status_code=303)


@router.get("/admin/setup", response_class=HTMLResponse)
def admin_setup_page(request: Request) -> HTMLResponse:
    """One-time TOTP setup page. Protected by ADMIN_TOKEN query param."""
    import pyotp

    cfg   = get_config()
    token = request.query_params.get("token", "")
    if not cfg.admin_token or token != cfg.admin_token:
        raise HTTPException(status_code=403, detail="Forbidden")

    secret = cfg.admin_totp_secret or pyotp.random_base32()
    totp   = pyotp.TOTP(secret)
    uri    = totp.provisioning_uri(name="admin", issuer_name="ele")

    return _render(request, "admin_setup.html", {
        "user":             None,
        "totp_secret":      secret,
        "provisioning_uri": uri,
        "current_code":     totp.now(),
    })


@router.get("/admin/keys", response_class=HTMLResponse)
def admin_keys_page(
    request: Request,
    db:      Session = Depends(get_db),
) -> HTMLResponse:
    if not _admin_authenticated(request):
        return RedirectResponse("/admin/login", status_code=302)

    keys  = db.query(AmbassadorKey).order_by(AmbassadorKey.created_at.desc()).all()
    codes = db.query(DiscountCode).order_by(DiscountCode.created_at.desc()).all()

    return _render(request, "admin_keys.html", {
        "user":  get_current_user_from_session(request, db),
        "keys":  keys,
        "codes": codes,
        "token": "",
    })
