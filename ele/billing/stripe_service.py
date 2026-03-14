"""Stripe integration — checkout sessions and webhook event handling."""

from __future__ import annotations

import logging

import stripe
from sqlalchemy.orm import Session

from ele.billing.config import get_config
from ele.billing.models import CheckoutSessionLog, DiscountCode, User

log = logging.getLogger(__name__)

# Credits granted per credit pack, keyed by Stripe price ID lookup
CREDIT_PACK_AMOUNTS = {
    "creator_10":  10,
    "creator_50":  50,
    "creator_200": 200,
}

_ACTIVE_SUB_STATUSES = {"active", "trialing"}


def _init_stripe() -> None:
    cfg = get_config()
    if cfg.stripe_secret_key:
        stripe.api_key = cfg.stripe_secret_key


_init_stripe()


# ---------------------------------------------------------------------------
# Stripe Coupon helper (lazily created, cached on DiscountCode row)
# ---------------------------------------------------------------------------

def get_or_create_stripe_coupon(db: Session, code: DiscountCode) -> str:
    """Return the Stripe coupon ID for a discount code, creating it if needed."""
    if code.stripe_coupon_id:
        return code.stripe_coupon_id

    coupon = stripe.Coupon.create(
        percent_off=code.percent_off,
        duration="once",
        name=f"ele {code.code}",
        metadata={"ele_code": code.code},
    )
    code.stripe_coupon_id = coupon["id"]
    db.commit()
    return coupon["id"]


# ---------------------------------------------------------------------------
# Checkout session creation
# ---------------------------------------------------------------------------

def create_checkout_session(
    db: Session,
    user: User,
    checkout_type: str,
    price_id: str,
    discount_code: DiscountCode | None = None,
) -> str:
    """Create a Stripe Checkout Session and return its redirect URL."""
    cfg = get_config()
    mode = "payment" if checkout_type == "creator_pack" else "subscription"

    discounts: list[dict] = []
    if discount_code:
        coupon_id = get_or_create_stripe_coupon(db, discount_code)
        discounts = [{"coupon": coupon_id}]

    session_kwargs: dict = {
        "mode": mode,
        "line_items": [{"price": price_id, "quantity": 1}],
        "success_url": (
            f"{cfg.app_base_url}/billing/success"
            "?session_id={CHECKOUT_SESSION_ID}"
        ),
        "cancel_url": f"{cfg.app_base_url}/billing/cancel",
        "metadata": {
            "user_id":       str(user.id),
            "checkout_type": checkout_type,
            "price_id":      price_id,
            "discount_code": discount_code.code if discount_code else "",
        },
    }

    if user.stripe_customer_id:
        session_kwargs["customer"] = user.stripe_customer_id
    else:
        session_kwargs["customer_email"] = user.email

    if discounts:
        session_kwargs["discounts"] = discounts

    if mode == "subscription":
        session_kwargs["subscription_data"] = {
            "metadata": {"user_id": str(user.id)},
        }

    session = stripe.checkout.Session.create(**session_kwargs)

    # Persist a log entry
    log_entry = CheckoutSessionLog(
        user_id=user.id,
        stripe_session_id=session["id"],
        checkout_type=checkout_type,
        price_lookup_key=price_id,
        discount_code_entered=discount_code.code if discount_code else None,
        status="pending",
    )
    db.add(log_entry)
    db.commit()

    return session["url"]


# ---------------------------------------------------------------------------
# Webhook handlers
# ---------------------------------------------------------------------------

def handle_checkout_completed(db: Session, event: dict) -> None:
    """checkout.session.completed — fulfil credit pack purchases."""
    from ele.billing import services as svc

    session_obj = event["data"]["object"]
    meta        = session_obj.get("metadata", {})
    user_id_str = meta.get("user_id")
    if not user_id_str:
        return

    user = svc.get_user_by_id(db, int(user_id_str))
    if not user:
        return

    # Cache stripe_customer_id if we don't have it yet
    customer_id = session_obj.get("customer")
    if customer_id and not user.stripe_customer_id:
        user.stripe_customer_id = customer_id
        db.commit()

    # Update log entry status
    stripe_session_id = session_obj.get("id", "")
    log_entry = (
        db.query(CheckoutSessionLog)
        .filter(CheckoutSessionLog.stripe_session_id == stripe_session_id)
        .first()
    )
    if log_entry:
        log_entry.status = "completed"
        db.commit()

    checkout_type = meta.get("checkout_type", "")
    price_id      = meta.get("price_id", "")

    if checkout_type == "creator_pack":
        pack_key       = _resolve_pack_key(price_id)
        credits_amount = CREDIT_PACK_AMOUNTS.get(pack_key, 0)
        if credits_amount > 0:
            # Ensure plan is at least creator
            if user.plan_type == "none":
                user.plan_type = "creator"
                db.commit()
            svc.add_credits(db, user, credits_amount, "credit_purchase", {
                "price_id":          price_id,
                "stripe_session_id": stripe_session_id,
            })
            log.info("Added %d credits to user#%d", credits_amount, user.id)

    # Increment discount code usage
    discount_code_str = meta.get("discount_code", "")
    if discount_code_str:
        dc = (
            db.query(DiscountCode)
            .filter(DiscountCode.code == discount_code_str)
            .first()
        )
        if dc:
            dc.current_uses += 1
            db.commit()


def handle_subscription_created_or_updated(db: Session, event: dict) -> None:
    """customer.subscription.created / customer.subscription.updated."""
    from ele.billing import services as svc

    sub         = event["data"]["object"]
    customer_id = sub.get("customer")
    status      = sub.get("status", "")
    sub_id      = sub.get("id")

    # Resolve user — prefer metadata user_id, fall back to customer lookup
    user_id_str = sub.get("metadata", {}).get("user_id")
    user: User | None = None
    if user_id_str:
        user = svc.get_user_by_id(db, int(user_id_str))
    if user is None and customer_id:
        user = svc.get_user_by_stripe_customer_id(db, customer_id)
    if user is None:
        log.warning("No user found for subscription %s (customer=%s)", sub_id, customer_id)
        return

    user.stripe_subscription_id = sub_id
    user.stripe_customer_id     = customer_id
    user.subscription_status    = status

    if status in _ACTIVE_SUB_STATUSES:
        user.plan_type = "pro"
    elif status in {"canceled", "unpaid", "incomplete_expired"}:
        if user.plan_type != "ambassador":
            user.plan_type = "none"

    db.commit()
    log.info("User#%d subscription %s → status=%s plan=%s", user.id, sub_id, status, user.plan_type)


def handle_subscription_deleted(db: Session, event: dict) -> None:
    """customer.subscription.deleted — downgrade plan."""
    from ele.billing import services as svc

    sub    = event["data"]["object"]
    sub_id = sub.get("id")

    user = svc.get_user_by_stripe_subscription_id(db, sub_id)
    if user is None:
        return

    user.subscription_status    = "canceled"
    user.stripe_subscription_id = None
    if user.plan_type != "ambassador":
        user.plan_type = "none"

    db.commit()
    log.info("User#%d subscription deleted — plan downgraded", user.id)


def handle_invoice_payment_failed(db: Session, event: dict) -> None:
    """invoice.payment_failed — mark subscription as past_due."""
    from ele.billing import services as svc

    invoice     = event["data"]["object"]
    customer_id = invoice.get("customer")
    sub_id      = invoice.get("subscription")

    user: User | None = None
    if sub_id:
        user = svc.get_user_by_stripe_subscription_id(db, sub_id)
    if user is None and customer_id:
        user = svc.get_user_by_stripe_customer_id(db, customer_id)
    if user is None:
        return

    user.subscription_status = "past_due"
    db.commit()
    log.info("User#%d invoice payment failed — subscription past_due", user.id)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_pack_key(price_id: str) -> str:
    """Map a Stripe price ID to a local credit pack key."""
    cfg = get_config()
    return {
        cfg.stripe_price_creator_10:  "creator_10",
        cfg.stripe_price_creator_50:  "creator_50",
        cfg.stripe_price_creator_200: "creator_200",
    }.get(price_id, "")
