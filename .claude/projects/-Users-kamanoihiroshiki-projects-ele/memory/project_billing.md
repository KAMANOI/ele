---
name: billing_system
description: Complete monetization layer added to ele — plans, Stripe, access control, auth
type: project
---

Billing system implemented 2026-03-13.

**Plans:**
- creator: 1 credit (standard export), 3 credits (print export)
- pro: Stripe monthly subscription, unlimited
- ambassador: key-based, unlimited free
- none: can browse, cannot export

**Key files:**
- `ele/billing/` — all billing code (config, db, models, services, stripe_service, auth, admin, routes)
- `ele/billing/admin.py` — `ele-admin` CLI commands
- `ele/billing/routes.py` — auth routes + Stripe routes + admin page
- `ele/web/routes.py` — updated with `_access_denied_response` and credit consumption

**DB:** SQLite at `storage/ele.db` via SQLAlchemy 2.0 ORM, `create_all()` on startup

**Auth:** bcrypt (direct, not passlib), Starlette SessionMiddleware cookies

**Stripe:** checkout sessions, webhook handler for 5 event types, lazy coupon creation for discount codes

**CLI admin:**
- `ele-admin create-ambassador-key --label "name"`
- `ele-admin create-discount-code --code MAG-ELE20 --percent 20 --max-uses 500`
- `ele-admin list-ambassador-keys`
- `ele-admin list-discount-codes`

**Tests:** 66 total — 25 billing unit, 21 access control, 20 existing pipeline/web

**Why passlib was NOT used:** bcrypt>=4.0 has a 72-byte limit that causes passlib's internal `detect_wrap_bug` to raise ValueError. Using `bcrypt` directly instead.

**How to apply:** When modifying auth or access control, keep `can_export()` as the single gate — never duplicate the logic in routes.
