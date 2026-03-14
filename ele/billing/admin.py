"""Admin CLI commands for ambassador keys and discount codes.

Usage (after `pip install -e .`):
    ele-admin create-ambassador-key --label "photographer_name"
    ele-admin create-discount-code --code MAG-ELE20 --percent 20 --max-uses 500
    ele-admin list-ambassador-keys
    ele-admin list-discount-codes
"""

from __future__ import annotations

import secrets

import typer

admin_app = typer.Typer(
    name="ele-admin",
    help="Admin tools for ele billing — keys and discount codes.",
    add_completion=False,
)


def _get_db():
    from ele.billing.db import SessionLocal, init_db
    init_db()
    return SessionLocal()


@admin_app.command("create-ambassador-key")
def create_ambassador_key(
    label: str = typer.Option("", "--label", "-l", help="Ambassador name or label"),
) -> None:
    """Create a new ambassador key and print it."""
    db = _get_db()
    try:
        from ele.billing.models import AmbassadorKey

        key_value = f"AMB-{secrets.token_urlsafe(16).upper()}"
        key = AmbassadorKey(
            key_value=key_value,
            label=label.strip() or None,
            is_active=True,
        )
        db.add(key)
        db.commit()
        typer.echo(f"Ambassador key: {key_value}")
        if label:
            typer.echo(f"Label:          {label}")
        typer.echo("Status:         active (unredeemed)")
    finally:
        db.close()


@admin_app.command("list-ambassador-keys")
def list_ambassador_keys() -> None:
    """List all ambassador keys."""
    db = _get_db()
    try:
        from ele.billing.models import AmbassadorKey

        keys = db.query(AmbassadorKey).order_by(AmbassadorKey.created_at.desc()).all()
        if not keys:
            typer.echo("No ambassador keys found.")
            return
        for k in keys:
            status   = "active" if k.is_active else "inactive"
            redeemed = (
                f"redeemed by user#{k.redeemed_by_user_id}"
                if k.redeemed_by_user_id else "unredeemed"
            )
            label = k.label or "(no label)"
            typer.echo(f"{k.key_value}  [{status}]  {label}  —  {redeemed}")
    finally:
        db.close()


@admin_app.command("create-discount-code")
def create_discount_code(
    code:      str = typer.Option(...,  "--code",     "-c", help="Discount code (e.g. MAG-ELE20)"),
    percent:   int = typer.Option(...,  "--percent",  "-p", help="Percent off (e.g. 20)"),
    max_uses:  int = typer.Option(0,    "--max-uses", "-m", help="Max redemptions (0 = unlimited)"),
) -> None:
    """Create a new percent-off discount code."""
    db = _get_db()
    try:
        from ele.billing.models import DiscountCode

        normalised = code.upper().strip()
        dc = DiscountCode(
            code=normalised,
            kind="percent",
            percent_off=percent,
            is_active=True,
            max_uses=max_uses if max_uses > 0 else None,
            current_uses=0,
        )
        db.add(dc)
        db.commit()
        typer.echo(f"Discount code:  {normalised}")
        typer.echo(f"Percent off:    {percent}%")
        typer.echo(f"Max uses:       {'unlimited' if max_uses == 0 else max_uses}")
        typer.echo("Status:         active")
    finally:
        db.close()


@admin_app.command("list-discount-codes")
def list_discount_codes() -> None:
    """List all discount codes."""
    db = _get_db()
    try:
        from ele.billing.models import DiscountCode

        codes = db.query(DiscountCode).order_by(DiscountCode.created_at.desc()).all()
        if not codes:
            typer.echo("No discount codes found.")
            return
        for dc in codes:
            status = "active" if dc.is_active else "inactive"
            uses   = (
                f"{dc.current_uses}/{dc.max_uses}"
                if dc.max_uses else
                f"{dc.current_uses}/∞"
            )
            typer.echo(f"{dc.code}  {dc.percent_off}% off  [{status}]  uses: {uses}")
    finally:
        db.close()
