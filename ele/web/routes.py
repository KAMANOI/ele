"""ele web routes — image upload, pipeline, preview, result, download."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ele.billing.auth import get_current_user_from_session
from ele.billing.db import get_db
from ele.billing import services as billing_svc
from ele.web import services

log = logging.getLogger(__name__)

router = APIRouter()

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))


def _render(
    request: Request,
    template: str,
    ctx: dict,
    status_code: int = 200,
) -> HTMLResponse:
    return templates.TemplateResponse(request, template, ctx, status_code=status_code)


def _base_ctx(request: Request, db: Session) -> dict:
    """Return a base template context dict that includes the current user."""
    return {"user": get_current_user_from_session(request, db)}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/terms", response_class=HTMLResponse)
def terms_page(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    return _render(request, "terms.html", _base_ctx(request, db))


@router.get("/privacy", response_class=HTMLResponse)
def privacy_page(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    return _render(request, "privacy.html", _base_ctx(request, db))


@router.get("/tokushoho", response_class=HTMLResponse)
def tokushoho_page(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    return _render(request, "tokushoho.html", _base_ctx(request, db))


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "app": "ele", "version": "0.1.0"}


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse)
def index(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    return _render(request, "index.html", _base_ctx(request, db))


# ---------------------------------------------------------------------------
# Upload → process → redirect
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=None)
async def upload(
    request:     Request,
    file:        Annotated[UploadFile, File()],
    mode:        Annotated[str, Form()] = "creator",
    flow:        Annotated[str, Form()] = "quick",
    print_scale: Annotated[Optional[int], Form()] = None,
    print_style: Annotated[str, Form()] = "natural",
    db:          Session = Depends(get_db),
) -> RedirectResponse | HTMLResponse:

    services.ensure_storage()
    user = get_current_user_from_session(request, db)

    # ── Validate extension ────────────────────────────────────────────────
    fname = file.filename or "upload"
    ext   = services.validate_extension(fname)
    if ext is None:
        ctx = _base_ctx(request, db)
        ctx["error"] = "Unsupported file type. Please upload a JPEG, PNG, or TIFF."
        return _render(request, "index.html", ctx, status_code=400)

    if mode not in ("free", "creator", "pro", "print"):
        ctx = _base_ctx(request, db)
        ctx["error"] = f"Unknown mode: {mode!r}"
        return _render(request, "index.html", ctx, status_code=400)

    if mode == "print" and print_style not in ("natural", "ai-detail"):
        ctx = _base_ctx(request, db)
        ctx["error"] = f"Unknown print style: {print_style!r}."
        return _render(request, "index.html", ctx, status_code=400)

    # ── Access control ────────────────────────────────────────────────────
    export_kind = "print" if mode == "print" else "standard"
    can, reason = billing_svc.can_export(user, export_kind)
    if not can:
        return _access_denied_response(request, db, user, reason)

    # ── Save upload ───────────────────────────────────────────────────────
    job_id      = services.new_job_id()
    data        = await file.read()
    upload_path = services.save_upload(data, fname, job_id)

    state = services.init_job_state(job_id, fname, mode, flow)
    state["upload_path"] = str(upload_path)
    state["status"]      = "processing"
    services.save_job_state(job_id, state)

    # ── Run pipeline ──────────────────────────────────────────────────────
    try:
        output_path, report, metadata, input_size = services.run_pipeline(
            upload_path, mode, job_id,
            print_scale=print_scale if mode == "print" else None,
            print_style=print_style if mode == "print" else "natural",
        )
    except Exception as exc:
        state["status"] = "error"
        state["error"]  = str(exc)
        services.save_job_state(job_id, state)
        ctx = _base_ctx(request, db)
        ctx["error"] = f"Processing failed: {exc}"
        return _render(request, "index.html", ctx, status_code=500)

    # ── Consume credits after successful export ───────────────────────────
    if user:
        billing_svc.consume_export_credit(db, user, export_kind, job_id)

    state["output_path"]   = output_path
    state["report"]        = report
    state["metadata"]      = metadata
    state["status"]        = "ready"
    state["print_scale"]   = print_scale
    state["print_style"]   = print_style if mode == "print" else None
    state["input_size"]    = input_size
    state["export_size"]   = metadata.get("export_size", "")
    # Tag print quick-exports the same way the preview→export flow does so
    # download_filename() and the result page can identify the print artifact.
    if mode == "print":
        state["export_target"] = "print"

    log.info(
        "[%s] upload done  mode=%s  flow=%s  output_path=%s  export_target=%s",
        job_id, mode, flow, output_path, state.get("export_target"),
    )

    # Hard assertion — quick export must not produce preview-sized output
    if flow == "quick" and mode in {"creator", "pro", "print"}:
        try:
            in_parts = input_size.split("x")
            in_long  = max(int(in_parts[0]), int(in_parts[1])) if len(in_parts) == 2 else 0
            dl_dims  = services.image_dims(output_path)
            if dl_dims:
                dl_long = max(dl_dims)
                log.info(
                    "[%s] quick_export assertion  mode=%s  input_long=%d  master_long=%d  "
                    "output_path=%s  exists=%s",
                    job_id, mode, in_long, dl_long,
                    output_path, Path(output_path).exists(),
                )
                if in_long > 2000 and dl_long <= 2000:
                    raise RuntimeError(
                        f"Download artifact incorrectly points to preview-sized output. "
                        f"mode={mode!r}  job_id={job_id!r}  "
                        f"input_long_edge={in_long}px  download_long_edge={dl_long}px"
                    )
                # For print mode: output must be strictly larger than input (upscale verification)
                if mode == "print" and in_long > 0 and dl_long <= in_long:
                    raise RuntimeError(
                        f"Print upscale assertion failed: "
                        f"output long edge ({dl_long}px) must exceed input ({in_long}px). "
                        f"job_id={job_id!r}  output_path={output_path!r}"
                    )
        except RuntimeError:
            raise
        except Exception:
            pass

    # Generate previews from saved master TIFF
    try:
        orig_url, proc_url = services.create_previews(
            job_id, upload_path, Path(output_path)
        )
        state["preview_original"]  = orig_url
        state["preview_processed"] = proc_url
    except Exception:
        pass

    services.save_job_state(job_id, state)

    if flow == "preview":
        return RedirectResponse(f"/preview/{job_id}", status_code=303)
    return RedirectResponse(f"/result/{job_id}", status_code=303)


# ---------------------------------------------------------------------------
# Preview page
# ---------------------------------------------------------------------------

@router.get("/preview/{job_id}", response_class=HTMLResponse)
def preview_page(
    request: Request,
    job_id:  str,
    db:      Session = Depends(get_db),
) -> HTMLResponse:
    state = services.load_job_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")

    ctx = _base_ctx(request, db)
    ctx.update({
        "job_id":      job_id,
        "state":       state,
        "report_rows": services.format_report(state.get("report") or {}),
        "targets":     services.EXPORT_TARGETS,
        "scales":      [2, 4, 6],
    })
    return _render(request, "preview.html", ctx)


# ---------------------------------------------------------------------------
# Export (from preview page)
# ---------------------------------------------------------------------------

@router.post("/export/{job_id}", response_model=None)
async def export_target(
    request:     Request,
    job_id:      str,
    target:      Annotated[str, Form()],
    print_scale: Annotated[Optional[int], Form()] = None,
    print_style: Annotated[str, Form()] = "natural",
    db:          Session = Depends(get_db),
) -> RedirectResponse | HTMLResponse:

    state = services.load_job_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")

    user = get_current_user_from_session(request, db)

    # ── Access control ────────────────────────────────────────────────────
    export_kind = "print" if target == "print" else "standard"
    can, reason = billing_svc.can_export(user, export_kind)
    if not can:
        return _access_denied_response(request, db, user, reason)

    if target == "print" and print_style not in ("natural", "ai-detail"):
        print_style = "natural"

    output_path, error = services.apply_export_target(
        job_id, target, state,
        print_scale=print_scale,
        print_style=print_style,
    )

    if error:
        ctx = _base_ctx(request, db)
        ctx.update({
            "job_id":       job_id,
            "state":        state,
            "report_rows":  services.format_report(state.get("report") or {}),
            "targets":      services.EXPORT_TARGETS,
            "scales":       [2, 4, 6],
            "export_error": error,
        })
        return _render(request, "preview.html", ctx)

    # ── Consume credits after successful export ───────────────────────────
    if user:
        billing_svc.consume_export_credit(db, user, export_kind, job_id)

    state["export_target"] = target
    state["output_path"]   = output_path
    state["print_scale"]   = print_scale
    state["print_style"]   = print_style if target == "print" else state.get("print_style")
    state["status"]        = "done"
    services.save_job_state(job_id, state)

    return RedirectResponse(f"/result/{job_id}", status_code=303)


# ---------------------------------------------------------------------------
# Result page
# ---------------------------------------------------------------------------

@router.get("/result/{job_id}", response_class=HTMLResponse)
def result_page(
    request: Request,
    job_id:  str,
    db:      Session = Depends(get_db),
) -> HTMLResponse:
    state = services.load_job_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")

    user = get_current_user_from_session(request, db)

    if state.get("status") == "error":
        ctx = _base_ctx(request, db)
        ctx["error"] = state.get("error", "Unknown error")
        return _render(request, "index.html", ctx, status_code=500)

    target_key   = state.get("export_target", "")
    target_label = services.EXPORT_TARGETS.get(target_key, {}).get("label") if target_key else None

    # Measure actual on-disk dimensions
    download_path = state.get("output_path", "")
    download_dims = services.image_dims(download_path) if download_path else None
    download_size = f"{download_dims[0]}x{download_dims[1]}" if download_dims else ""

    preview_url  = state.get("preview_processed", "")
    preview_path = ""
    preview_size = ""
    if preview_url:
        preview_path = str(services.PREVIEWS_DIR / Path(preview_url).name)
        p_dims = services.image_dims(preview_path)
        if p_dims:
            preview_size = f"{p_dims[0]}x{p_dims[1]}"

    log.info(
        "[%s] result_page  mode=%s  flow=%s  export_target=%s  "
        "master=%s  master_exists=%s  master_size=%s  preview=%s  preview_size=%s",
        job_id, state.get("mode"), state.get("flow"), state.get("export_target"),
        download_path, Path(download_path).exists() if download_path else False,
        download_size, preview_path, preview_size,
    )

    orig_src    = state.get("upload_path", "")
    orig_hist   = services.compute_histogram_data(orig_src)
    master_hist = services.compute_histogram_data(download_path)

    crops        = services.generate_crop_previews(job_id, orig_src, download_path)
    edit_metrics = services.compute_editability_metrics(orig_src, preview_path, master_hist)

    ctx = _base_ctx(request, db)
    ctx.update({
        "job_id":        job_id,
        "state":         state,
        "report_rows":   services.format_report(state.get("report") or {}),
        "dl_filename":   services.download_filename(state),
        "target_label":  target_label,
        "download_path": download_path,
        "download_size": download_size,
        "preview_path":  preview_path,
        "preview_size":  preview_size,
        "orig_hist":     orig_hist,
        "master_hist":   master_hist,
        "crops":         crops,
        "edit_metrics":  edit_metrics,
    })
    return _render(request, "result.html", ctx)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

@router.get("/download/{job_id}")
def download(
    request: Request,
    job_id:  str,
    db:      Session = Depends(get_db),
) -> FileResponse:
    state = services.load_job_state(job_id)
    if not state:
        raise HTTPException(status_code=404, detail="Job not found")

    # Download requires an active session or valid completed job
    # (we don't re-check billing here — the credit was consumed at export time)
    out = state.get("output_path")
    if not out or not Path(out).exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    out_path   = Path(out)
    size_kb    = out_path.stat().st_size // 1024
    dl_dims    = services.image_dims(out)
    dl_w, dl_h = dl_dims if dl_dims else (0, 0)
    log.info(
        "[%s] download  target=%s  mode=%s  flow=%s  "
        "path=%s  exists=%s  size_kb=%d  w=%d  h=%d",
        job_id, state.get("export_target"), state.get("mode"), state.get("flow"),
        out, out_path.exists(), size_kb, dl_w, dl_h,
    )

    return FileResponse(
        path=out,
        media_type="image/tiff",
        filename=services.download_filename(state),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _access_denied_response(
    request: Request,
    db:      Session,
    user,
    reason:  str | None,
) -> RedirectResponse | HTMLResponse:
    """Return a redirect or rendered page for blocked exports."""
    if reason == "login_required":
        return RedirectResponse("/login?next=/", status_code=303)

    if reason == "insufficient_credits":
        ctx = _base_ctx(request, db)
        ctx.update({
            "error":       "Not enough credits to export. Buy more on the pricing page.",
            "access_warn": "insufficient_credits",
        })
        return _render(request, "index.html", ctx, status_code=402)

    # no_plan
    return RedirectResponse("/pricing?warn=no_plan", status_code=303)
