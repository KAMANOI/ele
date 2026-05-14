"""ele web routes — image upload, pipeline, preview, result, download."""

from __future__ import annotations

import io
import logging
import re
import zipfile
from pathlib import Path
from typing import Annotated, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ele.billing.auth import get_current_user_from_session
from ele.billing.db import get_db
from ele.billing import services as billing_svc
from ele.web import services
from ele.web.scanner import scan_file

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

@router.get("/guide", response_class=HTMLResponse)
def guide_page(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    return _render(request, "guide.html", _base_ctx(request, db))


@router.get("/terms", response_class=HTMLResponse)
def terms_page(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    return _render(request, "terms.html", _base_ctx(request, db))


@router.get("/privacy", response_class=HTMLResponse)
def privacy_page(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    return _render(request, "privacy.html", _base_ctx(request, db))


@router.get("/tokushoho", response_class=HTMLResponse)
def tokushoho_page(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    return _render(request, "tokushoho.html", _base_ctx(request, db))


@router.get("/contact", response_class=HTMLResponse)
def contact_page(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    return _render(request, "contact.html", _base_ctx(request, db))


@router.get("/health")
def health() -> dict:
    return {"status": "ok", "app": "ele", "version": "0.1.0"}


# ---------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse)
def index(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    return _render(request, "index.html", _base_ctx(request, db))


@router.get("/ja", response_class=HTMLResponse)
def index_ja(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    return _render(request, "index_ja.html", _base_ctx(request, db))


# ---------------------------------------------------------------------------
# Upload → process → redirect
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=None)
async def upload(
    request:          Request,
    file:             Annotated[UploadFile, File()],
    mode:             Annotated[str, Form()] = "creator",
    flow:             Annotated[str, Form()] = "quick",
    print_scale:      Annotated[Optional[int], Form()] = None,
    print_style:      Annotated[str, Form()] = "natural",
    print_plus_tier:  Annotated[str, Form()] = "quality",
    lite_mode:        Annotated[bool, Form()] = False,
    db:               Session = Depends(get_db),
) -> RedirectResponse | HTMLResponse:

    services.ensure_storage()
    services.cleanup_old_uploads()
    user = get_current_user_from_session(request, db)

    # ── Validate extension ────────────────────────────────────────────────
    fname = file.filename or "upload"
    ext   = services.validate_extension(fname)
    if ext is None:
        ctx = _base_ctx(request, db)
        ctx["error"] = "Unsupported file type. Please upload a JPEG, PNG, or TIFF."
        return _render(request, "index.html", ctx, status_code=400)

    if mode not in ("free", "creator", "pro", "print", "print_plus"):
        ctx = _base_ctx(request, db)
        ctx["error"] = f"Unknown mode: {mode!r}"
        return _render(request, "index.html", ctx, status_code=400)

    if mode == "print" and print_style not in ("natural", "ai-detail"):
        ctx = _base_ctx(request, db)
        ctx["error"] = f"Unknown print style: {print_style!r}."
        return _render(request, "index.html", ctx, status_code=400)

    if mode == "print_plus" and print_plus_tier not in ("quality", "large", "ultra"):
        ctx = _base_ctx(request, db)
        ctx["error"] = f"Unknown Print+ tier: {print_plus_tier!r}."
        return _render(request, "index.html", ctx, status_code=400)

    # ── Access control ────────────────────────────────────────────────────
    if mode == "print":
        export_kind = "print"
    elif mode == "print_plus":
        export_kind = f"print_plus_{print_plus_tier}"
    else:
        export_kind = "standard"
    can, reason = billing_svc.can_export(user, export_kind)
    if not can:
        return _access_denied_response(request, db, user, reason)

    # ── Save upload ───────────────────────────────────────────────────────
    job_id      = services.new_job_id()
    data        = await file.read()
    upload_path = services.save_upload(data, fname, job_id)

    # ── Malware scan (ClamAV) ─────────────────────────────────────────────
    is_clean, scan_reason = scan_file(upload_path)
    if not is_clean:
        log.warning("[%s] Upload rejected by AV scan: %s", job_id, scan_reason)
        ctx = _base_ctx(request, db)
        ctx["error"] = "アップロードされたファイルでマルウェアが検出されました。別のファイルをお試しください。"
        return _render(request, "index.html", ctx, status_code=400)

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
            print_plus_tier=print_plus_tier if mode == "print_plus" else "quality",
            lite_mode=lite_mode,
        )
    except Exception as exc:
        log.error("[%s] Pipeline error: %s", job_id, exc, exc_info=True)
        err_str = str(exc)
        # Extract error code (E2xx etc.) from exception message if present
        import re as _re
        code_match = _re.search(r"\b(E\d{3})\b", err_str)
        error_code = code_match.group(1) if code_match else "E999"
        state["status"] = "error"
        state["error"]  = f"Processing failed. Please try again. ({error_code})"
        services.save_job_state(job_id, state)
        ctx = _base_ctx(request, db)
        ctx["error"] = f"処理に失敗しました。時間をおいて再度お試しください。\n{error_code}"
        return _render(request, "index.html", ctx, status_code=500)

    # ── Consume credits after successful export ───────────────────────────
    if user:
        billing_svc.consume_export_credit(db, user, export_kind, job_id)

    state["output_path"]      = output_path
    state["report"]           = report
    state["metadata"]         = metadata
    state["status"]           = "ready"
    state["print_scale"]      = print_scale
    state["print_style"]      = print_style if mode == "print" else None
    state["print_plus_tier"]  = print_plus_tier if mode == "print_plus" else None
    state["lite_mode"]        = lite_mode
    state["input_size"]       = input_size
    state["export_size"]      = metadata.get("export_size", "")
    # Tag print quick-exports so download_filename() and result page can identify them.
    if mode == "print":
        state["export_target"] = "print"
    elif mode == "print_plus":
        state["export_target"] = "print_plus"

    log.info(
        "[%s] upload done  mode=%s  flow=%s  output_path=%s  export_target=%s",
        job_id, mode, flow, output_path, state.get("export_target"),
    )

    # Hard assertion — quick export must not produce preview-sized output
    if flow == "quick" and mode in {"creator", "pro", "print", "print_plus"}:
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
                # For print/print_plus: output must be strictly larger than input
                if mode in {"print", "print_plus"} and in_long > 0 and dl_long <= in_long:
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
    if target == "print":
        export_kind = "print"
    elif target == "print_plus":
        pp_tier = state.get("print_plus_tier", "quality")
        export_kind = f"print_plus_{pp_tier}"
    else:
        export_kind = "standard"
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

# ---------------------------------------------------------------------------
# Batch upload
# ---------------------------------------------------------------------------

_BATCH_MAX_FILES    = 10
_BATCH_MAX_FILE_MB  = 50
_BATCH_ALLOWED_MODES = {"creator", "pro"}   # free/print excluded from batch


def _sanitize_zip_entry(filename: str) -> str:
    """Return a safe ZIP entry name: ASCII-safe stem + _ele.tiff."""
    stem = Path(filename).stem
    safe = re.sub(r"[^A-Za-z0-9_\-]", "_", stem)
    return (safe or "output") + "_ele.tiff"


@router.get("/batch", response_class=HTMLResponse)
def batch_page(request: Request, db: Session = Depends(get_db)) -> HTMLResponse:
    return _render(request, "batch.html", _base_ctx(request, db))


@router.post("/upload_batch", response_model=None)
async def upload_batch(
    request:          Request,
    files:            Annotated[List[UploadFile], File()],
    mode:             Annotated[str, Form()] = "creator",
    lite_mode:        Annotated[bool, Form()] = False,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db:               Session = Depends(get_db),
) -> StreamingResponse | JSONResponse:

    services.ensure_storage()
    services.cleanup_old_uploads()
    user = get_current_user_from_session(request, db)

    # ── Validate mode ─────────────────────────────────────────────────────
    if mode not in _BATCH_ALLOWED_MODES:
        return JSONResponse({"error": f"Batch processing is available for Creator and Pro modes only.", "code": "E120"}, status_code=400)

    # ── Validate file list ────────────────────────────────────────────────
    valid_files: list[UploadFile] = []
    for f in files:
        if services.validate_extension(f.filename or "") is not None:
            valid_files.append(f)

    if not valid_files:
        return JSONResponse({"error": "No supported files found. Please upload JPEG, PNG, or TIFF.", "code": "E121"}, status_code=400)

    if len(valid_files) > _BATCH_MAX_FILES:
        return JSONResponse({"error": f"Maximum {_BATCH_MAX_FILES} files per batch.", "code": "E122"}, status_code=400)

    n_files = len(valid_files)

    # ── Access control ────────────────────────────────────────────────────
    can, reason = billing_svc.can_export_batch(user, "standard", n_files)
    if not can:
        if reason == "login_required":
            return JSONResponse({"error": "Login required to export.", "code": "E501"}, status_code=401)
        if reason == "insufficient_credits":
            cost = n_files * billing_svc.EXPORT_COSTS.get("standard", 1)
            return JSONResponse(
                {"error": f"Insufficient credits. {n_files} file(s) require {cost} credit(s).", "code": "E502"},
                status_code=402,
            )
        return JSONResponse({"error": "No active plan. Please visit Pricing to get started.", "code": "E503"}, status_code=403)

    # ── Process each file ─────────────────────────────────────────────────
    batch_id          = services.new_job_id()
    successes:        list[tuple[str, str]] = []   # (tiff_path, safe_zip_entry_name)
    failures:         list[str]             = []   # original filenames (any reason)
    malware_detected: bool                  = False

    for uf in valid_files:
        fname  = uf.filename or "upload"
        job_id = services.new_job_id()
        try:
            data = await uf.read()

            # Per-file size limit
            if len(data) > _BATCH_MAX_FILE_MB * 1024 * 1024:
                log.warning("[batch %s][%s] file exceeds %dMB limit, skipped", batch_id, fname, _BATCH_MAX_FILE_MB)
                failures.append(fname)
                continue

            upload_path = services.save_upload(data, fname, job_id)

            try:
                is_clean, _ = scan_file(upload_path)
            except Exception as scan_exc:
                log.error("[batch %s][%s] AV scan error: %s — treating as unsafe", batch_id, fname, scan_exc)
                is_clean = False

            if not is_clean:
                malware_detected = True
                failures.append(fname)
                upload_path.unlink(missing_ok=True)
                continue

            output_path, _, _, _ = services.run_pipeline(
                upload_path, mode, job_id, lite_mode=lite_mode,
            )
            successes.append((output_path, _sanitize_zip_entry(fname)))

        except Exception as exc:
            import re as _re
            code_match = _re.search(r"\b(E\d{3})\b", str(exc))
            err_code   = code_match.group(1) if code_match else "E999"
            log.error("[batch %s][%s] pipeline error %s: %s", batch_id, fname, err_code, exc, exc_info=True)
            failures.append(fname)

    if not successes:
        return JSONResponse(
            {"error": "All files failed to process. Please try again.", "code": "E230"},
            status_code=500,
        )

    # ── Consume credits in one transaction (successful files only) ────────
    if user and user.plan_type == "creator":
        n_ok_credits = len(successes)
        billing_svc.consume_export_credit_bulk(db, user, "standard", n_ok_credits, batch_id)

    # ── Build ZIP in memory ───────────────────────────────────────────────
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for tiff_path, entry_name in successes:
            zf.write(tiff_path, arcname=entry_name)
    buf.seek(0)

    tiff_paths = [p for p, _ in successes]
    def _cleanup_tiffs(paths: list[str]) -> None:
        for p in paths:
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass
    background_tasks.add_task(_cleanup_tiffs, tiff_paths)

    n_ok   = len(successes)
    n_fail = len(failures)
    # Safe ASCII filename only
    filename = f"ele_batch_{n_ok}files.zip"
    headers  = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "X-Batch-Success":     str(n_ok),
        "X-Batch-Failed":      str(n_fail),
        "X-Batch-Malware":     "1" if malware_detected else "0",
    }

    return StreamingResponse(buf, media_type="application/zip", headers=headers)


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
            "error":       "クレジットが不足しています。料金ページでクレジットを購入してください。",
            "access_warn": "insufficient_credits",
        })
        return _render(request, "index.html", ctx, status_code=402)

    # no_plan
    return RedirectResponse("/pricing?warn=no_plan", status_code=303)
