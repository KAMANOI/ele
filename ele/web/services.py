"""ele web services layer.

Handles storage, pipeline orchestration, preview generation,
export target mapping, and job state persistence.

Storage layout
--------------
uploads/   {job_id}{ext}                — raw upload bytes
outputs/   {job_id}_master.tiff                           — full-resolution export (free/creator/pro)
           {job_id}_print_x{n}_{style}_master.tiff      — SR print variant (quick or preview→export)
previews/  {job_id}_original.jpg        — browser preview of input (≤1200 px)
           {job_id}_preview.jpg         — browser preview of processed (≤1200 px)
tmp/       {job_id}.json                — job state

The master TIFF and preview JPEG are always separate files.
The preview is derived from a copy of the post-pipeline array; the TIFF is
written from the same array independently.  Neither can influence the other.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ele.config import SUPPORTED_PRINT_SCALES, SUPPORTED_PRINT_STYLES

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage layout
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent.parent
STORAGE_ROOT  = _PROJECT_ROOT / "storage"
UPLOADS_DIR   = STORAGE_ROOT / "uploads"
OUTPUTS_DIR   = STORAGE_ROOT / "outputs"
PREVIEWS_DIR  = STORAGE_ROOT / "previews"
TMP_DIR       = STORAGE_ROOT / "tmp"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

EXPORT_TARGETS: dict[str, dict[str, str]] = {
    "lightroom":  {"label": "Lightroom",    "sub": "16-bit TIFF · ProPhoto RGB"},
    "photoshop":  {"label": "Photoshop",    "sub": "16-bit TIFF · ProPhoto RGB"},
    "captureone": {"label": "Capture One",  "sub": "16-bit TIFF · ProPhoto RGB"},
    "adobe_dng":  {"label": "Adobe DNG",    "sub": "Linear DNG · planned"},
    "print":      {"label": "Print TIFF",   "sub": "Super resolution · Natural / AI Detail · ×2 / ×4 / ×6"},
}

_PREVIEW_MAX = 1200   # max long edge for browser JPEG previews — NEVER used for export

# Mode-specific minimum expected export long-edge (pixels).
# If a pipeline produces output smaller than this AND the input was bigger,
# it is a bug.  The sanity check in run_pipeline will raise.
_MIN_EXPORT_LONG_EDGE: dict[str, int] = {
    "free":    1,     # input may genuinely be tiny
    "creator": 1,     # same
    "pro":     1,
    "print":   1,
}
# Threshold below which an output is "suspiciously preview-sized".
# If the export long-edge is <= this AND the original was larger, raise.
_PREVIEW_SIZE_THRESHOLD = 1600


def ensure_storage() -> None:
    """Create all storage directories if they don't exist."""
    for d in (UPLOADS_DIR, OUTPUTS_DIR, PREVIEWS_DIR, TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

def new_job_id() -> str:
    return uuid.uuid4().hex[:14]


def _state_path(job_id: str) -> Path:
    return TMP_DIR / f"{job_id}.json"


def init_job_state(
    job_id: str,
    filename: str,
    mode: str,
    flow: str,
) -> dict[str, Any]:
    return {
        "job_id":            job_id,
        "original_filename": filename,
        "mode":              mode,
        "flow":              flow,
        "upload_path":       None,
        "output_path":       None,
        "preview_original":  None,
        "preview_processed": None,
        "report":            None,
        "metadata":          None,
        "export_target":     None,
        "print_scale":       None,
        "status":            "uploaded",
        "error":             None,
        "created_at":        datetime.now(timezone.utc).isoformat(),
    }


def load_job_state(job_id: str) -> dict[str, Any] | None:
    p = _state_path(job_id)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def save_job_state(job_id: str, state: dict[str, Any]) -> None:
    _state_path(job_id).write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def validate_extension(filename: str) -> str | None:
    """Return lowercased extension if valid, else None."""
    ext = Path(filename).suffix.lower()
    return ext if ext in ALLOWED_EXTENSIONS else None


def save_upload(data: bytes, filename: str, job_id: str) -> Path:
    ext = Path(filename).suffix.lower()
    dest = UPLOADS_DIR / f"{job_id}{ext}"
    dest.write_bytes(data)
    return dest


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def _pil_dims(path: Path) -> tuple[int, int]:
    """Return (width, height) of an image without decoding pixel data."""
    with Image.open(path) as img:
        return img.size  # (width, height)


def image_dims(path: Path | str) -> tuple[int, int] | None:
    """Return (width, height) of any image file, or None on failure.

    Safe to call on TIFF, JPEG, or PNG files.  Used by routes to measure
    the actual on-disk dimensions of the master TIFF and preview JPEG so
    that the result page can display verified (not assumed) resolutions.
    """
    try:
        return _pil_dims(Path(path))
    except Exception:
        return None


def run_pipeline(
    upload_path: Path,
    mode: str,
    job_id: str,
    print_scale: int | None = None,
    print_style: str = "natural",
) -> tuple[str, dict, dict, str]:
    """Run the pipeline for *mode*.

    Returns
    -------
    (output_path, report_dict, metadata_dict, input_size_str)

    *output_path*     — absolute path to the written master TIFF.
    *input_size_str*  — "WxH" string of the original uploaded image.

    The TIFF export has already been written to *output_path* before this
    function returns.  Preview generation must be done from *output_path*
    (reading the saved TIFF), NOT from the pipeline's in-memory array.
    This guarantees the preview and export are always derived from the same
    on-disk master and cannot influence each other.
    """
    # Print exports use a dedicated filename that encodes scale + style so the
    # artifact is never confused with a standard (non-upscaled) master TIFF.
    # All other modes share the generic {job_id}_master.tiff name.
    if mode == "print":
        _scale     = print_scale or 2
        _style_tag = "aidetail" if print_style == "ai-detail" else "natural"
        output_path = OUTPUTS_DIR / f"{job_id}_print_x{_scale}_{_style_tag}_master.tiff"
    else:
        output_path = OUTPUTS_DIR / f"{job_id}_master.tiff"

    # Record original input dimensions BEFORE any processing
    try:
        in_w, in_h = _pil_dims(upload_path)
        input_size_str = f"{in_w}x{in_h}"
        in_long_edge = max(in_w, in_h)
    except Exception:
        input_size_str = "unknown"
        in_long_edge = 0

    log.info(
        "[%s] pipeline START  mode=%s  scale=%s  style=%s  input=%s  upload=%s  output=%s",
        job_id, mode,
        print_scale if mode == "print" else "-",
        print_style if mode == "print" else "-",
        input_size_str, upload_path.name, output_path.name,
    )

    _cb = lambda step, label: None  # no-op progress for web context

    common = dict(
        input_path=str(upload_path),
        output_path=str(output_path),
        _progress_cb=_cb,
    )

    if mode == "free":
        from ele.pipeline.free_pipeline import run_free_pipeline
        result = run_free_pipeline(**common)
    elif mode == "creator":
        from ele.pipeline.creator_pipeline import run_creator_pipeline
        result = run_creator_pipeline(**common)
    elif mode == "pro":
        from ele.pipeline.pro_pipeline import run_pro_pipeline
        result = run_pro_pipeline(**common)
    elif mode == "print":
        from ele.pipeline.print_pipeline import run_print_pipeline
        result = run_print_pipeline(
            scale=_scale,
            print_style=print_style,
            **common,
        )
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    # --- Verify export dimensions ---
    exp_h, exp_w = result.image.shape[:2]
    exp_long_edge = max(exp_w, exp_h)
    export_size_str = f"{exp_w}x{exp_h}"

    log.info(
        "[%s] pipeline END  mode=%s  export_master=%s  dims=%s  file=%s  size_kb=%d",
        job_id, mode, output_path.name, export_size_str,
        output_path, output_path.stat().st_size // 1024 if output_path.exists() else -1,
    )
    # Explicit per-mode export dimension log — used to diagnose quick-export
    # resolution bugs where preview-sized output was returned for PRO mode.
    log.info(
        "[%s] quick_export mode=%s export_w=%d export_h=%d master=%s",
        job_id, mode, exp_w, exp_h, output_path.name,
    )

    # Hard sanity check: if the export is preview-sized but the input was not,
    # something has gone wrong in the pipeline (preview array leaked into export).
    if (
        in_long_edge > _PREVIEW_SIZE_THRESHOLD
        and exp_long_edge <= _PREVIEW_SIZE_THRESHOLD
    ):
        raise RuntimeError(
            f"Export resolution sanity check FAILED for job {job_id}: "
            f"input was {input_size_str} (long edge {in_long_edge}px) but "
            f"export master is only {export_size_str} (long edge {exp_long_edge}px). "
            f"This is preview-sized output from a large input — "
            f"the preview-resize path has contaminated the export path. "
            f"mode={mode!r}"
        )

    report_dict: dict[str, Any] = {
        "compression_score":   round(result.report.compression_score, 4),
        "clipping_score":      round(result.report.clipping_score, 4),
        "sharpness_score":     round(result.report.sharpness_score, 4),
        "noise_score":         round(result.report.noise_score, 4),
        "dynamic_range_score": round(result.report.dynamic_range_score, 4),
        "notes":               result.report.notes,
    }
    metadata_dict = {str(k): str(v) for k, v in result.metadata.items()}
    # Return 4-tuple: do NOT expose result.image to callers.
    # Preview generation must read from the saved master TIFF (see create_previews).
    return str(output_path), report_dict, metadata_dict, input_size_str


# ---------------------------------------------------------------------------
# Preview generation
# ---------------------------------------------------------------------------

def create_previews(
    job_id: str,
    original_path: Path,
    master_tiff_path: Path,
) -> tuple[str, str]:
    """Create browser-ready JPEG previews from the saved master TIFF.

    The processed preview is generated by reading *master_tiff_path* from disk
    and decoding it through the inverse color pipeline.  This guarantees
    complete separation between export generation and preview generation:
    the in-memory pipeline array is never reused here, so no pipeline stage
    can accidentally produce a preview-sized export.

    Preview pipeline (processed)
    ----------------------------
    master_tiff (uint16 ROMM-gamma ProPhoto) → inverse ROMM TRC → linear ProPhoto
    → ProPhoto→sRGB matrix → sRGB OETF → resize to ≤1200 px → JPEG

    Export pipeline (separate, already written before this is called)
    ----------------------------------------------------------------
    pipeline image → sRGB→ProPhoto matrix → ROMM gamma → uint16 → TIFF (on disk)

    Args:
        job_id:           Job identifier used to name output files.
        original_path:    Path to the original upload (for the "before" preview).
        master_tiff_path: Path to the full-resolution master TIFF written by the
                          pipeline.  The preview is derived from this file, NOT
                          from any in-memory array.

    Returns:
        (original_url, processed_url) as URL paths served by FastAPI.
    """
    import tifffile
    from ele.export.color_management import decode_prophoto_tiff_for_preview

    # --- Original ---
    orig_dest = PREVIEWS_DIR / f"{job_id}_original.jpg"
    with Image.open(original_path) as img:
        img = img.convert("RGB")
        img.thumbnail((_PREVIEW_MAX, _PREVIEW_MAX), Image.LANCZOS)
        img.save(orig_dest, format="JPEG", quality=82, optimize=True)

    # --- Processed: load master TIFF from disk → decode → resize → JPEG ---
    # Reading from the saved file (not from any in-memory array) is the only
    # way to guarantee the preview reflects exactly what was exported.
    proc_dest = PREVIEWS_DIR / f"{job_id}_preview.jpg"
    try:
        raw_u16 = tifffile.imread(str(master_tiff_path))          # uint16, ROMM-gamma ProPhoto
        float_img = raw_u16.astype(np.float32) / 65535.0
        display = decode_prophoto_tiff_for_preview(float_img)      # → sRGB [0,1]
        u8 = (display * 255.0 + 0.5).astype(np.uint8)
        pil_proc = Image.fromarray(u8, mode="RGB")
        pil_proc.thumbnail((_PREVIEW_MAX, _PREVIEW_MAX), Image.LANCZOS)
        pil_proc.save(proc_dest, format="JPEG", quality=82, optimize=True)
        log.info(
            "[%s] preview generated from master TIFF  tiff_w=%d tiff_h=%d  preview_w=%d preview_h=%d",
            job_id, raw_u16.shape[1], raw_u16.shape[0],
            pil_proc.width, pil_proc.height,
        )
    except Exception:
        import shutil
        shutil.copy2(orig_dest, proc_dest)

    return f"/previews/{job_id}_original.jpg", f"/previews/{job_id}_preview.jpg"


# ---------------------------------------------------------------------------
# Export target mapping
# ---------------------------------------------------------------------------

def apply_export_target(
    job_id: str,
    target: str,
    state: dict[str, Any],
    print_scale: int | None = None,
    print_style: str = "natural",
) -> tuple[str | None, str | None]:
    """Apply target-specific export logic.

    Returns (output_path, error_message).  Exactly one will be non-None.
    """
    existing = state.get("output_path")
    upload_path = state.get("upload_path")
    mode = state.get("mode", "creator")

    if target in ("lightroom", "photoshop", "captureone"):
        # Existing 16-bit TIFF is already correct for all three
        return existing, None

    if target == "adobe_dng":
        return None, (
            "Adobe DNG export is planned but not available in this build. "
            "Use the Lightroom / Photoshop / Capture One TIFF targets instead — "
            "the exported TIFF already has ProPhoto RGB ICC embedded."
        )

    if target == "print":
        scale = print_scale or 2
        if scale not in SUPPORTED_PRINT_SCALES:
            return None, (
                f"Invalid print scale {scale}. "
                f"Supported: {list(SUPPORTED_PRINT_SCALES)}"
            )
        if print_style not in SUPPORTED_PRINT_STYLES:
            return None, (
                f"Invalid print style {print_style!r}. "
                f"Supported: {list(SUPPORTED_PRINT_STYLES)}"
            )
        style_tag  = "aidetail" if print_style == "ai-detail" else "natural"
        output_path = str(
            OUTPUTS_DIR / f"{job_id}_print_x{scale}_{style_tag}_master.tiff"
        )

        # Measure input dimensions for the post-run assertion
        input_dims = image_dims(upload_path)
        in_w, in_h = input_dims if input_dims else (0, 0)

        log.info(
            "[%s] print-render  target=print  style=%s  scale=%d  "
            "input=%dx%d  output_path=%s",
            job_id, print_style, scale, in_w, in_h, output_path,
        )

        try:
            from ele.pipeline.print_pipeline import run_print_pipeline
            result = run_print_pipeline(
                input_path=upload_path,
                output_path=output_path,
                scale=scale,
                print_style=print_style,
            )

            # Hard assertion: output must be larger than input when scale > 1
            if scale > 1 and in_w > 0 and in_h > 0:
                out_dims = image_dims(result.output_path)
                out_w, out_h = out_dims if out_dims else (0, 0)
                log.info(
                    "[%s] print-render done  input=%dx%d  output=%dx%d  path=%s  exists=%s",
                    job_id, in_w, in_h, out_w, out_h,
                    result.output_path, Path(result.output_path).exists(),
                )
                if out_w <= in_w or out_h <= in_h:
                    raise RuntimeError(
                        f"Print upscale assertion failed: "
                        f"input={in_w}x{in_h}  output={out_w}x{out_h}  scale={scale}. "
                        f"Output must be strictly larger than input."
                    )

            # Regenerate processed preview from the saved print master TIFF.
            try:
                _, proc_url = create_previews(
                    job_id, Path(upload_path), Path(result.output_path)
                )
                state["preview_processed"] = proc_url
            except Exception:
                pass
            return result.output_path, None
        except Exception as exc:
            return None, f"Print pipeline error: {exc}"

    return None, f"Unknown export target: {target!r}"


def download_filename(state: dict[str, Any]) -> str:
    """Generate a sensible download filename."""
    stem   = Path(state.get("original_filename", "output")).stem
    target = state.get("export_target")
    suffix = ""
    if target == "print":
        scale  = state.get("print_scale") or 2
        pstyle = state.get("print_style", "natural")
        style_tag = "aidetail" if pstyle == "ai-detail" else "natural"
        suffix = f"_print_x{scale}_{style_tag}"
    return f"{stem}_ele{suffix}.tiff"


# ---------------------------------------------------------------------------
# Histogram computation
# ---------------------------------------------------------------------------

def compute_histogram_data(path: str | Path | None) -> dict | None:
    """Compute a 256-bin luminance histogram for any supported image file.

    Handles JPEG/PNG (uint8 sRGB) and 16-bit TIFF (uint16 ROMM-gamma ProPhoto).
    Luminance is computed from the file's native encoding — intentionally
    not colour-managed, because we want to compare the tonal *distribution*
    as stored, not to do a colour-correct comparison.

    Returns a dict with:
        svg_points        — SVG polygon points string for a 600×100 chart area (legacy)
        svg_path          — smooth SVG path string using cubic bezier (preferred)
        dr_stops          — estimated usable dynamic range in stops (float | None)
        shadow_pct        — percentage of pixels in the bottom 10% of range
        highlight_pct     — percentage of pixels in the top 10% of range
        shadow_clip_pct   — percentage of pixels that are truly clipped (<3%)
        highlight_clip_pct— percentage of pixels that are truly clipped (>97%)

    Returns None if the file cannot be read.

    Lightweight by design: reads a downsampled thumbnail (≤512 px long edge)
    instead of the full image so that even 100 MP TIFFs are instant.
    """
    if not path:
        return None
    try:
        path = Path(path)
        if not path.exists():
            return None

        lum_flat = _read_luminance_flat(path)
        if lum_flat is None or len(lum_flat) == 0:
            return None

        counts, _ = np.histogram(lum_flat, bins=256, range=(0.0, 1.0))
        counts     = counts.astype(np.float32)

        max_c = counts.max()
        counts_norm = (counts / max_c).tolist() if max_c > 0 else [0.0] * 256

        dr_stops           = _estimate_dr_stops(lum_flat)
        shadow_pct         = round(float((lum_flat < 0.10).mean() * 100), 1)
        high_pct           = round(float((lum_flat > 0.90).mean() * 100), 1)
        shadow_clip_pct    = round(float((lum_flat < 0.03).mean() * 100), 2)
        highlight_clip_pct = round(float((lum_flat > 0.97).mean() * 100), 2)

        svg_points = _histogram_svg_points(counts_norm, width=600, height=100)
        svg_path   = _histogram_svg_path(counts_norm, width=600, height=100)

        return {
            "svg_points":         svg_points,
            "svg_path":           svg_path,
            "dr_stops":           dr_stops,
            "shadow_pct":         shadow_pct,
            "highlight_pct":      high_pct,
            "shadow_clip_pct":    shadow_clip_pct,
            "highlight_clip_pct": highlight_clip_pct,
        }

    except Exception:
        return None


def _read_luminance_flat(path: Path) -> np.ndarray | None:
    """Return a flat float32 luminance array from an image, using a thumbnail."""
    THUMB = 512   # max long edge for histogram sampling — keeps it fast

    suffix = path.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        try:
            import tifffile
            arr = tifffile.imread(str(path))
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            if arr.dtype == np.uint16:
                arr_f = arr.astype(np.float32) / 65535.0
            elif arr.dtype == np.uint8:
                arr_f = arr.astype(np.float32) / 255.0
            elif arr.dtype == np.uint32:
                arr_f = arr.astype(np.float32) / 4294967295.0
            else:
                arr_f = arr.astype(np.float32)
            # Downsample for speed: take every N-th pixel
            h, w = arr_f.shape[:2]
            step = max(1, max(h, w) // THUMB)
            arr_f = arr_f[::step, ::step, :3]
            lum = (0.2126 * arr_f[..., 0]
                   + 0.7152 * arr_f[..., 1]
                   + 0.0722 * arr_f[..., 2])
            return lum.ravel().astype(np.float32)
        except Exception:
            pass   # fall through to PIL path

    # PIL path (JPEG, PNG, and TIFF fallback)
    with Image.open(path) as img:
        img.thumbnail((THUMB, THUMB), Image.LANCZOS)
        img = img.convert("RGB")
        arr_f = np.asarray(img, dtype=np.float32) / 255.0
    lum = (0.2126 * arr_f[..., 0]
           + 0.7152 * arr_f[..., 1]
           + 0.0722 * arr_f[..., 2])
    return lum.ravel().astype(np.float32)


def _estimate_dr_stops(lum: np.ndarray) -> float | None:
    """Estimate usable dynamic range in stops from a flat luminance array."""
    import math
    active = lum[lum > 1e-4]
    if len(active) < 100:
        return None
    p_low  = float(np.percentile(active, 1.0))
    p_high = float(np.percentile(active, 99.0))
    if p_low < 1e-5 or p_high <= p_low:
        return None
    return round(math.log2(p_high / p_low), 1)


def _histogram_svg_points(counts_norm: list[float], width: int, height: int) -> str:
    """Build an SVG polygon points string from 256 normalized bin counts.

    The polygon closes along the bottom of the chart so it renders as a
    filled area.  Y=0 is the top, Y=height is the bottom (SVG convention).
    """
    n  = len(counts_norm)
    h  = height
    # 5 % headroom at the top so a full-height bar doesn't touch the border
    scale = h * 0.95

    pts = [f"0,{h}"]
    for i, v in enumerate(counts_norm):
        x = round(i / (n - 1) * width, 1)
        y = round(h - v * scale, 1)
        pts.append(f"{x},{y}")
    pts.append(f"{width},{h}")
    return " ".join(pts)


def _histogram_svg_path(counts_norm: list[float], width: int, height: int) -> str:
    """Build a smooth SVG path using Catmull-Rom cubic bezier curves.

    Subsamples to every 5th bin (≈51 points) then interpolates smoothly.
    The path closes along the bottom to render as a filled area.
    Y=0 is top, Y=height is bottom (SVG convention).
    """
    n     = len(counts_norm)
    h     = height
    scale = h * 0.95

    # Subsample: every 5th bin to reduce point count while keeping smooth shape
    step  = 5
    idxs  = list(range(0, n, step))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)

    def pt(i: int):
        x = i / (n - 1) * width
        y = h - counts_norm[i] * scale
        return (round(x, 1), round(y, 1))

    pts = [pt(i) for i in idxs]
    m   = len(pts)

    if m < 2:
        return ""

    # Catmull-Rom → cubic bezier control points
    d = [f"M 0,{h} L {pts[0][0]},{pts[0][1]}"]
    for i in range(1, m):
        p0 = pts[max(0, i - 2)]
        p1 = pts[i - 1]
        p2 = pts[i]
        p3 = pts[min(m - 1, i + 1)]

        cp1x = round(p1[0] + (p2[0] - p0[0]) / 6, 1)
        cp1y = round(p1[1] + (p2[1] - p0[1]) / 6, 1)
        cp2x = round(p2[0] - (p3[0] - p1[0]) / 6, 1)
        cp2y = round(p2[1] - (p3[1] - p1[1]) / 6, 1)

        d.append(f"C {cp1x},{cp1y} {cp2x},{cp2y} {p2[0]},{p2[1]}")

    d.append(f"L {width},{h} Z")
    return " ".join(d)


# ---------------------------------------------------------------------------
# Crop comparison previews
# ---------------------------------------------------------------------------

# Resolution at which source images are loaded for crop selection/extraction.
# Higher = better mask accuracy and richer crop pixels; lower = less memory.
_CROP_LOAD_MAX    = 1200   # max long edge for loading source images
# Display parameters for saved crop PNG files.
_CROP_DISPLAY_MAX = 400    # max long edge of the saved crop PNG
_CROP_NATIVE_MIN  = 80     # min short side of the native crop — smaller = suppressed
_CROP_MAX_UPSCALE = 2.0    # upscale beyond this factor is suppressed


def generate_crop_previews(
    job_id: str,
    orig_path: str | Path | None,
    master_tiff_path: str | Path | None,
) -> dict:
    """Generate heuristic skin and sky crop previews from the source files.

    Crops are extracted from:
      - orig_path       : full-resolution original upload (JPEG/PNG)
      - master_tiff_path: full-resolution processed master TIFF (ProPhoto, 16-bit)

    Both sources are loaded at _CROP_LOAD_MAX before crop extraction so that
    crop pixels are taken from the highest-quality available source, not from
    a pre-downscaled or pre-JPEG-compressed preview file.

    Crop previews are saved as lossless PNG (no extra compression artifacts).
    A crop is suppressed if the native window is too small or would require
    excessive upscaling to fill the display panel.

    Returns a dict with keys:
        skin_orig_url, skin_master_url  — /previews/... URLs or None
        sky_orig_url,  sky_master_url   — /previews/... URLs or None
        has_skin, has_sky               — bool
    """
    result = {
        "skin_orig_url": None, "skin_master_url": None,
        "sky_orig_url":  None, "sky_master_url":  None,
        "has_skin": False,     "has_sky": False,
    }
    if not orig_path or not master_tiff_path:
        return result
    try:
        orig_arr = _read_thumb_rgb(orig_path, max_size=_CROP_LOAD_MAX)
        mast_arr = _load_master_for_crops(master_tiff_path, max_size=_CROP_LOAD_MAX)
        if orig_arr is None or mast_arr is None:
            return result

        oh, ow = orig_arr.shape[:2]
        mh, mw = mast_arr.shape[:2]
        # If master has different dimensions (e.g. print SR), normalise to orig space.
        if (oh, ow) != (mh, mw):
            pil_m = Image.fromarray((mast_arr * 255).astype(np.uint8))
            pil_m = pil_m.resize((ow, oh), Image.LANCZOS)
            mast_arr = np.asarray(pil_m, dtype=np.float32) / 255.0

        crop_size = max(80, min(600, min(oh, ow) // 3))

        # ── Ranked candidate selection ──────────────────────────────────────
        skin_candidates = _select_skin_candidates(orig_arr, crop_size)
        sky_candidates  = _select_sky_candidates(orig_arr, crop_size)

        # Pick best skin box, then best sky box that is spatially distinct.
        skin_box, skin_conf = _pick_distinct(skin_candidates, None)
        sky_box,  sky_conf  = _pick_distinct(sky_candidates,  skin_box)

        iou = _crop_iou(skin_box, sky_box) if skin_box and sky_box else 0.0
        log.debug(
            "crop skin: box=%s conf=%.3f  sky: box=%s conf=%.3f  IoU=%.3f",
            skin_box, skin_conf, sky_box, sky_conf, iou,
        )

        # ── Skin crop ──────────────────────────────────────────────────────
        if skin_box:
            orig_url = _save_crop(
                orig_arr, skin_box,
                PREVIEWS_DIR / f"{job_id}_crop_skin_orig.png",
                label=f"[{job_id}] skin/orig",
            )
            mast_url = _save_crop(
                mast_arr, skin_box,
                PREVIEWS_DIR / f"{job_id}_crop_skin_master.png",
                label=f"[{job_id}] skin/master",
            )
            if orig_url and mast_url:
                result["has_skin"] = True
                result["skin_orig_url"]   = orig_url
                result["skin_master_url"] = mast_url

        # ── Sky crop ───────────────────────────────────────────────────────
        if sky_box:
            orig_url = _save_crop(
                orig_arr, sky_box,
                PREVIEWS_DIR / f"{job_id}_crop_sky_orig.png",
                label=f"[{job_id}] sky/orig",
            )
            mast_url = _save_crop(
                mast_arr, sky_box,
                PREVIEWS_DIR / f"{job_id}_crop_sky_master.png",
                label=f"[{job_id}] sky/master",
            )
            if orig_url and mast_url:
                result["has_sky"] = True
                result["sky_orig_url"]   = orig_url
                result["sky_master_url"] = mast_url

    except Exception:
        pass
    return result


def _read_thumb_rgb(path: str | Path | None, max_size: int = 512) -> np.ndarray | None:
    """Read any supported image as float32 RGB [0,1] at thumbnail resolution."""
    if not path:
        return None
    try:
        with Image.open(Path(path)) as img:
            img.thumbnail((max_size, max_size), Image.LANCZOS)
            img = img.convert("RGB")
            return np.asarray(img, dtype=np.float32) / 255.0
    except Exception:
        return None


def _load_master_for_crops(
    path: str | Path | None, max_size: int = _CROP_LOAD_MAX
) -> np.ndarray | None:
    """Load the processed master TIFF, decode ProPhoto → sRGB, thumbnail to max_size.

    Falls back to _read_thumb_rgb (which handles JPEG previews) if the TIFF
    load/decode fails — so the function works in tests that don't produce a real TIFF.
    """
    if not path:
        return None
    p = Path(path)
    if p.suffix.lower() in {".tif", ".tiff"} and p.exists():
        try:
            import tifffile
            from ele.export.color_management import decode_prophoto_tiff_for_preview

            raw_u16  = tifffile.imread(str(p))                         # uint16 ProPhoto
            float_img = raw_u16.astype(np.float32) / 65535.0
            srgb      = decode_prophoto_tiff_for_preview(float_img)    # float32 sRGB
            u8        = (srgb * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
            pil       = Image.fromarray(u8, mode="RGB")
            pil.thumbnail((max_size, max_size), Image.LANCZOS)
            return np.asarray(pil, dtype=np.float32) / 255.0
        except Exception:
            pass  # fall through to thumbnail fallback
    return _read_thumb_rgb(path, max_size=max_size)


# ---------------------------------------------------------------------------
# Crop candidate selection — thresholds
# ---------------------------------------------------------------------------

_SKIN_MIN_COVERAGE = 0.12   # min fraction of skin-mask pixels inside window
_SKY_MIN_COVERAGE  = 0.18   # min fraction of sky-mask pixels inside window
_CONF_THRESHOLD    = 0.05   # raw score must exceed this to be shown
_IOU_MAX           = 0.30   # max allowed IoU between skin and sky crops


def _compute_skin_mask(arr_f: np.ndarray) -> np.ndarray:
    """Boolean mask: heuristic skin-tone pixels."""
    r, g, b = arr_f[..., 0], arr_f[..., 1], arr_f[..., 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return (
        (r > g) & (g > b)
        & ((r - b) > 0.05) & ((r - g) < 0.35)
        & (lum > 0.12) & (lum < 0.88)
    )


def _compute_sky_mask(arr_f: np.ndarray) -> np.ndarray:
    """Boolean mask: heuristic sky pixels — blue-dominant, upper portion only."""
    h = arr_f.shape[0]
    r, g, b = arr_f[..., 0], arr_f[..., 1], arr_f[..., 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    # Require blue dominance, reasonable brightness, and upper 65% of image
    sky_colour = (b > r) & (b > g) & (lum > 0.28)
    spatial = np.zeros(arr_f.shape[:2], dtype=bool)
    spatial[: int(h * 0.65), :] = True
    return sky_colour & spatial


def _select_skin_candidates(
    arr_f: np.ndarray, crop_size: int
) -> list[tuple[float, tuple]]:
    """Return sorted (score, box) candidates for skin crop, best first.

    Scoring per window:
      coverage   — fraction of skin-mask pixels (must exceed _SKIN_MIN_COVERAGE)
      lum_score  — portrait luminance range, peaks at ~0.45
      texture    — moderate local contrast; flat or blown-out windows score 0
      center     — horizontal center bias (portraits usually mid-frame)
    """
    skin_mask = _compute_skin_mask(arr_f)
    h, w = arr_f.shape[:2]
    stride = max(1, crop_size // 3)
    scored: list[tuple[float, tuple]] = []

    for wy in range(0, h - crop_size + 1, stride):
        for wx in range(0, w - crop_size + 1, stride):
            wmask = skin_mask[wy : wy + crop_size, wx : wx + crop_size]
            coverage = wmask.mean()
            if coverage < _SKIN_MIN_COVERAGE:
                continue

            win = arr_f[wy : wy + crop_size, wx : wx + crop_size]
            lum = 0.2126 * win[..., 0] + 0.7152 * win[..., 1] + 0.0722 * win[..., 2]
            mean_lum = float(lum.mean())
            # Portrait usable range
            if mean_lum < 0.15 or mean_lum > 0.82:
                continue
            lum_score = max(0.0, 1.0 - abs(mean_lum - 0.45) / 0.35)

            # Texture: prefer moderate contrast (std ~0.04–0.12)
            std = float(lum.std())
            if std < 0.01:
                continue
            texture_score = min(1.0, std / 0.04) * max(
                0.0, 1.0 - max(0.0, std - 0.12) / 0.15
            )

            # Horizontal center bias
            cx_rel = (wx + crop_size / 2) / w
            center_score = max(0.4, 1.0 - abs(cx_rel - 0.5) * 1.2)

            score = coverage * lum_score * texture_score * center_score
            if score > 0:
                scored.append((score, (wx, wy, wx + crop_size, wy + crop_size)))

    scored.sort(reverse=True)
    return scored


def _select_sky_candidates(
    arr_f: np.ndarray, crop_size: int
) -> list[tuple[float, tuple]]:
    """Return sorted (score, box) candidates for sky crop, best first.

    Scoring per window:
      coverage    — fraction of sky-mask pixels (must exceed _SKY_MIN_COVERAGE)
      vert_score  — upper image areas score higher; windows below 65% are skipped
      smoothness  — sky should be smooth; high lum-std penalised
      skin_penalty— windows with substantial skin coverage are down-scored
    """
    sky_mask  = _compute_sky_mask(arr_f)
    skin_mask = _compute_skin_mask(arr_f)
    h, w = arr_f.shape[:2]
    stride = max(1, crop_size // 3)
    scored: list[tuple[float, tuple]] = []

    for wy in range(0, h - crop_size + 1, stride):
        # Reject windows whose centre is below 65% of the frame height
        vert_center = (wy + crop_size / 2) / h
        if vert_center > 0.65:
            continue
        vert_score = max(0.1, 1.0 - vert_center / 0.50)

        for wx in range(0, w - crop_size + 1, stride):
            wmask = sky_mask[wy : wy + crop_size, wx : wx + crop_size]
            coverage = wmask.mean()
            if coverage < _SKY_MIN_COVERAGE:
                continue

            win = arr_f[wy : wy + crop_size, wx : wx + crop_size]
            lum = 0.2126 * win[..., 0] + 0.7152 * win[..., 1] + 0.0722 * win[..., 2]
            std = float(lum.std())
            smoothness = max(0.0, 1.0 - std / 0.20)

            # Penalise windows dominated by skin (sky shouldn't contain faces)
            wskin = skin_mask[wy : wy + crop_size, wx : wx + crop_size]
            skin_fraction = wskin.mean()
            skin_penalty = max(0.0, 1.0 - skin_fraction * 4.0)

            score = coverage * vert_score * smoothness * skin_penalty
            if score > 0:
                scored.append((score, (wx, wy, wx + crop_size, wy + crop_size)))

    scored.sort(reverse=True)
    return scored


def _crop_iou(box_a: tuple, box_b: tuple) -> float:
    """Intersection-over-Union of two (x0,y0,x1,y1) rectangles."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    if inter == 0:
        return 0.0
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _pick_distinct(
    candidates: list[tuple[float, tuple]],
    exclude_box: tuple | None,
) -> tuple[tuple | None, float]:
    """Return the best (box, score) from *candidates* that:
    - has score >= _CONF_THRESHOLD
    - does not overlap *exclude_box* by more than _IOU_MAX

    Returns (None, 0.0) when no suitable candidate exists.
    """
    for score, box in candidates:
        if score < _CONF_THRESHOLD:
            break
        if exclude_box is None or _crop_iou(box, exclude_box) <= _IOU_MAX:
            return box, score
    return None, 0.0


def _save_crop(
    arr_f: np.ndarray,
    box: tuple,
    dest: Path,
    label: str = "",
) -> str | None:
    """Extract box from arr_f, resize to display size, save as lossless PNG.

    Guardrails:
      - Returns None (suppresses the crop) if the native short side is below
        _CROP_NATIVE_MIN — the crop is too small to be informative.
      - Returns None if reaching _CROP_DISPLAY_MAX would require upscaling by
        more than _CROP_MAX_UPSCALE — avoids blowing up tiny regions into
        crunchy messes.

    For downscaling, LANCZOS is used.
    For upscaling within the allowed factor, BICUBIC is used.
    """
    try:
        x0, y0, x1, y1 = box
        crop = arr_f[y0:y1, x0:x1]
        native_h, native_w = crop.shape[:2]
        native_short = min(native_h, native_w)
        native_long  = max(native_h, native_w)

        # Guardrail 1: native crop too small → suppress
        if native_short < _CROP_NATIVE_MIN:
            log.debug(
                "%s  SUPPRESSED: native short side %dpx < min %dpx",
                label, native_short, _CROP_NATIVE_MIN,
            )
            return None

        # Compute the scale needed to fit inside _CROP_DISPLAY_MAX
        scale = _CROP_DISPLAY_MAX / native_long if native_long > _CROP_DISPLAY_MAX else (
            _CROP_DISPLAY_MAX / native_long   # upscale path
        )

        # Guardrail 2: required upscale exceeds budget → suppress
        if scale > _CROP_MAX_UPSCALE:
            log.debug(
                "%s  SUPPRESSED: upscale %.2fx > max %.1fx  (native %dx%d)",
                label, scale, _CROP_MAX_UPSCALE, native_w, native_h,
            )
            return None

        out_w = max(1, round(native_w * scale))
        out_h = max(1, round(native_h * scale))

        u8  = (crop * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(u8, mode="RGB")

        if (out_w, out_h) != (native_w, native_h):
            resample = Image.LANCZOS if scale < 1.0 else Image.BICUBIC
            pil = pil.resize((out_w, out_h), resample=resample)

        pil.save(dest, format="PNG", optimize=True)
        log.debug(
            "%s  native=%dx%d  display=%dx%d  scale=%.2fx  format=PNG",
            label, native_w, native_h, out_w, out_h, scale,
        )
        return f"/previews/{dest.name}"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Editability metrics
# ---------------------------------------------------------------------------

def compute_editability_metrics(
    orig_preview_path: str | Path | None,
    master_preview_path: str | Path | None,
    master_hist: dict | None = None,
) -> dict:
    """Compute heuristic editability metrics from preview JPEG images.

    All values are heuristic estimates — clearly not scientifically precise.
    Scores are in the range [0, 10] (higher = more headroom / stability).

    Returns a dict with:
        dr_stops            — estimated dynamic range in stops (float | None)
        highlight_headroom  — score 0-10: how compressed are master highlights
        shadow_headroom     — score 0-10: how much shadow detail is available
        skin_stability      — score 0-10: smoothness of skin highlight zone (None if no skin)
        sky_stability       — score 0-10: sky gradient & blue-channel safety (None if no sky)
        has_skin, has_sky   — bool
    """
    out: dict = {
        "dr_stops":           None,
        "highlight_headroom": None,
        "shadow_headroom":    None,
        "skin_stability":     None,
        "sky_stability":      None,
        "has_skin":           False,
        "has_sky":            False,
    }

    # Pull DR from already-computed histogram
    if master_hist:
        out["dr_stops"] = master_hist.get("dr_stops")

    master_arr = _read_thumb_rgb(master_preview_path)
    if master_arr is None:
        return out

    m_lum = (0.2126 * master_arr[..., 0]
             + 0.7152 * master_arr[..., 1]
             + 0.0722 * master_arr[..., 2])

    # Highlight headroom: (1 − 90th-percentile) × 12.5
    # → p90 = 0.80 → score 2.5; p90 = 0.60 → score 5.0; p90 = 0.40 → score 7.5
    p90 = float(np.percentile(m_lum, 90))
    out["highlight_headroom"] = round(min(10.0, max(0.0, (1.0 - p90) * 12.5)), 1)

    # Shadow headroom: 5th percentile of non-black pixels × 55
    # → p5 = 0.05 → score 2.8; p5 = 0.10 → score 5.5; p5 = 0.16 → score 8.8
    active = m_lum[m_lum > 0.005]
    if len(active) > 50:
        p5 = float(np.percentile(active, 5))
        out["shadow_headroom"] = round(min(10.0, max(0.0, p5 * 55.0)), 1)

    orig_arr = _read_thumb_rgb(orig_preview_path)
    if orig_arr is None:
        return out

    # Match sizes (needed for mask application to master)
    oh, ow = orig_arr.shape[:2]
    mh, mw = master_arr.shape[:2]
    if (oh, ow) != (mh, mw):
        pil_m = Image.fromarray((master_arr * 255).astype(np.uint8))
        pil_m = pil_m.resize((ow, oh), Image.LANCZOS)
        master_matched = np.asarray(pil_m, dtype=np.float32) / 255.0
        m_lum_m = (0.2126 * master_matched[..., 0]
                   + 0.7152 * master_matched[..., 1]
                   + 0.0722 * master_matched[..., 2])
    else:
        master_matched = master_arr
        m_lum_m        = m_lum

    # Skin stability — smoothness of mid-high lum skin in the master
    skin_mask = _compute_skin_mask(orig_arr)
    if skin_mask.sum() >= 200:
        out["has_skin"] = True
        skin_zone   = skin_mask & (m_lum_m > 0.30) & (m_lum_m < 0.85)
        skin_pixels = m_lum_m[skin_zone]
        if len(skin_pixels) >= 50:
            std_val = float(skin_pixels.std())
            # Target: std < 0.06 = very smooth; std > 0.16 = patchy
            score = max(0.0, min(10.0, (1.0 - std_val / 0.14) * 10.0))
            out["skin_stability"] = round(score, 1)

    # Sky stability — blue-channel headroom + gradient uniformity
    sky_mask = _compute_sky_mask(orig_arr)
    if sky_mask.sum() >= 200:
        out["has_sky"] = True
        sky_zone    = sky_mask
        sky_b       = master_matched[..., 2][sky_zone]
        sky_lum_pix = m_lum_m[sky_zone]
        if len(sky_b) >= 50:
            # Blue headroom: how far below 1.0 is the 95th-percentile blue?
            b_head  = max(0.0, 1.0 - float(np.percentile(sky_b, 95)))
            # Gradient smoothness: low std = smooth gradients, good banding resistance
            lum_std = float(sky_lum_pix.std())
            smooth  = max(0.0, 1.0 - lum_std / 0.25)
            score   = b_head * 6.0 + smooth * 4.0
            out["sky_stability"] = round(min(10.0, max(0.0, score)), 1)

    return out


# ---------------------------------------------------------------------------
# Report formatting for templates
# ---------------------------------------------------------------------------

_REPORT_LABELS = {
    "compression_score":   ("Compression artefacts", "Low = clean source"),
    "clipping_score":      ("Clipping",               "Highlights / shadows at limit"),
    "sharpness_score":     ("Sharpness",              "Higher = sharper input"),
    "noise_score":         ("Noise",                  "Higher = more noise detected"),
    "dynamic_range_score": ("Dynamic Range",          "Higher = wider recoverable range"),
}


def format_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert report dict into display-ready rows for templates."""
    rows = []
    for key, (label, hint) in _REPORT_LABELS.items():
        val = float(report.get(key, 0.0))
        rows.append({
            "label": label,
            "hint":  hint,
            "value": f"{val:.3f}",
            "pct":   int(val * 100),
        })
    return rows
