"""Shared pipeline utilities: image loading, linearisation, and metadata."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from ele.types import DegradationReport


def load_linear(path: str | Path) -> np.ndarray:
    """Load image from disk and return linear ProPhoto-space float32.

    Linearises sRGB gamma (assumed for JPEG/PNG unless ICC overrides).

    Args:
        path: Path to JPEG, PNG, or other PIL-supported format.

    Returns:
        H×W×3 float32, linear, values in [0, 1].
    """
    with Image.open(path) as pil_img:
        pil_img = pil_img.convert("RGB")
        arr = np.asarray(pil_img, dtype=np.float32) / 255.0

    # Remove sRGB gamma → linear
    return _srgb_to_linear(arr)


def _srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """Apply sRGB inverse EOTF (gamma ≈ 2.2 with linear toe)."""
    out = np.where(
        img <= 0.04045,
        img / 12.92,
        ((img + 0.055) / 1.055) ** 2.4,
    )
    return out.astype(np.float32)


def build_metadata(
    mode: str,
    input_path: str,
    report: DegradationReport,
    image: np.ndarray,
    extra: dict | None = None,
) -> dict:
    """Build the standard pipeline metadata dict.

    Includes mode, input path, all degradation scores, and export_size
    (height × width of the image as it will be written to TIFF).

    Args:
        mode:       Pipeline mode string ("free", "creator", "pro", "print").
        input_path: Original input file path.
        report:     Degradation report from Stage 1.
        image:      Post-pipeline float32 array (used only for shape).
        extra:      Optional extra keys (e.g. {"scale": "4"} for print mode).

    Returns:
        dict with str values, suitable for TIFF metadata and CLI display.
    """
    h, w = image.shape[:2]
    meta: dict = {
        "mode":        mode,
        "input":       str(input_path),
        "export_size": f"{w}×{h}",   # width × height, matching convention
        **{k: str(v) for k, v in vars(report).items() if k != "notes"},
    }
    if extra:
        meta.update(extra)
    return meta
