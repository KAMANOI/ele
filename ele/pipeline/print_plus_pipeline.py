"""Print+ pipeline using Replicate Clarity Upscaler.

Tiers
-----
quality : single pass ×2  (3 credits)
large   : single pass ×4  (4 credits)
ultra   : two passes ×2→×4 = ×8 total  (6 credits)
"""
from __future__ import annotations

import logging
import os
import tempfile

from PIL import Image

from ele.core.clarity_upscale import run_clarity_upscale
from ele.export.tiff_export import export_tiff
from ele.utils import pil_to_float32_linear_rgb

log = logging.getLogger(__name__)

# tier → ordered list of scale_factor values for each Replicate call
_TIER_PASSES: dict[str, list[int]] = {
    "quality": [2],
    "large":   [4],
    "ultra":   [2, 4],
}

SUPPORTED_TIERS = tuple(_TIER_PASSES.keys())


def run_print_plus_pipeline(
    input_path: str,
    output_path: str,
    tier: str = "quality",
) -> str:
    """Run Print+ upscale via Replicate and export as 16-bit TIFF.

    Returns output_path.
    Raises ValueError with error code E210 (API) or E211 (TIFF conversion).
    """
    passes = _TIER_PASSES.get(tier, _TIER_PASSES["quality"])
    log.info("print_plus_pipeline: tier=%s  passes=%s  input=%s", tier, passes, input_path)

    current = input_path
    tmp_files: list[str] = []

    try:
        for scale in passes:
            # NamedTemporaryFile avoids TOCTOU: file is created atomically
            tmp_f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_f.close()
            tmp_files.append(tmp_f.name)
            run_clarity_upscale(current, tmp_f.name, scale_factor=scale)
            current = tmp_f.name

        try:
            with Image.open(current) as img:
                arr = pil_to_float32_linear_rgb(img)
            export_tiff(arr, output_path, metadata={"mode": "print_plus", "tier": tier})
        except Exception as exc:
            raise ValueError(f"E211: TIFF conversion failed: {exc}") from exc

    finally:
        for f in tmp_files:
            try:
                os.unlink(f)
            except FileNotFoundError:
                pass

    log.info("print_plus_pipeline: done  output=%s", output_path)
    return output_path
