"""Stage 6a — 16-bit TIFF export.

Writes a linear float32 image to a 16-bit RGB TIFF with:
  - Deflate (zlib) + horizontal predictor compression
    (LZW requires the optional 'imagecodecs' package; deflate uses stdlib zlib)
  - ProPhoto RGB ICC profile always embedded (TIFF tag 34675)
    Source priority: bundled → system ColorSync/colord → generated fallback
  - Colour pipeline metadata in TIFF ImageDescription

Requires: tifffile
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import tifffile

from ele.export.color_management import (
    build_pipeline_metadata,
    load_prophoto_icc,
    prepare_for_export,
    to_export_prophoto_tiff,
)
from ele.utils import safe_mkdir_for_file

log = logging.getLogger(__name__)


def export_tiff(
    image: np.ndarray,
    output_path: str,
    metadata: dict | None = None,
) -> str:
    """Export a float32 linear RGB image to a 16-bit TIFF file.

    Args:
        image:       H×W×3 float32, linear light, values in [0, 1].
        output_path: Destination path.  Extension .tiff/.tif enforced.
        metadata:    Additional metadata to merge into TIFF tags.

    Returns:
        Resolved output path string.
    """
    # Resolve path
    path = Path(output_path)
    if path.suffix.lower() not in {".tif", ".tiff"}:
        path = path.with_suffix(".tiff")
    safe_mkdir_for_file(path)

    h, w = image.shape[:2]
    log.info(
        "export_tiff: mode=<from caller>  width=%d  height=%d  path=%s",
        w, h, path,
    )

    # Prepare array:
    #   1. Validate / clip to [0, 1]
    #   2. Convert linear sRGB → linear ProPhoto, then apply ROMM gamma
    #      so stored values match what the embedded ProPhoto ICC declares.
    img_f32 = prepare_for_export(image)
    img_enc = to_export_prophoto_tiff(img_f32)   # ROMM-gamma encoded ProPhoto
    img_u16 = (img_enc * 65535.0 + 0.5).astype(np.uint16)

    # Metadata
    pipeline_meta = build_pipeline_metadata()
    if metadata:
        pipeline_meta.update({str(k): v for k, v in metadata.items()})

    # TIFF ImageDescription must be 7-bit ASCII; encode non-ASCII safely
    raw_desc = "; ".join(f"{k}={v}" for k, v in pipeline_meta.items())
    description = raw_desc.encode("ascii", errors="replace").decode("ascii")

    # ICC profile — always embed; warn only when using the generated fallback
    icc_bytes, icc_source = load_prophoto_icc()
    if "generated" in icc_source:
        log.warning(
            "No system ProPhoto RGB ICC profile found; "
            "embedding a generated minimal fallback. "
            "The output is correctly tagged but assign ProPhoto RGB manually "
            "if your editor does not honour the embedded profile."
        )
    else:
        log.debug("Embedding ICC profile from %s", icc_source)

    tifffile.imwrite(
        path,
        img_u16,
        photometric="rgb",
        compression="deflate",    # zlib/deflate — stdlib, no imagecodecs needed
        predictor=True,           # horizontal differencing predictor
        description=description,
        extratags=[(34675, "B", len(icc_bytes), icc_bytes, True)],
    )

    return str(path)


def export_tiff_srgb_debug(
    image: np.ndarray,
    output_path: str,
) -> str:
    """Export a float32 linear image as a gamma-encoded sRGB TIFF for diagnosis.

    This is a **debug-only** export path.  It applies the sRGB OETF and saves
    a 16-bit TIFF without an embedded ICC profile (Photoshop defaults to sRGB).

    Use this to validate that colour rendering in Photoshop matches the browser
    preview.  If the sRGB TIFF looks correct but the ProPhoto TIFF looks wrong,
    the issue is with how the editor handles the embedded ICC profile.

    Args:
        image:       H×W×3 float32, linear light, values in [0, 1].
        output_path: Destination path.  Suffix ``_srgb_debug.tiff`` is appended.

    Returns:
        Resolved output path string.
    """
    from ele.export.color_management import apply_srgb_display_trc

    path = Path(output_path)
    debug_path = path.with_stem(path.stem + "_srgb_debug").with_suffix(".tiff")
    safe_mkdir_for_file(debug_path)

    img_f32 = prepare_for_export(image)
    img_srgb = apply_srgb_display_trc(img_f32)
    img_u16  = (img_srgb * 65535.0 + 0.5).astype(np.uint16)

    tifffile.imwrite(
        debug_path,
        img_u16,
        photometric="rgb",
        compression="deflate",
        predictor=True,
        description="ele sRGB debug export - do not use for editing",
    )
    log.info("Debug sRGB TIFF written to %s", debug_path)
    return str(debug_path)
