"""Pro pipeline.

Same as creator pipeline.  If output path ends in .dng the DNG stub
raises NotImplementedError; otherwise exports TIFF.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from ele.config import MAX_LONG_EDGE
from ele.core.degradation_analysis import analyse
from ele.core.restoration import faithful_restore
from ele.core.scene_reconstruction import reconstruct_scene
from ele.core.pseudo_raw_reconstruction import reconstruct_pseudo_raw
from ele.export.tiff_export import export_tiff
from ele.export.dng_export import export_dng
from ele.pipeline.base import build_metadata
from ele.types import PipelineResult
from ele.utils import pil_to_float32_linear_rgb, resize_long_edge


def run_pro_pipeline(
    input_path: str,
    output_path: str,
    scale: int | None = None,
    _progress_cb=None,
) -> PipelineResult:
    """Run the pro-tier pipeline.

    Args:
        input_path:   Path to input JPEG / PNG / TIFF.
        output_path:  Destination path.  .tiff/.tif → 16-bit TIFF.
                      .dng → raises NotImplementedError (planned).
        scale:        Ignored in pro mode.
        _progress_cb: Optional callable(step: int, label: str).

    Returns:
        PipelineResult.
    """
    cb = _progress_cb or (lambda s, l: None)

    cb(1, "loading image")
    with Image.open(input_path) as pil_img:
        image = pil_to_float32_linear_rgb(pil_img)

    image = resize_long_edge(image, MAX_LONG_EDGE["pro"])

    cb(2, "analyzing degradation")
    report = analyse(image)

    cb(3, "faithful restoration")
    image = faithful_restore(image, report)

    cb(4, "scene reconstruction")
    image, scene_map = reconstruct_scene(image)

    cb(5, "pseudo-RAW reconstruction")
    image = reconstruct_pseudo_raw(image, report, scene_map)

    cb(6, "exporting")
    metadata = build_metadata("pro", input_path, report, image)

    want_dng = Path(output_path).suffix.lower() == ".dng"
    if want_dng:
        out = export_dng(image, output_path, metadata=metadata)  # raises NotImplementedError
    else:
        out = export_tiff(image, output_path, metadata=metadata)

    return PipelineResult(
        image=image,
        report=report,
        output_path=out,
        metadata=metadata,
    )
