"""Creator pipeline.

Same as free but max long edge is 8000 px.
"""

from __future__ import annotations

from PIL import Image

from ele.config import MAX_LONG_EDGE
from ele.core.degradation_analysis import analyse
from ele.core.restoration import faithful_restore
from ele.core.scene_reconstruction import reconstruct_scene
from ele.core.pseudo_raw_reconstruction import reconstruct_pseudo_raw
from ele.export.tiff_export import export_tiff
from ele.pipeline.base import build_metadata
from ele.types import PipelineResult
from ele.utils import pil_to_float32_linear_rgb, resize_long_edge


def run_creator_pipeline(
    input_path: str,
    output_path: str,
    scale: int | None = None,
    _progress_cb=None,
) -> PipelineResult:
    """Run the creator-tier pipeline.

    Args:
        input_path:   Path to input JPEG / PNG / TIFF.
        output_path:  Destination TIFF path.
        scale:        Ignored in creator mode.
        _progress_cb: Optional callable(step: int, label: str).

    Returns:
        PipelineResult.
    """
    cb = _progress_cb or (lambda s, l: None)

    cb(1, "loading image")
    with Image.open(input_path) as pil_img:
        image = pil_to_float32_linear_rgb(pil_img)

    image = resize_long_edge(image, MAX_LONG_EDGE["creator"])

    cb(2, "analyzing degradation")
    report = analyse(image)

    cb(3, "faithful restoration")
    image = faithful_restore(image, report)

    cb(4, "scene reconstruction")
    image, scene_map = reconstruct_scene(image)

    cb(5, "pseudo-RAW reconstruction")
    image = reconstruct_pseudo_raw(image, report, scene_map)

    cb(6, "exporting TIFF")
    metadata = build_metadata("creator", input_path, report, image)
    out = export_tiff(image, output_path, metadata=metadata)

    return PipelineResult(
        image=image,
        report=report,
        output_path=out,
        metadata=metadata,
    )
