"""Print pipeline.

Full pipeline + super resolution before TIFF export.
Supported scales: 2, 4, 6.
"""

from __future__ import annotations

from PIL import Image

from ele.config import MAX_LONG_EDGE, SUPPORTED_PRINT_SCALES
from ele.core.degradation_analysis import analyse
from ele.core.restoration import faithful_restore
from ele.core.scene_reconstruction import reconstruct_scene
from ele.core.pseudo_raw_reconstruction import reconstruct_pseudo_raw
from ele.core.upscale import upscale_image
from ele.export.tiff_export import export_tiff
from ele.pipeline.base import build_metadata
from ele.types import PipelineResult
from ele.utils import pil_to_float32_linear_rgb, resize_long_edge


def run_print_pipeline(
    input_path: str,
    output_path: str,
    scale: int | None = 2,
    print_style: str = "natural",
    _progress_cb=None,
) -> PipelineResult:
    """Run the print-tier pipeline with super resolution.

    Args:
        input_path:   Path to input JPEG / PNG / TIFF.
        output_path:  Destination TIFF path.
        scale:        Upscale factor (2, 4, or 6).  Defaults to 2.
        print_style:  Upscale style: ``"natural"`` (default) or ``"ai-detail"``.
        _progress_cb: Optional callable(step: int, label: str).

    Returns:
        PipelineResult.

    Raises:
        ValueError: If scale or print_style is not supported.
    """
    cb = _progress_cb or (lambda s, l: None)
    _scale = scale if scale in SUPPORTED_PRINT_SCALES else 2

    cb(1, "loading image")
    with Image.open(input_path) as pil_img:
        image = pil_to_float32_linear_rgb(pil_img)

    image = resize_long_edge(image, MAX_LONG_EDGE["print"])

    cb(2, "analyzing degradation")
    report = analyse(image)

    cb(3, "faithful restoration")
    image = faithful_restore(image, report)

    cb(4, "scene reconstruction")
    image, scene_map = reconstruct_scene(image)

    cb(5, "pseudo-RAW reconstruction")
    image = reconstruct_pseudo_raw(image, report, scene_map)

    cb(5, f"super resolution ×{_scale} [{print_style}]")
    image = upscale_image(image, scale=_scale, mode="print")

    cb(6, "exporting TIFF")
    # image has already been upscaled; build_metadata reads final shape for export_size
    metadata = build_metadata(
        "print", input_path, report, image,
        extra={"scale": str(_scale), "print_style": print_style},
    )
    out = export_tiff(image, output_path, metadata=metadata)

    return PipelineResult(
        image=image,
        report=report,
        output_path=out,
        metadata=metadata,
    )
