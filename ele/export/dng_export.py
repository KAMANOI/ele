"""Stage 6b — Linear DNG export (planned, not yet implemented).

Linear DNG is the pro-tier output format of ele.  A proper Linear DNG
requires precise construction of DNG-specific TIFF tags including:

  - DNGVersion / DNGBackwardVersion
  - CalibrationIlluminant1/2
  - ColorMatrix1/2 (scene-to-XYZ)
  - ForwardMatrix1/2
  - AsShotNeutral / AsShotWhiteXY
  - LinearityLimit / BlackLevel / WhiteLevel
  - ActiveArea / DefaultCropOrigin / DefaultCropSize
  - Embedded JPEG thumbnail (IFD0 SubIFD)

This level of binary TIFF/DNG construction is planned for the next
implementation milestone.  For now, calls to this module raise
NotImplementedError with a descriptive message.

Pro mode currently falls back to 16-bit TIFF export.
"""

from __future__ import annotations

import numpy as np


def export_dng(
    image: np.ndarray,
    output_path: str,
    metadata: dict | None = None,
    embed_preview: bool = True,
) -> str:
    """Export a linear float32 image as a Linear DNG file.

    NOT YET IMPLEMENTED.

    Args:
        image:         H×W×3 float32 linear ProPhoto RGB.
        output_path:   Destination .dng path.
        metadata:      Optional pipeline metadata dict.
        embed_preview: If True, embed a JPEG preview (planned).

    Raises:
        NotImplementedError: Always.  Linear DNG writing is reserved for
            the next implementation step.  Use TIFF export for now.
    """
    raise NotImplementedError(
        "Linear DNG export is planned but not implemented in this build.\n"
        "Use --output out.tiff for 16-bit TIFF (fully supported).\n"
        "DNG support will be added in the next release."
    )
