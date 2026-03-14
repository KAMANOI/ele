"""ele configuration constants."""

from __future__ import annotations

APP_NAME = "ele"
APP_VERSION = "0.1.0"

# Internal precision
INTERNAL_DTYPE = "float32"

# Max long edge per mode (pixels).
# None means no limit — image is processed and exported at full input resolution.
MAX_LONG_EDGE: dict[str, int | None] = {
    "free":    4000,
    "creator": 8000,
    "pro":     None,    # no forced resize — export at full input resolution
    "print":   16000,   # ceiling before upscale; after SR may exceed this
}

# TIFF
DEFAULT_TIFF_COMPRESSION = "deflate"  # zlib/deflate; no imagecodecs needed

# Super resolution
SUPPORTED_PRINT_SCALES  = (2, 4, 6)
SUPPORTED_PRINT_STYLES  = ("natural", "ai-detail")

# Gamma
SRGB_GAMMA = 2.2
LINEAR_GAMMA = 1.0

# Color pipeline description strings
COLOR_PRIMARIES = "ProPhoto RGB (ROMM RGB)"
WHITE_POINT = "D50"
TRANSFER_FUNCTION = "Linear (gamma 1.0)"
BIT_DEPTH_EXPORT = 16

# Clipping thresholds for degradation analysis
CLIP_LOW_THRESHOLD  = 0.02   # below this → near-black clipping
CLIP_HIGH_THRESHOLD = 0.98   # above this → near-white clipping
CLIP_FRACTION_WARN  = 0.005  # 0.5 % of pixels triggers a note
