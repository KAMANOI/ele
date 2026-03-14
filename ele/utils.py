"""ele utility functions."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------

def ensure_rgb_pil(image: Image.Image) -> Image.Image:
    """Convert any PIL image to RGB mode."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def pil_to_float32_linear_rgb(image: Image.Image) -> np.ndarray:
    """Convert an sRGB PIL image to a linear-light float32 array.

    Steps:
      1. Ensure RGB.
      2. Normalise uint8 → [0, 1].
      3. Remove sRGB EOTF (gamma ≈ 2.2 with linear toe) → linear light.

    Returns:
        H×W×3 float32 array in [0, 1].
    """
    image = ensure_rgb_pil(image)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return _srgb_to_linear(arr)


def _srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """Inverse sRGB EOTF (IEC 61966-2-1)."""
    return np.where(
        img <= 0.04045,
        img / 12.92,
        ((img + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def float32_linear_rgb_to_uint16(image: np.ndarray) -> np.ndarray:
    """Convert float32 [0, 1] to uint16 [0, 65535].

    Values outside [0, 1] are clamped first.
    """
    clamped = clamp01(image)
    return (clamped * 65535.0 + 0.5).astype(np.uint16)


def to_uint8_rgb_for_pil(image: np.ndarray) -> np.ndarray:
    """Normalise any H×W×3 array to uint8 for use with Image.fromarray().

    Accepted dtypes:
      - float32 / float64 : assumed [0, 1], clamped and scaled to [0, 255]
      - uint16             : scaled from [0, 65535] to [0, 255]
      - uint32             : scaled from [0, 4294967295] to [0, 255]
      - uint8              : returned as-is
    """
    if image.dtype == np.uint8:
        return image
    if image.dtype in (np.float32, np.float64):
        return (np.clip(image, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if image.dtype == np.uint16:
        return (image >> 8).astype(np.uint8)
    if image.dtype == np.uint32:
        return (image >> 24).astype(np.uint8)
    # Fallback: normalise to [0, 255] by range
    mn, mx = image.min(), image.max()
    if mx == mn:
        return np.zeros(image.shape, dtype=np.uint8)
    return ((image.astype(np.float32) - mn) / (mx - mn) * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def resize_long_edge(image_array: np.ndarray, max_long_edge: int | None) -> np.ndarray:
    """Resize so the longest side is at most *max_long_edge* pixels.

    Uses LANCZOS resampling.  Returns the original array unchanged if:
      - *max_long_edge* is ``None`` (no limit — pass-through for PRO mode), or
      - the image already fits within the limit.
    """
    if max_long_edge is None:
        return image_array

    h, w = image_array.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return image_array

    scale = max_long_edge / long_edge
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    # Pillow operates in uint8 or uint16; use float → uint16 round-trip.
    # PIL cannot handle (H,W,3) uint32 — resize per-channel as uint16 instead.
    u16 = float32_linear_rgb_to_uint16(image_array)
    channels = []
    for c in range(3):
        ch = Image.fromarray(u16[..., c])
        ch = ch.resize((new_w, new_h), resample=Image.LANCZOS)
        channels.append(np.asarray(ch, dtype=np.uint16))
    resized_u16 = np.stack(channels, axis=-1)
    return (resized_u16.astype(np.float32) / 65535.0)


# ---------------------------------------------------------------------------
# Array math helpers
# ---------------------------------------------------------------------------

def safe_mkdir_for_file(path: str | Path) -> None:
    """Create parent directories for a file path if they don't exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def clamp01(array: np.ndarray) -> np.ndarray:
    """Clamp array values to [0, 1]."""
    return np.clip(array, 0.0, 1.0)


def luminance(array: np.ndarray) -> np.ndarray:
    """Compute per-pixel luminance (Rec. 709 coefficients).

    Args:
        array: H×W×3 float32 linear RGB.

    Returns:
        H×W float32 luminance.
    """
    return (
        0.2126 * array[..., 0]
        + 0.7152 * array[..., 1]
        + 0.0722 * array[..., 2]
    )


def apply_per_channel(
    array: np.ndarray,
    fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Apply *fn* independently to each channel of an H×W×C array."""
    return np.stack([fn(array[..., c]) for c in range(array.shape[2])], axis=-1)


def basic_white_balance_from_gray_world(array: np.ndarray) -> np.ndarray:
    """Apply grey-world white balance.

    Scales each channel so its mean equals the overall mean luminance.
    Operates in-place on a copy.
    """
    mean_per_channel = array.mean(axis=(0, 1)).clip(1e-6, None)  # (3,)
    overall_mean = mean_per_channel.mean()
    gains = overall_mean / mean_per_channel
    return clamp01((array * gains[np.newaxis, np.newaxis, :]).astype(np.float32))
