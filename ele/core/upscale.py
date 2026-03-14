"""Unified upscale module — three-tier mode dispatch.

All modes operate entirely in linear-light float32.  No external AI model
or GPU dependency is required.  The output of every path is a clamped
float32 array compatible with the 16-bit ProPhoto TIFF export stage.

Modes
─────
creator
    Pure Lanczos interpolation via Pillow uint16 round-trip.
    Fastest path; no post-processing.  Suitable when the editor will do
    their own sharpening in Lightroom / Photoshop.

pro
    Lanczos resize followed by mild edge-aware unsharp mask.
    Sharpening strength is modulated by a luminance gradient map so smooth
    areas (skin gradients, defocused backgrounds) receive no boost while
    genuine edges (hair, fabric, eyelashes) are sharpened precisely.
    Output clamped to [0, 1] — prevents bright halo clipping.

print
    Lanczos resize + two-pass high-quality post-processing:

    Pass 1 — Anti-ringing suppression
        LANCZOS can produce overshoot (ringing) at high-contrast edges.
        A Gaussian-difference risk map identifies affected pixels;
        a targeted smooth blend damps the overshoot without softening
        the rest of the image.

    Pass 2 — Print-optimised local contrast
        Mild unsharp mask tuned for larger viewing distances and print
        dot-gain.  Radius is tighter than pro mode (0.6 px vs 1.0 px)
        to avoid coarse halos at print resolution; amount is slightly
        higher to compensate for perceptual contrast loss at distance.

    Output clamped to [0, 1].

AI super-resolution (Real-ESRGAN) is reserved as a future optional
server-side feature and is not part of this module.

Public API
──────────
    upscale_image(image, scale, mode) -> np.ndarray

Returns H*scale × W*scale × 3 float32 in [0, 1].
"""

from __future__ import annotations

import logging

import numpy as np
from PIL import Image

from ele.utils import float32_linear_rgb_to_uint16, clamp01

logger = logging.getLogger(__name__)


# ── Tuning constants ───────────────────────────────────────────────────────────

# pro mode — edge-aware sharpening
_PRO_EDGE_SIGMA_DETECT = 1.0    # Gaussian sigma used for edge detection
_PRO_EDGE_SIGMA_BLUR   = 0.8    # Gaussian sigma for the sharpening kernel
_PRO_EDGE_THRESHOLD    = 0.05   # Gradient magnitude that counts as a hard edge
_PRO_EDGE_AMOUNT       = 0.40   # Unsharp-mask strength at full edge weight

# print mode — anti-ringing pass
_PRINT_RING_SIGMA      = 0.5    # Detection blur sigma (sub-pixel ringing)
_PRINT_RING_THRESHOLD  = 0.04   # Ringing magnitude threshold
_PRINT_RING_AMOUNT     = 0.35   # Blend-toward-smooth strength at full risk

# print mode — local contrast pass
_PRINT_LC_RADIUS       = 0.6    # Tighter radius than pro — avoids coarse halos
_PRINT_LC_AMOUNT       = 0.12   # Slightly stronger than pro to recover print contrast


# ── Public API ────────────────────────────────────────────────────────────────

def upscale_image(
    image: np.ndarray,
    scale: int,
    mode: str,
) -> np.ndarray:
    """Upscale *image* by *scale* using the strategy for *mode*.

    Args:
        image: H×W×3 float32 linear RGB, values in [0, 1].
        scale: Integer upscale factor (e.g. 2, 4).
        mode:  ``"creator"``, ``"pro"``, or ``"print"``.

    Returns:
        H*scale × W*scale × 3 float32, values in [0, 1].

    Raises:
        ValueError: If *mode* is not recognised.
    """
    logger.info("Upscale mode: %s  ×%d", mode, scale)

    if mode == "creator":
        return lanczos_resize(image, scale)

    if mode == "pro":
        img = lanczos_resize(image, scale)
        return edge_sharpen(img)

    if mode == "print":
        img = lanczos_resize(image, scale)
        img = _suppress_ringing(img)
        img = _local_contrast(img, radius=_PRINT_LC_RADIUS, amount=_PRINT_LC_AMOUNT)
        return clamp01(img)

    raise ValueError(
        f"Unknown upscale mode {mode!r}. Expected 'creator', 'pro', or 'print'."
    )


# ── Core resize ───────────────────────────────────────────────────────────────

def lanczos_resize(image: np.ndarray, scale: int) -> np.ndarray:
    """Per-channel Lanczos resize via Pillow uint16 round-trip.

    Processing each channel independently through a uint16 intermediate
    preserves more precision than uint8 and prevents luminance/chroma
    cross-contamination inside the Pillow resampling kernel.
    """
    h, w = image.shape[:2]
    new_h, new_w = h * scale, w * scale

    u16 = float32_linear_rgb_to_uint16(image)
    channels: list[np.ndarray] = []
    for c in range(3):
        ch_pil = Image.fromarray(u16[..., c])
        ch_pil = ch_pil.resize((new_w, new_h), resample=Image.LANCZOS)
        channels.append(np.asarray(ch_pil, dtype=np.uint16))

    return np.stack(channels, axis=-1).astype(np.float32) / 65535.0


# ── pro mode helper ───────────────────────────────────────────────────────────

def edge_sharpen(image: np.ndarray) -> np.ndarray:
    """Mild edge-aware unsharp mask for pro mode.

    Sharpening is gated by a per-pixel luminance gradient weight so that
    smooth tonal regions (skin, bokeh, sky) are unaffected while structural
    edges receive a proportional boost.

    Steps
    ─────
    1. Compute Rec. 709 luminance Y.
    2. Detect edges: edge_weight = clamp(|Y − gauss(Y, σ=1.0)| / 0.05, 0, 1).
    3. Compute detail layer: image − gauss(image, σ=0.8).
    4. result = image + 0.40 × edge_weight × detail.
    5. Clamp to [0, 1] to prevent halo clipping.
    """
    from scipy.ndimage import gaussian_filter  # type: ignore[import]

    lum = _luminance(image)

    lum_blur    = gaussian_filter(lum, sigma=_PRO_EDGE_SIGMA_DETECT)
    edge_weight = np.clip(
        np.abs(lum - lum_blur) / _PRO_EDGE_THRESHOLD, 0.0, 1.0
    )[..., np.newaxis]

    blur = np.stack(
        [gaussian_filter(image[..., c], sigma=_PRO_EDGE_SIGMA_BLUR) for c in range(3)],
        axis=-1,
    )
    detail    = image - blur
    sharpened = image + _PRO_EDGE_AMOUNT * edge_weight * detail

    return np.clip(sharpened, 0.0, 1.0).astype(np.float32)


# ── print mode helpers ────────────────────────────────────────────────────────

def _suppress_ringing(image: np.ndarray) -> np.ndarray:
    """Damp LANCZOS overshoot (ringing) at high-contrast edges.

    LANCZOS can produce bright/dark fringes (ringing) at sharp transitions.
    This pass measures the per-pixel overshoot risk via |image − gauss(image)|
    and blends affected pixels toward the smooth reference proportionally.

    Pixels with no overshoot risk (diff ≈ 0) are completely unchanged.
    The blend is progressive — low-risk pixels receive a small correction,
    high-risk pixels receive a stronger correction up to _PRINT_RING_AMOUNT.
    """
    from scipy.ndimage import gaussian_filter  # type: ignore[import]

    smooth = np.stack(
        [gaussian_filter(image[..., c], sigma=_PRINT_RING_SIGMA) for c in range(3)],
        axis=-1,
    )
    risk = np.clip(
        np.abs(image - smooth).max(axis=2) / _PRINT_RING_THRESHOLD, 0.0, 1.0
    )[..., np.newaxis]

    return (
        image * (1.0 - risk * _PRINT_RING_AMOUNT)
        + smooth * (risk * _PRINT_RING_AMOUNT)
    ).clip(0.0, 1.0).astype(np.float32)


def _local_contrast(
    image: np.ndarray,
    radius: float,
    amount: float,
) -> np.ndarray:
    """Unsharp-mask local contrast enhancement.

    Adds a fraction of the high-frequency detail layer back to the image.
    Used as the final step in print mode with tighter radius than pro to
    avoid coarse halos at print resolution.
    """
    from scipy.ndimage import gaussian_filter  # type: ignore[import]

    blur   = np.stack(
        [gaussian_filter(image[..., c], sigma=radius) for c in range(3)],
        axis=-1,
    )
    detail = image - blur
    return (image + amount * detail).astype(np.float32)


# ── Shared utility ────────────────────────────────────────────────────────────

def _luminance(image: np.ndarray) -> np.ndarray:
    """Rec. 709 luminance (correct for linear-light RGB)."""
    return (
        0.2126 * image[..., 0]
        + 0.7152 * image[..., 1]
        + 0.0722 * image[..., 2]
    ).astype(np.float32)
