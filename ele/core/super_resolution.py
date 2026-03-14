"""Stage 5 (optional) — Print Super Resolution.

Two upscale styles are supported:

natural  (default)
    Photography-first interpolation pipeline.  Prioritises smooth tonal
    gradients, clean edges, and natural print appearance.  Anti-crunch
    cleanup suppresses LANCZOS ringing; portrait-region protection keeps
    hair, fabric, and foliage from looking crispy.

ai-detail
    Stronger perceived-detail pathway intended for AI-generated images and
    visually dense work where synthetic-looking texture is acceptable.

    ⚠  NOTE: ai-detail is NOT backed by a trained generative SR model
    (Real-ESRGAN / SUPIR / etc.) in this build.  It is implemented as a
    multi-scale unsharp-mask enhancement on top of LANCZOS upscaling.
    The placeholder is structured so a real model can be plugged in behind
    ``_upscale_ai_detail()`` without changing the public API.

Public API
──────────
    upscale_for_print(image, scale, style="natural") -> np.ndarray

Operates entirely in linear light (float32) to avoid gamma artefacts.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from ele.config import SUPPORTED_PRINT_SCALES, SUPPORTED_PRINT_STYLES
from ele.utils import float32_linear_rgb_to_uint16, clamp01


# ── Public API ────────────────────────────────────────────────────────────────

def upscale_for_print(
    image: np.ndarray,
    scale: int,
    style: str = "natural",
) -> np.ndarray:
    """Upscale a linear float32 image for print output.

    Args:
        image: H×W×3 float32 linear RGB, values in [0, 1].
        scale: Upscale factor.  Must be one of SUPPORTED_PRINT_SCALES (2, 4, 6).
        style: Upscale style.  One of ``"natural"`` or ``"ai-detail"``.
               Defaults to ``"natural"``.

    Returns:
        Upscaled image, same dtype, shape (H*scale, W*scale, 3).

    Raises:
        ValueError: If scale or style is not supported.
    """
    if scale not in SUPPORTED_PRINT_SCALES:
        raise ValueError(
            f"Unsupported scale {scale!r}. Supported: {SUPPORTED_PRINT_SCALES}"
        )
    if style not in SUPPORTED_PRINT_STYLES:
        raise ValueError(
            f"Unsupported print style {style!r}. "
            f"Supported: {list(SUPPORTED_PRINT_STYLES)}"
        )

    if style == "natural":
        return _upscale_natural(image, scale)
    else:  # "ai-detail"
        return _upscale_ai_detail(image, scale)


# ── Natural style ─────────────────────────────────────────────────────────────

def _upscale_natural(image: np.ndarray, scale: int) -> np.ndarray:
    """Photography-first interpolation upscale.

    Pipeline
    ────────
    1. Per-channel LANCZOS upscale via Pillow (uint16 round-trip).
    2. Anti-crunch cleanup — suppress LANCZOS ringing in high-frequency zones.
    3. Portrait-region protection — extra gentle smoothing on hair / dark-low-sat.
    4. Mild local contrast recovery — lower strength than ai-detail to avoid
       crunchy appearance.
    """
    upscaled = _lanczos_upscale(image, scale)
    upscaled = _apply_anti_crunch_cleanup(upscaled)
    upscaled = _protect_portrait_regions(upscaled)
    upscaled = _local_contrast_enhance(upscaled, radius=1.2, amount=0.08)
    return clamp01(upscaled)


# ── AI-detail style ───────────────────────────────────────────────────────────

def _upscale_ai_detail(image: np.ndarray, scale: int) -> np.ndarray:
    """Stronger perceived-detail upscale intended for AI-generated images.

    ⚠  This is an enhanced multi-scale interpolation pass, NOT a trained
    generative SR model.  It produces visually sharper output that may
    appear synthetic on photographic subjects (skin, hair, foliage).

    Pipeline
    ────────
    1. Per-channel LANCZOS upscale.
    2. Multi-scale detail enhancement (fine + mid frequency layers).
    3. Fine local contrast recovery (stronger than natural style).

    Plug-in point: replace the body of this function with a Real-ESRGAN /
    SUPIR / other model call without changing the public API.
    """
    upscaled = _lanczos_upscale(image, scale)
    upscaled = _apply_detail_enhancement(upscaled)
    upscaled = _local_contrast_enhance(upscaled, radius=0.5, amount=0.18)
    return clamp01(upscaled)


# ── Core upscale kernel ───────────────────────────────────────────────────────

def _lanczos_upscale(image: np.ndarray, scale: int) -> np.ndarray:
    """Per-channel LANCZOS resize via Pillow uint16 round-trip."""
    h, w = image.shape[:2]
    new_h, new_w = h * scale, w * scale

    u16 = float32_linear_rgb_to_uint16(image)
    channels: list[np.ndarray] = []
    for c in range(3):
        ch_pil = Image.fromarray(u16[..., c])
        ch_pil = ch_pil.resize((new_w, new_h), resample=Image.LANCZOS)
        channels.append(np.asarray(ch_pil, dtype=np.uint16))

    return np.stack(channels, axis=-1).astype(np.float32) / 65535.0


# ── Natural helpers ───────────────────────────────────────────────────────────

def _apply_anti_crunch_cleanup(
    image: np.ndarray,
    sigma: float = 0.5,
    threshold: float = 0.04,
    amount: float = 0.35,
) -> np.ndarray:
    """Suppress LANCZOS ringing and crunchy texture in natural upscale output.

    LANCZOS can introduce overshoot (ringing) at high-contrast edges, which
    looks like crispy or jagged texture — especially in hair, fine fabric,
    and foliage edges.

    Method
    ──────
    1. Compute a mildly blurred reference (sigma=0.5).
    2. Measure per-pixel ringing magnitude: max |image − blurred| across channels.
    3. Where magnitude exceeds threshold, blend toward the smooth version.
       The blend is progressive: gentle at low risk, stronger at high risk.

    Smooth areas (diff ≈ 0) are completely unaffected.
    """
    from scipy.ndimage import gaussian_filter  # type: ignore[import]

    blurred = np.stack(
        [gaussian_filter(image[..., c], sigma=sigma) for c in range(3)],
        axis=-1,
    )
    diff    = np.abs(image - blurred).max(axis=2)           # H×W
    risk    = np.clip(diff / threshold, 0.0, 1.0)[..., np.newaxis]  # H×W×1

    return (image * (1.0 - risk * amount) + blurred * (risk * amount)
            ).clip(0.0, 1.0).astype(np.float32)


def _protect_portrait_regions(
    image: np.ndarray,
    sigma: float = 0.6,
    strength: float = 0.22,
) -> np.ndarray:
    """Extra gentle smoothing in hair, dark-skin, and low-saturation regions.

    After upscaling, dark low-saturation pixels (hair, eyebrows, dark fabric,
    deep shadow foliage) are the most likely to show crispy or zipper-edge
    artefacts.  A small Gaussian blend weighted by a hair-region heuristic
    keeps these areas clean without blurring bright mid-tone structure.

    Hair/dark-fabric heuristic: luminance < 0.28  and  inter-channel spread < 0.15
    Mask is Gaussian-smoothed (sigma=2) for feathered region boundaries.
    """
    from scipy.ndimage import gaussian_filter  # type: ignore[import]

    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    low_sat     = (np.abs(r - g) + np.abs(g - b)) < 0.15
    dark        = lum < 0.28
    region_raw  = (dark & low_sat).astype(np.float32)
    region_w    = gaussian_filter(region_raw, sigma=2.0)[..., np.newaxis]  # H×W×1

    soft = np.stack(
        [gaussian_filter(image[..., c], sigma=sigma) for c in range(3)],
        axis=-1,
    )
    return (image * (1.0 - region_w * strength) + soft * (region_w * strength)
            ).clip(0.0, 1.0).astype(np.float32)


def _protect_edges(
    image: np.ndarray,
    upsampled_ref: np.ndarray,
    sigma_edge: float = 0.8,
    strength: float = 0.15,
) -> np.ndarray:
    """Reduce zipper artefacts in the pixels immediately adjacent to strong edges.

    Detects edge-adjacent pixels using gradient magnitude of a smoothed
    luminance map, then applies a targeted blend toward a mildly smoothed
    version to prevent staircase / aliasing artefacts in those zones.

    This function is available for callers who need it but is not in the
    default natural pipeline — the anti-crunch pass handles most cases.
    """
    from scipy.ndimage import gaussian_filter  # type: ignore[import]

    lum_r = 0.2126 * upsampled_ref[..., 0] + 0.7152 * upsampled_ref[..., 1] + 0.0722 * upsampled_ref[..., 2]
    lum_s = gaussian_filter(lum_r, sigma=sigma_edge)
    grad  = np.abs(lum_r - lum_s)                    # H×W edge proxy
    edge_w = np.clip(grad / 0.05, 0.0, 1.0)[..., np.newaxis]

    soft = np.stack(
        [gaussian_filter(image[..., c], sigma=sigma_edge * 0.5) for c in range(3)],
        axis=-1,
    )
    return (image * (1.0 - edge_w * strength) + soft * (edge_w * strength)
            ).clip(0.0, 1.0).astype(np.float32)


# ── AI-detail helpers ─────────────────────────────────────────────────────────

def _apply_detail_enhancement(
    image: np.ndarray,
    fine_radius: float = 0.4,
    fine_amount: float = 0.20,
    mid_radius: float  = 1.5,
    mid_amount: float  = 0.08,
) -> np.ndarray:
    """Multi-scale unsharp-mask detail enhancement for ai-detail upscale.

    ⚠  This is a signal-processing enhancement, NOT a trained SR model.
    It recovers perceived fine detail by boosting two spatial frequency
    bands independently.  On photographic subjects it may introduce
    synthetic-looking texture.

    Fine band  (sigma ≈ 0.4): sub-pixel to 1-pixel texture detail.
    Mid band   (sigma ≈ 1.5): edge micro-contrast and mid-frequency structure.
    """
    from scipy.ndimage import gaussian_filter  # type: ignore[import]

    fine_blur   = np.stack(
        [gaussian_filter(image[..., c], sigma=fine_radius) for c in range(3)],
        axis=-1,
    )
    fine_detail = image - fine_blur

    mid_blur    = np.stack(
        [gaussian_filter(image[..., c], sigma=mid_radius) for c in range(3)],
        axis=-1,
    )
    mid_detail  = image - mid_blur

    result = image + fine_amount * fine_detail + mid_amount * mid_detail
    return result.clip(0.0, 1.0).astype(np.float32)


# ── Shared utility ────────────────────────────────────────────────────────────

def _local_contrast_enhance(
    image: np.ndarray,
    radius: float = 1.2,
    amount: float = 0.08,
) -> np.ndarray:
    """Mild unsharp-mask style local contrast for print detail recovery.

    Used as the final step in both styles.  Parameters differ per style:
      natural:   radius=1.2, amount=0.08  — gentle, avoids texture crunch
      ai-detail: radius=0.5, amount=0.18  — stronger fine-detail boost
    """
    from scipy.ndimage import gaussian_filter  # type: ignore[import]

    blurred = np.stack(
        [gaussian_filter(image[..., c], sigma=radius) for c in range(3)],
        axis=-1,
    )
    detail = image - blurred
    return (image + amount * detail).astype(np.float32)
