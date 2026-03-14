"""Stage 2 — Faithful Restoration.

Applies targeted, lightweight corrections based on a DegradationReport.
The goal is faithful recovery of detail, not stylistic enhancement.

Correction cascade
──────────────────
1. Targeted deblocking — suppresses JPEG 8×8 block boundaries that are
   disproportionately stronger than nearby image gradients.  Real content
   edges are detected and left untouched.

2. Anti-ringing — damps overshoot and oscillation in the 2–4-pixel band
   adjacent to high-contrast edges.  Only activates near genuine ringing.

3. Detail-aware denoising — neighbourhood averaging weighted by local
   detail energy; flat areas smoothed, textured areas preserved.

4. Highlight pre-softening — reduces hard clipping shoulder to preserve
   recovery headroom for the pseudo-RAW reconstruction stage.

All operations work in linear-light float32.  No ML or GPU required.
"""

from __future__ import annotations

import numpy as np

from ele.types import DegradationReport


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def faithful_restore(
    image: np.ndarray,
    report: DegradationReport,
) -> np.ndarray:
    """Apply targeted faithful restoration to a linear float32 RGB image.

    Correction strength is jointly modulated by compression_score,
    ringing_score, and sharpness_score so that high-quality JPEGs remain
    mostly untouched while badly compressed images receive proportionally
    stronger cleanup.

    Args:
        image: H×W×3 float32, linear light, values nominally in [0, 1].
        report: DegradationReport from Stage 1.

    Returns:
        Restored image, same shape and dtype.
    """
    img = image.astype(np.float32)

    # Quality factor: high sharpness = more real detail to preserve → lighter touch
    # Range [0.55, 1.0] — never zeroes out corrections entirely
    quality_factor = float(np.clip(1.0 - report.sharpness_score * 0.45, 0.55, 1.0))

    # 1. Targeted deblocking — strongest across likely 8×8 block boundaries
    if report.compression_score > 0.15:
        strength = min(report.compression_score * 0.75, 0.55) * quality_factor
        img = _apply_deblocking(img, strength=strength)

    # 2. Anti-ringing — only near strong edges, only when ringing is detected
    if report.ringing_score > 0.20:
        strength = min(report.ringing_score * 0.55, 0.45) * quality_factor
        img = _apply_anti_ringing(img, strength=strength)

    # 3. Mild denoising — only in flat / low-detail areas
    if report.noise_score > 0.2:
        strength = min(report.noise_score * 0.5, 0.4)
        img = _detail_aware_denoise(img, strength=strength)

    # 4. Highlight pre-softening — reduce hard clipping shoulder
    if report.clipping_score > 0.1:
        img = _soften_highlights(img, threshold=0.92, strength=0.35)

    return img


# ---------------------------------------------------------------------------
# Deblocking
# ---------------------------------------------------------------------------

def _protect_true_edges(gray: np.ndarray) -> np.ndarray:
    """Return a per-pixel real-edge confidence map in [0, 1].

    High values mark pixels that are part of genuine content edges.
    These pixels should receive little or no deblocking correction.

    Method: Sobel-style gradient magnitude normalised to [0, 1].
    The 80th percentile is used as a soft ceiling so that the strongest
    content edges are fully protected without penalising mild gradients.
    """
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    grad = np.sqrt(gy ** 2 + gx ** 2)

    ceiling = float(np.percentile(grad, 80)) + 1e-6
    return np.clip(grad / ceiling, 0.0, 1.0).astype(np.float32)


def _protect_texture_regions(
    weight: np.ndarray,
    gray: np.ndarray,
    protect_strength: float = 0.70,
) -> np.ndarray:
    """Attenuate a correction weight map in high-variance (textured) regions.

    Computes local variance in a 5×5 window.  Pixels in the top quartile
    of variance are likely genuine texture (hair, fabric, foliage, skin
    pores) and should not be smoothed by compression cleanup.

    Args:
        weight:           H×W float32 correction weight to attenuate.
        gray:             H×W float32 luminance.
        protect_strength: Max attenuation fraction in peak-texture zones (0–1).

    Returns:
        Attenuated weight map, same shape.
    """
    from scipy.ndimage import uniform_filter  # type: ignore[import]

    lum_sq_mean = uniform_filter(gray ** 2, size=5)
    lum_mean_sq = uniform_filter(gray, size=5) ** 2
    local_var   = np.clip(lum_sq_mean - lum_mean_sq, 0.0, None)

    var_ceil = float(np.percentile(local_var, 75)) + 1e-8
    texture  = np.clip(local_var / var_ceil, 0.0, 1.0)

    return (weight * (1.0 - protect_strength * texture)).astype(np.float32)


def _apply_deblocking(image: np.ndarray, strength: float) -> np.ndarray:
    """Suppress JPEG 8×8 block boundaries using per-boundary artifact weighting.

    Algorithm
    ─────────
    For each block boundary (column or row at a multiple of 8):

    1. Measure the luminance discontinuity crossing the boundary (bd).
    2. Measure the average luminance gradient 3–4 pixels away from the
       boundary on both sides (nd — "nearby interior diff").
    3. Compute artifact_weight = clamp((bd/nd − 1) / 1.5, 0, 1) × strength.
       → High when the boundary jump greatly exceeds nearby gradients
         (true block artifact).
       → Near zero when nearby gradients are equally large (real edge).
    4. Blend the ±2 pixel boundary zone toward a Gaussian-smoothed reference
       proportionally to artifact_weight × real-edge protection.

    Texture protection
    ──────────────────
    The per-pixel artifact weight is further attenuated in high-variance
    (textured) regions via _protect_texture_regions, so hair, fabric, and
    foliage are not softened even if they happen to lie on a grid boundary.

    Real-edge protection
    ─────────────────────
    _protect_true_edges returns a per-pixel edge confidence map; high-
    confidence edge pixels receive proportionally weaker correction.
    """
    from scipy.ndimage import gaussian_filter  # type: ignore[import]

    h, w, _ = image.shape
    result   = image.copy()

    lum    = _luma(image)
    smooth = np.stack(
        [gaussian_filter(image[..., c], sigma=1.0) for c in range(3)],
        axis=-1,
    )

    # Precompute protection masks (H×W)
    edge_map    = _protect_true_edges(lum)           # 1 = real edge
    texture_map = _protect_texture_regions(           # 1 = safe to correct
        np.ones_like(lum), lum
    )
    # Combined protection: correction is scaled down near real edges AND texture
    protection  = (1.0 - edge_map * 0.80) * texture_map   # H×W

    # ── Horizontal boundaries ─────────────────────────────────────────────────
    for x in range(8, w, 8):
        if x >= w:
            continue

        bd = np.abs(lum[:, x] - lum[:, x - 1])  # H

        nd_samples: list[np.ndarray] = []
        for off in (3, 4):
            if x - off - 1 >= 0:
                nd_samples.append(np.abs(lum[:, x - off] - lum[:, x - off - 1]))
            if x + off < w:
                nd_samples.append(np.abs(lum[:, x + off] - lum[:, x + off - 1]))
        nd = np.mean(nd_samples, axis=0) + 1e-6 if nd_samples else np.full(h, 1e-6)

        artifact_w = np.clip((bd / nd - 1.0) / 1.5, 0.0, 1.0) * strength  # H

        # Apply at ±2 pixel zone with triangular falloff
        for xi, w_frac in ((x - 2, 0.35), (x - 1, 1.0), (x, 1.0), (x + 1, 0.35)):
            if not (0 <= xi < w):
                continue
            blend = (artifact_w * w_frac * protection[:, xi])[:, np.newaxis]
            result[:, xi, :] = (
                (1.0 - blend) * result[:, xi, :]
                + blend * smooth[:, xi, :]
            )

    # ── Vertical boundaries ───────────────────────────────────────────────────
    for y in range(8, h, 8):
        if y >= h:
            continue

        bd = np.abs(lum[y, :] - lum[y - 1, :])  # W

        nd_samples = []
        for off in (3, 4):
            if y - off - 1 >= 0:
                nd_samples.append(np.abs(lum[y - off, :] - lum[y - off - 1, :]))
            if y + off < h:
                nd_samples.append(np.abs(lum[y + off, :] - lum[y + off - 1, :]))
        nd = np.mean(nd_samples, axis=0) + 1e-6 if nd_samples else np.full(w, 1e-6)

        artifact_w = np.clip((bd / nd - 1.0) / 1.5, 0.0, 1.0) * strength  # W

        for yi, w_frac in ((y - 2, 0.35), (y - 1, 1.0), (y, 1.0), (y + 1, 0.35)):
            if not (0 <= yi < h):
                continue
            blend = (artifact_w * w_frac * protection[yi, :])[np.newaxis, :, np.newaxis]
            result[yi, :, :] = (
                (1.0 - blend) * result[yi, :, :]
                + blend * smooth[yi, :, :]
            )

    return result.clip(0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Anti-ringing
# ---------------------------------------------------------------------------

def _apply_anti_ringing(image: np.ndarray, strength: float) -> np.ndarray:
    """Suppress edge-adjacent ringing (overshoot / oscillation / halos).

    Algorithm
    ─────────
    1. Detect strong edges (top 15 % of gradient magnitude).
    2. Build a ring zone: 2–4 pixels from strong edges, excluding the 1-pixel
       edge core itself.  JPEG ringing lives in exactly this band.
    3. Compute a Gaussian-smooth reference (σ=2.5) that approximates what
       the image would look like without ringing oscillations.
    4. Measure per-pixel deviation |image − smooth| inside the ring zone.
       Large deviations are candidate ringing pixels.
    5. Blend those pixels toward the smooth reference proportionally to the
       deviation magnitude and the overall strength parameter.

    Real-edge pixels (core + 1 px) are excluded from the blend zone, so
    edge sharpness is not degraded.  Texture regions are further protected
    by _protect_texture_regions.
    """
    from scipy.ndimage import gaussian_filter, binary_dilation  # type: ignore[import]

    lum = _luma(image)

    # Strong edge detection
    gy       = np.abs(np.diff(lum, axis=0, prepend=lum[:1, :]))
    gx       = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
    grad_mag = np.sqrt(gy ** 2 + gx ** 2)

    edge_thresh = float(np.percentile(grad_mag, 85))
    if edge_thresh < 1e-5:
        return image  # flat image, nothing to do

    strong_edges = grad_mag > edge_thresh

    # Ring zone: 2–4 px halo around edges, not on the edges
    near4    = binary_dilation(strong_edges, iterations=4)
    near1    = binary_dilation(strong_edges, iterations=1)
    ring_zone = (near4 & ~near1).astype(np.float32)   # H×W, 0/1

    if ring_zone.sum() < 50:
        return image

    # Smooth reference for the ring zone
    smooth = np.stack(
        [gaussian_filter(image[..., c], sigma=2.5) for c in range(3)],
        axis=-1,
    )

    # Per-pixel ringing weight: deviation from smooth, gated to ring zone
    deviation   = np.abs(image - smooth).max(axis=2)           # H×W
    ring_weight = np.clip(deviation / 0.018, 0.0, 1.0) * ring_zone  # H×W

    # Texture protection (don't smooth genuine fine texture)
    ring_weight = _protect_texture_regions(ring_weight, lum, protect_strength=0.65)

    blend  = (ring_weight * strength)[..., np.newaxis]
    result = image + blend * (smooth - image)
    return result.clip(0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Denoising and highlight helpers (retained from prior version)
# ---------------------------------------------------------------------------

def _detail_aware_denoise(image: np.ndarray, strength: float) -> np.ndarray:
    """Neighbourhood averaging, attenuated by local detail energy.

    In high-detail areas the blend is near zero (detail preserved).
    In flat areas the blend can be up to *strength* (noise smoothed).
    """
    from scipy.ndimage import uniform_filter  # type: ignore[import]

    blurred = np.stack(
        [uniform_filter(image[..., c], size=3) for c in range(3)],
        axis=-1,
    )

    detail      = np.abs(image - blurred).mean(axis=2)
    detail_norm = np.clip(detail / (detail.max() + 1e-8), 0.0, 1.0)
    blend       = strength * (1.0 - detail_norm)

    result = image + blend[..., np.newaxis] * (blurred - image)
    return result.astype(np.float32)


def _soften_highlights(
    image: np.ndarray,
    threshold: float = 0.92,
    strength: float  = 0.35,
) -> np.ndarray:
    """Gently soften the transition into near-clipped highlights.

    Applies a mild per-pixel blend toward a lower value when the pixel
    luminance is near the clipping ceiling.  Does not affect shadows or
    midtones.
    """
    from ele.utils import luminance

    lum  = luminance(image)
    mask = np.clip((lum - threshold) / (1.0 - threshold), 0.0, 1.0)

    target = image * (threshold / (lum[..., np.newaxis] + 1e-8))
    target = np.clip(target, 0.0, 1.0)

    blend  = mask[..., np.newaxis] * strength
    result = image * (1.0 - blend) + target * blend
    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def _luma(image: np.ndarray) -> np.ndarray:
    """Rec. 709 luminance from linear-light RGB."""
    return (
        0.2126 * image[..., 0]
        + 0.7152 * image[..., 1]
        + 0.0722 * image[..., 2]
    ).astype(np.float32)
