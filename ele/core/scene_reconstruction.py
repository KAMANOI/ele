"""Stage 3 — Scene Reconstruction.

Builds a heuristic SceneMap from a linear float32 RGB image, then
applies scene-aware adjustments to improve editability per region.

No ML models are used.  All masks are computed from simple colour and
spatial heuristics.  The goal is a first-pass plausible segmentation
that guides downstream pseudo-RAW shaping.
"""

from __future__ import annotations

import numpy as np

from ele.types import SceneMap
from ele.utils import luminance, clamp01


def reconstruct_scene(
    image: np.ndarray,
) -> tuple[np.ndarray, SceneMap]:
    """Produce scene-adjusted image and soft region masks.

    Args:
        image: H×W×3 float32 linear RGB, values [0, 1].

    Returns:
        Tuple of:
          - adjusted image (same shape/dtype)
          - SceneMap with per-pixel soft masks
    """
    scene_map = _build_scene_map(image)
    adjusted  = _apply_scene_adjustments(image, scene_map)
    return adjusted, scene_map


# ---------------------------------------------------------------------------
# Scene map construction
# ---------------------------------------------------------------------------

def _build_scene_map(image: np.ndarray) -> SceneMap:
    h, w = image.shape[:2]
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    lum = luminance(image)

    sky          = _mask_sky(r, g, b, lum, h)
    foliage      = _mask_foliage(r, g, b)
    skin         = _mask_skin(r, g, b)
    architecture = _mask_architecture(r, g, b, lum)
    hair         = _mask_hair(r, g, b, lum)

    # Fabric: residual regions not covered by the above
    coverage = clamp01(sky + foliage + skin + architecture + hair)
    fabric   = clamp01(1.0 - coverage)

    return SceneMap(
        sky=sky,
        foliage=foliage,
        skin=skin,
        architecture=architecture,
        hair=hair,
        fabric=fabric,
    )


def _mask_sky(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    lum: np.ndarray,
    image_height: int,
) -> np.ndarray:
    """Bright, blue-dominant pixels in the upper half of the frame."""
    blue_bias  = (b > r) & (b > g)
    bright     = lum > 0.3
    low_sat    = (np.abs(r - g) + np.abs(g - b) + np.abs(b - r)) < 0.4

    h = image_height
    spatial = np.zeros_like(lum)
    spatial[: h // 2, :] = 1.0      # upper half weight

    raw = (blue_bias & bright & low_sat).astype(np.float32) * spatial
    return _smooth_mask(raw)


def _mask_foliage(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Green-dominant pixels."""
    green_dom = (g > r * 1.1) & (g > b * 1.05)
    return _smooth_mask(green_dom.astype(np.float32))


def _mask_skin(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Warm hue band approximating human skin tones.

    Skin typically: r > g > b, moderate luminance, limited saturation.

    Chromaticity ratio gates (R/G and G/B) narrow the hue band so that
    warm non-skin materials (wood, leather, terracotta) are excluded.
    Upper luminance bound raised to 0.90 to capture bright foreheads
    and cheeks that fall outside the old 0.80 limit.
    """
    warm     = (r > g) & (g > b)
    r_dom    = (r - b) > 0.05
    not_red  = (r - g) < 0.35       # exclude saturated reds

    # Chromaticity ratio gates — skin R/G ≈ 1.05–1.50, G/B ≈ 1.10–2.00
    rg_ratio = r / (g + 1e-4)
    gb_ratio = g / (b + 1e-4)
    rg_ok    = (rg_ratio > 1.02) & (rg_ratio < 1.60)
    gb_ok    = (gb_ratio > 1.05) & (gb_ratio < 2.20)

    lum      = luminance(np.stack([r, g, b], axis=-1))
    mid_lum  = (lum > 0.08) & (lum < 0.90)   # extended upper bound for bright skin

    raw = (warm & r_dom & not_red & rg_ok & gb_ok & mid_lum).astype(np.float32)
    return _smooth_mask(raw)


def _mask_architecture(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    lum: np.ndarray,
) -> np.ndarray:
    """Near-neutral, mid-to-high luminance areas — typical of built structures."""
    neutrality = 1.0 - (
        np.abs(r - g) + np.abs(g - b) + np.abs(b - r)
    ).clip(0, 1)
    near_neutral = neutrality > 0.6
    mid_lum      = (lum > 0.2) & (lum < 0.85)

    raw = (near_neutral & mid_lum).astype(np.float32)
    return _smooth_mask(raw)


def _mask_hair(
    r: np.ndarray,
    g: np.ndarray,
    b: np.ndarray,
    lum: np.ndarray,
) -> np.ndarray:
    """Dark pixels typical of hair — both neutral-dark and warm-dark tones.

    Dark neutral (black / grey hair):
        lum < 0.28 and low inter-channel spread.

    Warm-dark (brown / auburn / dark-blonde hair):
        lum < 0.28, R > B (warm), R−B < 0.12 and R−G < 0.08.
        Catches brown tones that are too warm for the low-sat gate but
        still clearly darker than typical skin luminance.

    Dark threshold extended from 0.20 → 0.28 to capture more of the
    hair mid-shadow region that was previously missed.
    """
    # Neutral / grey dark hair
    dark_neutral = (lum < 0.28) & ((np.abs(r - g) + np.abs(g - b)) < 0.15)

    # Warm-dark: brown / auburn hair adjacent to skin
    warm_dark = (
        (lum < 0.28)
        & (r > b)
        & ((r - b) < 0.12)
        & ((r - g) < 0.08)
    )

    raw = (dark_neutral | warm_dark).astype(np.float32)
    return _smooth_mask(raw)


def _smooth_mask(mask: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Gaussian-smooth a binary float mask to create soft transitions."""
    from scipy.ndimage import gaussian_filter  # type: ignore[import]
    return gaussian_filter(mask, sigma=sigma).astype(np.float32)


# ---------------------------------------------------------------------------
# Scene-aware adjustments
# ---------------------------------------------------------------------------

def _apply_scene_adjustments(
    image: np.ndarray,
    scene_map: SceneMap,
) -> np.ndarray:
    """Apply region-specific adjustments to improve gradeability.

    Rules:
    - Sky: subtle gradient smoothing to reduce banding.
    - Skin: no micro-contrast; protect from aggressive sharpening.
    - Architecture: preserve local edge structure.
    - Hair: gentle shadow open-up.
    - Foliage / fabric: leave mostly untouched.
    """
    img = image.astype(np.float32)

    # --- Sky: smooth gentle to suppress banding ---
    if scene_map.sky.max() > 0.1:
        from scipy.ndimage import gaussian_filter  # type: ignore[import]
        sky_w = scene_map.sky[..., np.newaxis]
        smooth = np.stack(
            [gaussian_filter(img[..., c], sigma=1.5) for c in range(3)],
            axis=-1,
        )
        img = img * (1 - sky_w * 0.4) + smooth * (sky_w * 0.4)

    # --- Skin: ensure no micro-contrast over-enhancement ---
    if scene_map.skin.max() > 0.1:
        from scipy.ndimage import gaussian_filter  # type: ignore[import]
        skin_w = scene_map.skin[..., np.newaxis]
        soft   = np.stack(
            [gaussian_filter(img[..., c], sigma=0.8) for c in range(3)],
            axis=-1,
        )
        img = img * (1 - skin_w * 0.2) + soft * (skin_w * 0.2)

    # --- Hair: gently lift near-black detail ---
    if scene_map.hair.max() > 0.1:
        hair_w = scene_map.hair[..., np.newaxis]
        lifted = img + 0.03 * hair_w * (1.0 - img)   # additive lift, fades at white
        img    = clamp01(lifted)

    return img.astype(np.float32)
