"""Stage 4 — Pseudo-RAW Master Reconstruction.

Transforms a scene-referred linear float32 image into a pseudo-RAW
editable master that behaves like an unrendered capture in Lightroom,
Photoshop, or Capture One.

Design targets
──────────────
• Noticeably flatter than source JPEG — Exposure ±2 moves without clipping
• Highlight flexibility: quadratic ease-out shoulder from 0.74 (global) /
  0.68 (skin) — wider recovery zone than a single-point shoulder
• Shadow editability: true-black floor below Y=0.04 protected; lift
  concentrated in Y=0.05–0.34, not the extreme toe
• White balance flexibility: luminance-weighted grey-world, scene-map protected
• Chroma safety near clipping: targets only genuinely near-clip pixels
• Portrait stability: skin warmth, hair depth, micro-contrast all preserved
• No halo risk: all operations are global or smooth-weighted

All operations are in linear-light float32.
All luminance calculations use Rec. 709 coefficients.
"""

from __future__ import annotations

import logging

import numpy as np

from ele.types import DegradationReport, SceneMap

logger = logging.getLogger(__name__)


# ── Tuning parameters ──────────────────────────────────────────────────────────
# Central constants — adjust these to retune the master without touching
# function bodies.  Each section corresponds to one pipeline stage.

# Highlight shoulder — where luminance compression begins
_HL_GLOBAL_THRESHOLD = 0.74   # was 0.78 — earlier start, wider recovery range
_HL_SKIN_THRESHOLD   = 0.68   # was 0.70 — skin specular compresses sooner
_HL_CURVE_POWER      = 2      # was 3 (cubic→quadratic) — smoother knee, less chalky

# Shadow expansion — redesigned for smoother, deeper, more natural shadow tonality
_SH_LIFT                  = 0.025   # additive lift at peak; slightly higher for deeper recovery
BLACK_ANCHOR              = 0.025   # Y below this: true black floor, no lift ever applied
SHADOW_LIFT_START         = 0.030   # shadow lift ease-in begins; exposes more deep-shadow range
SHADOW_LIFT_PEAK_LOW      = 0.12    # full lift zone ends here (used by _apply_shadow_expansion)
SHADOW_LIFT_PEAK          = 0.08    # rebalance curve strong-recovery ceiling (used by _rebalance_shadow_curve)
SHADOW_LIFT_FADE          = 0.30    # lift and rebalance taper to zero by here
SHADOW_CONTRAST_SMOOTHING = 0.28    # tonal compression strength inside shadow zone (0 = off)

# Mid-contrast flattening
_CF_STRENGTH         = 0.42   # was 0.30 — ~40% more midtone headroom

# Near-clip saturation safety
_SAT_GLOBAL_THRESHOLD = 0.78  # was 0.75 — targets only very-near-clip pixels
_SAT_GLOBAL_AMOUNT    = 0.08  # was 0.12 — lighter desaturation, more colour
_WARM_SAT_STRENGTH    = 0.07  # was 0.10 — relaxed warm-hue safety

# Log-luminance tonal reconstruction (step 3 — replaces linear shadow expansion)
_LOG_SHADOW_THRESHOLD  = -3.0   # log₂ ceiling of the shadow zone  (Y ≈ 0.125)
_LOG_SHADOW_LIFT       = 0.45   # fraction of gap to close per pass (0 = no lift)
_LOG_HL_THRESHOLD      = -0.5   # log₂ floor of the highlight zone  (Y ≈ 0.707)
_LOG_HL_COMPRESS       = 0.30   # fraction of excess to compress (0 = no compression)

# Local tone compression (step 3f)
LOCAL_TONE_STRENGTH        = 0.12   # base local-contrast compression (0 = off, 1 = full flatten)
LOCAL_TONE_RADIUS          = 50     # base-layer blur radius in pixels (large → no halos)
SHADOW_LOCAL_BOOST         = 0.10   # extra compression in shadow zone  (Y < 0.20)
HIGHLIGHT_LOCAL_BOOST      = 0.08   # extra compression in highlight zone (Y > 0.55)
DETAIL_PROTECTION_STRENGTH = 0.68   # gradient-based texture protection (0 = none, 1 = full)

# Chroma expansion — color headroom expansion in Lab space (step 3e)
_CE_CHROMA_THRESHOLD  = 40.0  # max C* below which expansion is applied
_CE_LIFT_STRENGTH     = 0.18  # scale: expansion = 1 + 0.18 × (1 − L/100)
_CE_HL_THRESHOLD      = 82.0  # L* above which chroma is progressively limited
_CE_HL_FADE_SPAN      = 40.0  # denominator for highlight chroma rolloff curve
_CE_HL_MIN_FACTOR     = 0.65  # minimum chroma scale factor in brightest highlights
_CE_SKIN_REDUCTION    = 0.50  # fraction by which chroma expansion is reduced in skin


# ── Public API ────────────────────────────────────────────────────────────────

def reconstruct_pseudo_raw(
    image: np.ndarray,
    report: DegradationReport,
    scene_map: SceneMap,
) -> np.ndarray:
    """Convert a scene-referred image into a pseudo-RAW editable master.

    Pipeline order
    ──────────────
    1.  Luminance-weighted partial WB neutralisation (scene-map aware)
    1b. Portrait skin warmth restoration after WB
    2.  Polynomial mid-contrast flattening (strength=_CF_STRENGTH, extended zone)
    3.  Log-luminance tonal reconstruction — shadow lift + highlight pre-compression
        in log₂ exposure space; EV-uniform lift avoids linear patchiness;
        black anchor (BLACK_ANCHOR) protected; replaces prior linear shadow block.
    3e. Lab-space chroma expansion — low-mid saturation lift, highlight limit, skin protection
    3f. Local tone compression — large-radius base/detail decomposition; scene-aware per-pixel
        compression of local contrast; detail/texture protected via gradient gate.
    4.  Highlight shoulder (quadratic ease-out from _HL_GLOBAL_THRESHOLD=0.74)
    4b. Skin-specific earlier highlight rolloff (from _HL_SKIN_THRESHOLD=0.68)
    5.  Near-clip saturation safety (relaxed — only genuinely near-clip pixels)
    5b. Warm-hue / skin-orange near-clip safety (relaxed strength)
    6.  Skin shadow chroma stabilisation (prevent cyan/green drift)
    7.  Skin micro-contrast reduction (preserve lash/pore structure)
    8.  Hair depth protection (counteract shadow lift in dark hair)
    9.  Scene-aware micro-corrections (sky, foliage, residual skin warmth)

    Args:
        image:     H×W×3 float32, linear RGB, values in [0, 1].
        report:    DegradationReport from Stage 1.
        scene_map: SceneMap from Stage 3.

    Returns:
        H×W×3 float32, pseudo-RAW master, values in [0, 1].
    """
    img = image.astype(np.float32)

    # 1. Partial WB — luminance-weighted, loosen baked colour casts
    img = _apply_partial_wb_neutralization(img, scene_map, base_strength=0.38)
    # 1b. Restore skin warmth lost during WB neutralisation
    img = _protect_skin_warmth(img, scene_map)

    # 2. Contrast flattening — wider, ~40% more flat than previous version
    Y   = _compute_luminance(img)
    Y   = _apply_mid_contrast_flattening(Y, strength=_CF_STRENGTH)
    img = _recombine_luma_chroma(img, Y)

    # 3. Log-luminance tonal reconstruction
    #    Shadow lift and highlight pre-compression are applied in log₂ exposure
    #    space so each stop of recovery is perceptually uniform — avoids the
    #    patchiness that linear additive lifting introduces in mid-shadows.
    logger.info("Log-luminance reconstruction applied")
    Y   = _compute_luminance(img)
    Y   = _apply_log_tonal_reconstruction(Y)
    img = _recombine_luma_chroma(img, Y)

    # 3e. Color headroom expansion — Lab-space chroma lift, skin-protected
    img = _apply_color_headroom_expansion(img, scene_map)

    # 3f. Local tone compression — scene-aware, detail-protected
    img = _apply_local_tone_compression(img, scene_map)

    # 4. Highlight shoulder — quadratic ease-out from _HL_GLOBAL_THRESHOLD
    Y   = _compute_luminance(img)
    Y   = _apply_highlight_shoulder(Y)
    img = _recombine_luma_chroma(img, Y)
    # 4b. Earlier highlight rolloff in skin regions
    img = _apply_skin_highlight_rolloff(img, scene_map)

    # 5. Near-clip saturation safety — relaxed vs previous version
    img = _apply_near_clip_saturation_safety(img)
    # 5b. Warm-hue / orange skin near-clip safety
    img = _apply_warm_hue_safety(img, scene_map)

    # 6. Prevent cyan/green/magenta drift in dark skin shadow zones
    img = _stabilize_skin_shadow_chroma(img, scene_map)

    # 7. Smooth micro-contrast in mid-lum skin (preserve lash/pore edges)
    img = _reduce_skin_microcontrast(img, scene_map)

    # 8. Counteract shadow lift in hair to restore intentional black depth
    img = _protect_hair_depth(img, scene_map)

    # 9. Scene micro-corrections (sky chroma, foliage green, residual skin warmth)
    img = _apply_scene_protection(img, scene_map)

    return np.clip(img, 0.0, 1.0).astype(np.float32)


# ── Portrait-specific helpers ──────────────────────────────────────────────────

def _protect_skin_warmth(
    image: np.ndarray,
    scene_map: SceneMap,
    strength: float = 0.25,
) -> np.ndarray:
    """Restore warmth bias in skin tones after WB neutralisation.

    Grey-world WB may cool skin (reduce R/B ratio).  A small warm nudge
    (+R, −B) is added proportional to the skin mask and a mid-luminance
    gate [0.12, 0.70].  The per-pixel boost scales with the *warmth
    deficit* (how far R−B is below the expected skin floor of 0.05), so
    pixels that are already warm enough receive little or no correction.
    """
    if scene_map.skin.max() < 0.04:
        return image

    img = image.copy()
    skin_w = scene_map.skin          # H×W, [0, 1]
    Y      = _compute_luminance(img)

    # Mid-luminance gate: peaks around Y = 0.30–0.50, fades at 0.12 and 0.70
    lum_gate = (
        np.clip((Y - 0.12) / 0.12, 0.0, 1.0)
        * np.clip((0.70 - Y) / 0.25, 0.0, 1.0)
    )

    # Warmth deficit: 0 when R−B ≥ 0.05 (already warm), 1 when R−B ≤ 0
    r, b         = img[..., 0], img[..., 2]
    deficit      = np.clip(0.05 - (r - b), 0.0, 0.05) / 0.05  # H×W

    weight = skin_w * lum_gate * deficit * strength   # H×W

    warm         = np.zeros_like(img)
    warm[..., 0] =  0.018    # +R
    warm[..., 2] = -0.010    # −B

    img = img + weight[..., np.newaxis] * warm
    return img.clip(0.0, 1.0).astype(np.float32)


def _apply_skin_highlight_rolloff(
    image: np.ndarray,
    scene_map: SceneMap,
    threshold: float = _HL_SKIN_THRESHOLD,
) -> np.ndarray:
    """Apply an earlier highlight shoulder to bright skin pixels.

    The global shoulder starts at _HL_GLOBAL_THRESHOLD.  Skin benefits from
    an even earlier rolloff (_HL_SKIN_THRESHOLD=0.68) so that forehead,
    cheek, and shoulder specular retains recovery headroom.  Uses the same
    quadratic ease-out as the global shoulder for consistency.

    Only pixels above *threshold* in skin-masked regions are affected.
    The correction fades smoothly with the skin mask, so non-skin areas
    are entirely unchanged.
    """
    if scene_map.skin.max() < 0.04:
        return image

    img    = image.copy()
    skin_w = scene_map.skin           # H×W
    Y      = _compute_luminance(img)

    # Quadratic ease-out shoulder starting at threshold
    span      = 1.0 - threshold + 1e-8
    x         = np.clip((Y - threshold) / span, 0.0, 1.0)
    shoulder  = 1.0 - (1.0 - x) ** _HL_CURVE_POWER
    Y_new     = threshold + (1.0 - threshold) * shoulder

    # Blend weight: skin mask × how far into the highlight zone
    in_zone = np.clip((Y - threshold) / (1.0 - threshold + 1e-6), 0.0, 1.0)
    blend   = skin_w * in_zone       # H×W

    # Scale channels so luminance matches Y + blend*(Y_new−Y)
    Y_target = Y + blend * (Y_new - Y)
    scale    = np.where(Y > 1e-4, Y_target / (Y + 1e-7), 1.0)
    scale    = np.clip(scale, 0.0, 3.0)[..., np.newaxis]

    return (img * scale).clip(0.0, 1.0).astype(np.float32)


def _stabilize_skin_shadow_chroma(
    image: np.ndarray,
    scene_map: SceneMap,
    strength: float = 0.18,
) -> np.ndarray:
    """Prevent cyan/green/magenta drift in dark skin shadow regions.

    Shadow expansion shifts near-black pixels uniformly and can introduce
    subtle hue shifts in dark skin (e.g. cyan-drift in deep brown skin).
    A gentle blend toward luminance-matched grey inside the shadow zone
    [0.06, 0.28] stabilises the chromaticity without darkening pixels.
    """
    if scene_map.skin.max() < 0.04:
        return image

    img    = image.copy()
    skin_w = scene_map.skin          # H×W
    Y      = _compute_luminance(img)

    # Shadow zone gate: peaks around Y ≈ 0.12, fades below 0.06 and above 0.28
    shadow_gate = (
        np.clip((Y - 0.06) / 0.06, 0.0, 1.0)
        * np.clip((0.28 - Y) / 0.10, 0.0, 1.0)
    )

    grey  = Y[..., np.newaxis]
    blend = (skin_w * shadow_gate * strength)[..., np.newaxis]

    img = img * (1.0 - blend) + grey * blend
    return img.clip(0.0, 1.0).astype(np.float32)


def _reduce_skin_microcontrast(
    image: np.ndarray,
    scene_map: SceneMap,
    sigma: float = 0.6,
    blend: float = 0.18,
) -> np.ndarray:
    """Smooth high-frequency micro-contrast in mid-luminance skin.

    JPEG artefacts and prior restoration stages can leave micro-contrast
    in skin that complicates grading.  A narrow Gaussian blend (sigma ≤ 1)
    in the mid-lum skin zone [0.10, 0.65] reduces this without blurring
    edges.  The small sigma deliberately preserves lash/brow detail and
    pore structure so the editor retains full control over softening.

    Lash and eye structure is protected because:
      • The luminance gate excludes very dark pixels (Y < 0.10).
      • sigma = 0.6 blurs only sub-pixel noise, not structural edges.
    """
    if scene_map.skin.max() < 0.04:
        return image

    from scipy.ndimage import gaussian_filter  # type: ignore[import]

    img    = image.copy()
    skin_w = scene_map.skin          # H×W
    Y      = _compute_luminance(img)

    # Mid-luminance gate: excludes hair/lashes (dark) and specular (bright)
    mid_gate = (
        np.clip((Y - 0.10) / 0.10, 0.0, 1.0)
        * np.clip((0.65 - Y) / 0.20, 0.0, 1.0)
    )

    eff_blend = (skin_w * mid_gate * blend)[..., np.newaxis]   # H×W×1

    soft = np.stack(
        [gaussian_filter(img[..., c], sigma=sigma) for c in range(3)],
        axis=-1,
    )
    img = img * (1.0 - eff_blend) + soft * eff_blend
    return img.clip(0.0, 1.0).astype(np.float32)


def _apply_warm_hue_safety(
    image: np.ndarray,
    scene_map: SceneMap,
    threshold: float = 0.72,
    strength: float = _WARM_SAT_STRENGTH,
) -> np.ndarray:
    """Prevent warm/orange skin highlights from clipping to unrecoverable hue.

    Near specular highlights, skin shifts from warm-orange to yellow-green
    as the blue channel clips first.  This is almost impossible to fix in
    an editor.  A targeted desaturation blend — narrower than the global
    near-clip safety — is applied specifically to warm-hue, near-clip
    pixels in skin-masked regions.

    Warm-hue gate:  R > G × 1.05  and  R > B × 1.25
    Combined with skin mask and near-clip luminance proximity.
    """
    if scene_map.skin.max() < 0.04:
        return image

    img    = image.copy()
    skin_w = scene_map.skin          # H×W
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    Y      = _compute_luminance(img)
    max_ch = img.max(axis=2)         # H×W

    # Warm/orange hue gate (skin specular zone)
    warm_hue  = ((r > g * 1.05) & (r > b * 1.25)).astype(np.float32)

    # Near-clip proximity
    near_clip = np.clip(
        (max_ch - threshold) / (1.0 - threshold + 1e-6), 0.0, 1.0
    )

    weight = (skin_w * warm_hue * near_clip * strength)[..., np.newaxis]

    grey = Y[..., np.newaxis]
    img  = img * (1.0 - weight) + grey * weight
    return img.clip(0.0, 1.0).astype(np.float32)


def _protect_hair_depth(
    image: np.ndarray,
    scene_map: SceneMap,
    strength: float = 0.022,
) -> np.ndarray:
    """Counteract shadow lift in hair regions to preserve black depth.

    Shadow expansion raises all near-blacks including hair, which can make
    dark hair look muddy rather than intentionally deep.  A selective
    subtractive offset in hair-masked shadow pixels (Y ≤ 0.30) restores
    the tonal depth that an editor expects from an unprocessed capture.

    The pullback fades smoothly to zero at Y = 0.30 so that midtone hair
    texture (brown, auburn) is unaffected.
    """
    if scene_map.hair.max() < 0.04:
        return image

    img    = image.copy()
    hair_w = scene_map.hair          # H×W
    Y      = _compute_luminance(img)

    # Shadow gate: full strength at Y = 0, zero by Y = 0.30
    depth_gate = np.clip(1.0 - Y / 0.30, 0.0, 1.0)

    pullback = (hair_w * depth_gate * strength)[..., np.newaxis]
    img      = img - pullback
    return img.clip(0.0, 1.0).astype(np.float32)


# ── Luminance / chroma utilities ──────────────────────────────────────────────

def _compute_luminance(image: np.ndarray) -> np.ndarray:
    """Return per-pixel Rec. 709 luminance (H×W float32).

    Coefficients are correct for linear-light RGB.
    """
    return (
        0.2126 * image[..., 0]
        + 0.7152 * image[..., 1]
        + 0.0722 * image[..., 2]
    ).astype(np.float32)


def _recombine_luma_chroma(
    original: np.ndarray,
    new_Y: np.ndarray,
) -> np.ndarray:
    """Scale all channels so that luminance matches *new_Y*, preserving hue.

    Multiplies each pixel by new_Y / old_Y, keeping per-channel ratios
    (and therefore chromaticity) intact.  Scale is clamped to [0, 3] to
    avoid extreme amplification in near-black regions.
    """
    old_Y = _compute_luminance(original)
    scale = np.where(old_Y > 1e-4, new_Y / (old_Y + 1e-7), 1.0)
    scale = np.clip(scale, 0.0, 3.0)[..., np.newaxis]   # H×W×1
    return (original * scale).clip(0.0, None).astype(np.float32)


# ── Stage helpers ─────────────────────────────────────────────────────────────

def _apply_partial_wb_neutralization(
    image: np.ndarray,
    scene_map: SceneMap,
    base_strength: float = 0.38,
) -> np.ndarray:
    """Partially neutralise grey-world white balance, luminance- and scene-map aware.

    Strategy
    ────────
    • Compute per-channel grey-world gain (overall mean / channel mean).
    • Build a per-pixel effective strength that is stronger in highlights
      and weaker in midtones / shadows — preserving WB flexibility where
      the editor's WB tool is least sensitive.
    • Pull back toward original in skin regions (preserve natural warmth).
    • Pull back toward original in foliage regions (protect green richness).

    Luminance weighting ramp
    ────────────────────────
    effective_strength = base_strength × (0.5 + 0.5 × clamp(Y / 0.6))
    → 50 % of base in deep shadows, 100 % by Y = 0.6 and above.
    """
    mean   = image.mean(axis=(0, 1)).clip(1e-5, None)   # (3,) channel means
    gains  = mean.mean() / mean                          # (3,) per-channel WB gains

    neutralised = (image * gains[np.newaxis, np.newaxis, :]).clip(0.0, 1.0)

    # Luminance-weighted strength: less aggressive in shadows/midtones
    Y              = _compute_luminance(image)
    lum_weight     = 0.5 + 0.5 * np.clip(Y / 0.6, 0.0, 1.0)     # H×W ∈ [0.5, 1.0]
    eff_strength   = base_strength * lum_weight[..., np.newaxis]  # H×W×1

    result = image + eff_strength * (neutralised - image)

    # Scene-map pullback — reduce WB correction in regions with intentional colour
    skin_w    = scene_map.skin[..., np.newaxis]    * 0.70   # 70 % pullback in skin
    foliage_w = scene_map.foliage[..., np.newaxis] * 0.45   # 45 % pullback in foliage

    result = result + skin_w    * (image - result)
    result = result + foliage_w * (image - result)

    return result.clip(0.0, 1.0).astype(np.float32)


def _apply_mid_contrast_flattening(
    Y: np.ndarray,
    strength: float = _CF_STRENGTH,
) -> np.ndarray:
    """Reduce baked JPEG contrast in the midtone and upper-mid zone.

    Method
    ──────
    Target shape (full-strength):  Y_flat = Y × (0.9 + 0.1 × Y)

    This polynomial:
      • Leaves black (Y = 0) and white (Y = 1) unchanged.
      • Reduces midtone luminance by up to ~10 %, centred around Y ≈ 0.18–0.5.
      • Never brightens — Y_flat ≤ Y for all Y ∈ [0, 1].

    The blend weight confines the effect to the midtone zone:
      • Fades to zero below Y ≈ 0.05  (preserves black point)
      • Peaks from Y ≈ 0.05 to 0.68   (extended midtone zone, includes upper mids)
      • Fades to zero above Y ≈ 0.80  (highlight shoulder takes over)
    """
    Y_flat = Y * (0.9 + 0.1 * Y)   # gentle polynomial reduction

    # Zone weights: fade in from shadows, fade out before highlights
    # Upper fade now extends into the upper-mid zone (fades over [0.68, 0.80])
    w_shadow = np.clip(Y / 0.05, 0.0, 1.0)              # 0 → 1 over [0, 0.05]
    w_high   = np.clip((0.80 - Y) / 0.12, 0.0, 1.0)     # 1 → 0 over [0.68, 0.80]
    weight   = w_shadow * w_high

    return (Y + weight * strength * (Y_flat - Y)).clip(0.0, None).astype(np.float32)


# ── Log-luminance tonal reconstruction ────────────────────────────────────────

def _linear_to_log_luma(Y: np.ndarray) -> np.ndarray:
    """Convert linear luminance to log₂ exposure space.

    A small epsilon (1e-6) is added before taking log so that true-black
    pixels (Y = 0) map to a finite value (≈ −19.9 EV) rather than −∞.
    This keeps arithmetic well-defined without needing a separate mask.
    """
    return np.log2(Y + 1e-6).astype(np.float32)


def _log_to_linear_luma(log_Y: np.ndarray) -> np.ndarray:
    """Convert log₂ exposure space back to linear luminance.

    Returns 2^log_Y.  Values are not clamped here; the caller is responsible
    for clipping (the pipeline-level np.clip at the end of reconstruct_pseudo_raw
    handles this).
    """
    return np.exp2(log_Y).astype(np.float32)


def _apply_log_tonal_reconstruction(Y: np.ndarray) -> np.ndarray:
    """Lift shadows and pre-compress highlights in log₂ exposure space.

    Working in log domain means each unit of adjustment is one EV (stop),
    so the correction is perceptually uniform across the tonal range.  This
    avoids the patchiness that linear additive lifting introduces: in linear
    space a fixed additive lift affects mid-dark tones far more than very
    dark tones (relative to their value), while in log space the same EV
    lift is proportional everywhere.

    Shadow lift
    ───────────
    Pixels below _LOG_SHADOW_THRESHOLD (log₂ ≈ −3, Y ≈ 0.125) are pulled
    toward that ceiling by _LOG_SHADOW_LIFT × gap.

    The gate smoothly ramps from 0 at the BLACK_ANCHOR floor to 1 just
    below the threshold, so the true-black zone (Y < BLACK_ANCHOR ≈ 0.025,
    log₂ ≈ −5.3) receives no lift at all.

    Corrected sign vs the naive formula
    ────────────────────────────────────
    The shadow lift must INCREASE log_Y (move toward 0 from negative).
    The correct increment is:

        Δ = +lift × (threshold − log_Y)   ← positive, since log_Y < threshold

    Highlight pre-compression
    ──────────────────────────
    Pixels above _LOG_HL_THRESHOLD (log₂ ≈ −0.5, Y ≈ 0.707) are pulled
    back toward that floor by _LOG_HL_COMPRESS × excess.  This adds
    headroom before the quadratic highlight shoulder (step 4) shapes the
    final curve, giving the editor more latitude with the Highlights slider.
    """
    log_Y = _linear_to_log_luma(Y)

    log_anchor = float(np.log2(BLACK_ANCHOR + 1e-6))   # ≈ −5.32 EV

    # ── Shadow lift ───────────────────────────────────────────────────────────
    shadow_mask = (log_Y < _LOG_SHADOW_THRESHOLD) & (log_Y > log_anchor)
    if shadow_mask.any():
        gap  = _LOG_SHADOW_THRESHOLD - log_Y[shadow_mask]   # positive (stops to threshold)

        # Gate: 0 at the black anchor, 1 at the shadow threshold
        span = _LOG_SHADOW_THRESHOLD - log_anchor + 1e-6
        gate = np.clip((log_Y[shadow_mask] - log_anchor) / span, 0.0, 1.0)

        log_Y[shadow_mask] += _LOG_SHADOW_LIFT * gap * gate

    # ── Highlight pre-compression ─────────────────────────────────────────────
    hl_mask = log_Y > _LOG_HL_THRESHOLD
    if hl_mask.any():
        excess = log_Y[hl_mask] - _LOG_HL_THRESHOLD    # positive (stops above floor)
        log_Y[hl_mask] -= _LOG_HL_COMPRESS * excess

    return _log_to_linear_luma(log_Y)


def _apply_shadow_expansion(
    Y: np.ndarray,
    lift: float = _SH_LIFT,
    protect: float = BLACK_ANCHOR,
    lift_start: float = SHADOW_LIFT_START,
    peak: float = SHADOW_LIFT_PEAK_LOW,
    rolloff_end: float = SHADOW_LIFT_FADE,
) -> np.ndarray:
    """Increase shadow editability with a four-zone additive lift.

    Zone 1 — protected black floor  [0, BLACK_ANCHOR=0.020]:
        No lift applied.  True blacks are preserved.  The anchor sits lower
        than before (0.020 vs prior 0.040) so a larger slice of deep shadow
        is eligible for lift in Zone 2.

    Zone 2 — ease-in ramp  [BLACK_ANCHOR, SHADOW_LIFT_START=0.030]:
        Lift rises smoothly from 0 to full using a cosine² ramp.  The
        narrow span (0.020–0.030) means recovery begins almost immediately
        above the black floor — opening up darkest recoverable detail.

    Zone 3 — full lift  [SHADOW_LIFT_START, SHADOW_LIFT_PEAK_LOW=0.12]:
        Full lift applied.  Peak zone shifted lower vs prior 0.20, so the
        strongest recovery targets deeper shadows rather than mid-darks.
        This prevents brighter-shadow tones from popping relative to their
        darker neighbours.

    Zone 4 — cosine² fade  [SHADOW_LIFT_PEAK_LOW, SHADOW_LIFT_FADE=0.26]:
        Lift tapers smoothly from full → 0.  Upper shadows (0.18–0.26) now
        receive noticeably less lift than before (rolloff_end 0.34 → 0.26),
        avoiding the "bright patch inside shadow" look.  Tones above 0.26
        are completely unchanged.
    """
    # Ease in: 0 → 1 over [protect, lift_start]
    span_in = max(lift_start - protect, 1e-6)
    t_in    = np.clip((Y - protect) / span_in, 0.0, 1.0)
    w_in    = np.cos((1.0 - t_in) * (np.pi / 2.0)) ** 2   # 0 at protect, 1 at lift_start

    # Ease out: 1 → 0 over [peak, rolloff_end]
    span_out = max(rolloff_end - peak, 1e-6)
    t_out    = np.clip((Y - peak) / span_out, 0.0, 1.0)
    w_out    = np.cos(t_out * (np.pi / 2.0)) ** 2          # 1 at peak, 0 at rolloff_end

    weight = np.where(
        Y <= protect,    0.0,
        np.where(Y <= lift_start, w_in,
        np.where(Y <= peak,       1.0, w_out))
    )

    return (Y + lift * weight).astype(np.float32)


def _smooth_shadow_tonal_contrast(
    Y: np.ndarray,
    strength: float = SHADOW_CONTRAST_SMOOTHING,
    fade_end: float = 0.30,
) -> np.ndarray:
    """Reduce tonal separation inside the shadow zone after expansion.

    After shadow expansion, tones in the shadow zone can still carry
    uneven spacing — brighter sub-shadows appear to "pop" relative to
    deeper ones, adding a pseudo-HDR quality.  This step compresses the
    shadow tonal range around a pivot so that transitions feel gradual
    and continuous.  It is a pure luminance tone-curve operation; no
    spatial blurring is applied.

    Method
    ──────
    A pivot is placed at SHADOW_LIFT_PEAK_LOW × 0.55 (≈ 0.066 by
    default) — the lower half of the full-lift zone.

    For each pixel:
        delta = Y − pivot
        Y′ = Y − gate × strength × 0.5 × delta

    Pixels above the pivot are nudged slightly downward (darker).
    Pixels below the pivot are nudged slightly upward (lighter).
    Net effect: reduced spread across the shadow zone, more even ramp.

    Gates
    ─────
    • Spatial gate (raw_gate): 1 near black, fades to 0 at fade_end=0.30.
      Uses a 1.5-power falloff for a smooth non-linear taper.
    • Black-anchor gate (above_anchor): ensures no effect below
      BLACK_ANCHOR (true blacks untouched), rises to 1 at SHADOW_LIFT_START.
    """
    pivot = SHADOW_LIFT_PEAK_LOW * 0.55  # ≈ 0.066 with default constants

    # Spatial gate: strongest near black, zero at fade_end
    raw_gate = np.clip(1.0 - Y / fade_end, 0.0, 1.0) ** 1.5

    # Black-anchor protection: zero below BLACK_ANCHOR, full at SHADOW_LIFT_START
    anchor_span = max(SHADOW_LIFT_START - BLACK_ANCHOR, 1e-6)
    above_anchor = np.clip((Y - BLACK_ANCHOR) / anchor_span, 0.0, 1.0)
    gate = raw_gate * above_anchor

    # Compress tonal range: pull tones toward pivot
    delta = Y - pivot
    Y_smooth = Y - gate * strength * 0.5 * delta

    return np.clip(Y_smooth, 0.0, None).astype(np.float32)


def _rebalance_shadow_curve(x: np.ndarray) -> np.ndarray:
    """Rebalance shadow lift so deeper shadows open more evenly.

    Three zones below SHADOW_LIFT_FADE:

    • Protected  [0, BLACK_ANCHOR):         identity — true black untouched.
    • Strong recovery  [SHADOW_LIFT_START, SHADOW_LIFT_PEAK):
        Additive pull toward SHADOW_LIFT_PEAK ceiling scaled by 0.55 — the
        deepest recoverable darks receive the largest absolute lift.
    • Gentle recovery  [SHADOW_LIFT_PEAK, SHADOW_LIFT_FADE):
        Fractional lift that tapers linearly to zero at SHADOW_LIFT_FADE,
        preventing brighter sub-shadows from popping.

    x is per-pixel Rec. 709 luminance in [0, 1].
    """
    black_anchor  = BLACK_ANCHOR       # 0.025
    lift_start    = SHADOW_LIFT_START  # 0.030
    lift_peak_low = SHADOW_LIFT_PEAK   # 0.08
    lift_fade     = SHADOW_LIFT_FADE   # 0.30

    y = x.copy()

    # Zone 1: protect deepest blacks
    mask_anchor = x < black_anchor
    y[mask_anchor] = x[mask_anchor]

    # Zone 2: strong recovery — pull toward lift_peak_low ceiling
    mask_low = (x >= lift_start) & (x < lift_peak_low)
    y[mask_low] = x[mask_low] + (lift_peak_low - x[mask_low]) * 0.55

    # Zone 3: gentle, linearly tapering recovery
    mask_mid = (x >= lift_peak_low) & (x < lift_fade)
    strength = 0.25 * (1.0 - (x[mask_mid] - lift_peak_low) / (lift_fade - lift_peak_low))
    y[mask_mid] = x[mask_mid] + strength * x[mask_mid]

    return y.astype(np.float32)


def _smooth_shadow_tones(luma: np.ndarray) -> np.ndarray:
    """Blend shadow luminance toward a sqrt ramp below Y=0.22.

    Inside the shadow zone (Y < 0.22) a mild blend of the linear value
    with its square root softens the tonal ramp, reducing the harsh
    contrast steps that can appear between adjacent tonal bands after
    lifting.  The square-root component lifts darker tones more than
    brighter ones, which complements the preceding expansion steps.

    Above 0.22 the output is identical to the input.
    This is a tonal operation — no spatial smoothing is applied.
    """
    smoothed = luma.copy()
    mask = luma < 0.22
    smoothed[mask] = luma[mask] * 0.85 + np.sqrt(luma[mask]) * 0.15
    return smoothed.astype(np.float32)


# ── Local tone compression ────────────────────────────────────────────────────

def _compute_local_luma_base(
    luma: np.ndarray,
    radius: int = LOCAL_TONE_RADIUS,
) -> np.ndarray:
    """Compute a smooth regional luminance base via separable box blur.

    Uses scipy's uniform_filter, which runs two 1-D sliding-sum passes
    and is O(N) regardless of radius.  On a 5 K image the full-resolution
    pass completes in under one second.

    The large radius (default 50 px) ensures the base represents broad
    regional tone (sky gradient, skin patch, shadow pool) rather than
    pixel-level detail.  This gap between the blur scale and any real
    edge is what prevents halos: the compressed local-contrast layer can
    never span a hard edge in the way that a narrow-radius "dodge" would.

    Args:
        luma:   H×W float32 luminance.
        radius: Box half-width in pixels.

    Returns:
        H×W float32 smooth base luminance.
    """
    from scipy.ndimage import uniform_filter  # type: ignore[import]

    size = 2 * radius + 1
    return uniform_filter(luma, size=size).astype(np.float32)


def _apply_local_tone_compression(
    image: np.ndarray,
    scene_map: SceneMap,
) -> np.ndarray:
    """Reduce excessive local contrast to produce smoother, more RAW-like tonality.

    Decomposition
    ─────────────
    Every pixel's luminance is split into:

        luma = base + local_contrast

    where *base* is the large-radius smooth approximation (regional tone)
    and *local_contrast* = luma − base captures the fine-to-mid local
    tonal deviation (the part that can look patchy or harsh).

    A per-pixel compression factor *c* ∈ [0, 0.40] is built up from four
    contributions, then the output is:

        luma_new = base + local_contrast × (1 − c)

    When c = 0 the image is unchanged.  When c = 0.40 the local contrast
    is reduced by 40 %.  The global base is never touched — only the
    local deviation is compressed — so the image retains its broad tonal
    structure and does NOT look log-flat or HDR.

    Compression factor build-up
    ────────────────────────────
    (1) Base strength = LOCAL_TONE_STRENGTH (0.12) — uniform everywhere.

    (2) Shadow boost = SHADOW_LOCAL_BOOST × gate(Y, 0 → 0.20).
        Dark regions tend to show more patchiness after shadow lifting;
        compressing their local contrast smooths transitions without
        lifting or darkening the overall shadow level.

    (3) Highlight boost = HIGHLIGHT_LOCAL_BOOST × gate(Y, 0.55 → 0.70).
        Bright regions benefit from softer local contrast for smoother
        sky gradients and specular rolloff.

    (4) Scene-map delta (see _build_scene_compression_delta):
        sky +0.08, skin −0.02, foliage −0.05, architecture −0.08, hair −0.09.

    Texture / detail protection
    ────────────────────────────
    Gradient magnitude (∇Y) is normalised to [0, 1] against the 90th
    percentile.  The compression factor is multiplied by:

        (1 − DETAIL_PROTECTION_STRENGTH × gradient_norm)

    so at a strong edge (gradient_norm = 1) the effective compression is
    ≈ 32 % of the nominal value, and at zero gradient it is 100 %.
    This protects pores, hair strands, eyelashes, foliage veins, and
    architecture edges from being softened.

    Halo prevention
    ───────────────
    Halos arise when a base layer closely tracks a hard edge; compressing
    local contrast then pushes one side toward the edge value and the other
    away.  The LOCAL_TONE_RADIUS (50 px) is large enough that the base at
    any pixel is an average of a 101 × 101 neighbourhood — no individual
    edge can dominate it.  The texture gate adds a second line of defence
    for the few pixels immediately adjacent to very hard edges.

    Channel handling
    ─────────────────
    All compression is computed on luminance only, then applied via
    _recombine_luma_chroma so hue and saturation are preserved.
    """
    luma = _compute_luminance(image)
    base = _compute_local_luma_base(luma)

    local_contrast = luma - base   # H×W; positive = locally bright, negative = locally dark

    # ── Compression map ──────────────────────────────────────────────────────
    c = np.full(luma.shape, LOCAL_TONE_STRENGTH, dtype=np.float32)

    # Shadow boost: stronger compression in dark zones (Y < 0.20)
    shadow_gate = np.clip(1.0 - luma / 0.20, 0.0, 1.0)
    c += SHADOW_LOCAL_BOOST * shadow_gate

    # Highlight boost: stronger compression in bright zones (Y > 0.55)
    hl_gate = np.clip((luma - 0.55) / 0.15, 0.0, 1.0)
    c += HIGHLIGHT_LOCAL_BOOST * hl_gate

    # Scene-aware per-region delta
    c += _build_scene_compression_delta(luma, scene_map)

    # Gradient-based texture protection
    gy = np.abs(np.diff(luma, axis=0, prepend=luma[:1, :]))
    gx = np.abs(np.diff(luma, axis=1, prepend=luma[:, :1]))
    grad = np.sqrt(gy ** 2 + gx ** 2)
    grad_ceil = float(np.percentile(grad, 90)) + 1e-6
    texture = np.clip(grad / grad_ceil, 0.0, 1.0)
    c *= 1.0 - DETAIL_PROTECTION_STRENGTH * texture

    c = np.clip(c, 0.0, 0.40)   # hard ceiling: never flatten more than 40 %

    # ── Reconstruct luminance and scale channels ──────────────────────────────
    luma_new = base + local_contrast * (1.0 - c)
    luma_new = np.clip(luma_new, 0.0, None)

    return _recombine_luma_chroma(image, luma_new)


def _build_scene_compression_delta(
    luma: np.ndarray,
    scene_map: SceneMap,
) -> np.ndarray:
    """Build per-pixel compression strength delta from scene region masks.

    Returns an H×W additive offset to the base compression factor.
    Positive = more compression; negative = less (protect detail/edges).

    Sky       +0.08  — large smooth gradients benefit most from local flattening
    Skin      −0.02  — keep natural pore/micro-texture visible
    Foliage   −0.05  — protect leaf texture and green richness
    Architecture −0.08  — preserve hard edges and surface details strongly
    Hair      −0.09  — protect strand structure and intentional black depth
    """
    delta = np.zeros(luma.shape, dtype=np.float32)

    if scene_map.sky.max() > 0.04:
        delta += scene_map.sky * 0.08

    if scene_map.skin.max() > 0.04:
        delta -= scene_map.skin * 0.02

    if scene_map.foliage.max() > 0.04:
        delta -= scene_map.foliage * 0.05

    if scene_map.architecture.max() > 0.04:
        delta -= scene_map.architecture * 0.08

    if scene_map.hair.max() > 0.04:
        delta -= scene_map.hair * 0.09

    return delta


# ── Lab colour space helpers ───────────────────────────────────────────────────

def _rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """Convert linear RGB (sRGB primaries, D65) → CIE L*a*b*.

    Input:  H×W×3 float32 linear RGB, values in [0, 1].
    Output: H×W×3 float32 — L* in [0, 100], a*/b* nominally in [−128, 127].

    Uses the standard sRGB→XYZ D65 matrix (IEC 61966-2-1) followed by the
    CIE f() function.  No gamma encoding is applied — the input is already
    linear light.
    """
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)

    rgb = np.clip(image, 0.0, None)
    xyz = rgb @ M.T                          # H×W×3, XYZ D65

    # Normalise by D65 white point
    xyz[..., 0] /= 0.95047
    # xyz[..., 1] stays / 1.0
    xyz[..., 2] /= 1.08883

    # CIE f() piecewise function
    eps, kap = 0.008856, 903.3
    f = np.where(xyz > eps, np.cbrt(np.maximum(xyz, 0.0)), (kap * xyz + 16.0) / 116.0)

    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])

    return np.stack([L, a, b], axis=-1).astype(np.float32)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIE L*a*b* → linear RGB (sRGB primaries, D65).

    Input:  H×W×3 float32 Lab.
    Output: H×W×3 float32 linear RGB, unclamped (caller is responsible for
            clamping to maintain ProPhoto pipeline compatibility).
    """
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0

    eps, kap = 0.008856, 903.3
    x = np.where(fx ** 3 > eps, fx ** 3, (116.0 * fx - 16.0) / kap)
    y = np.where(L > kap * eps,  ((L + 16.0) / 116.0) ** 3, L / kap)
    z = np.where(fz ** 3 > eps, fz ** 3, (116.0 * fz - 16.0) / kap)

    # Restore D65 white point
    x *= 0.95047
    # y stays × 1.0
    z *= 1.08883

    xyz = np.stack([x, y, z], axis=-1)

    # XYZ D65 → linear sRGB
    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ], dtype=np.float32)

    return (xyz @ M_inv.T).astype(np.float32)


# ── Chroma expansion helpers ───────────────────────────────────────────────────

def _expand_chroma_lab(
    lab: np.ndarray,
    skin_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Expand color headroom in low-to-mid saturation regions.

    Operates in Lab space.  Only pixels whose chroma C* = √(a²+b²) falls
    below _CE_CHROMA_THRESHOLD (40) are affected, so already-vivid colours
    are left untouched and artificial oversaturation is avoided.

    Lift strength scales with inverse luminance:
        lift = 1 + _CE_LIFT_STRENGTH × (1 − L/100)
    Darker pixels receive more expansion than brighter ones, which opens
    shadow and mid-shadow colour information without touching highlights.

    Skin protection
    ───────────────
    Where skin_mask > 0, the chroma delta is attenuated by
    _CE_SKIN_REDUCTION (50 %).  Skin tones receive half the expansion,
    preventing warmth/saturation from turning unnatural.

    Args:
        lab:       H×W×3 float32 Lab image.
        skin_mask: H×W float32 in [0, 1], or None.  Skin region weight.

    Returns:
        H×W×3 float32 Lab with expanded chroma in targeted zones.
    """
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]

    chroma = np.sqrt(a * a + b * b)          # H×W

    mask = chroma < _CE_CHROMA_THRESHOLD     # H×W bool

    lift = 1.0 + _CE_LIFT_STRENGTH * (1.0 - L / 100.0)   # H×W

    a2 = a.copy()
    b2 = b.copy()
    a2[mask] *= lift[mask]
    b2[mask] *= lift[mask]

    # Skin protection: reduce the applied delta by _CE_SKIN_REDUCTION in skin zones
    if skin_mask is not None and skin_mask.max() > 0.04:
        # reduction ∈ [0.5, 1.0]: 1.0 outside skin, 0.5 at full skin weight
        reduction = 1.0 - skin_mask * _CE_SKIN_REDUCTION  # H×W
        a2 = a + (a2 - a) * reduction
        b2 = b + (b2 - b) * reduction

    lab2 = lab.copy()
    lab2[..., 1] = a2
    lab2[..., 2] = b2

    return lab2


def _limit_highlight_chroma(lab: np.ndarray) -> np.ndarray:
    """Progressively reduce chroma in very bright regions.

    Highlights above L*=_CE_HL_THRESHOLD (82) receive a chroma scale factor
    that decreases linearly toward _CE_HL_MIN_FACTOR (0.65) at L*=122.
    This prevents the chroma expansion step from causing unrecoverable
    hue shifts in specular or blown highlights.

    Works on a copy — the input array is not mutated.
    """
    lab2 = lab.copy()
    L    = lab2[..., 0]

    mask   = L > _CE_HL_THRESHOLD
    factor = 1.0 - (L - _CE_HL_THRESHOLD) / _CE_HL_FADE_SPAN
    factor = np.clip(factor, _CE_HL_MIN_FACTOR, 1.0)

    lab2[..., 1][mask] *= factor[mask]
    lab2[..., 2][mask] *= factor[mask]

    return lab2


def _apply_color_headroom_expansion(
    image: np.ndarray,
    scene_map: SceneMap,
) -> np.ndarray:
    """Integration wrapper: Lab-space chroma expansion with skin and highlight protection.

    Pipeline
    ────────
    1. linear RGB → Lab
    2. _expand_chroma_lab  — lift low-mid chroma, skin-protected
    3. _limit_highlight_chroma — clamp chroma near specular/clip zone
    4. Lab → linear RGB
    5. Clamp to [0, 1] (safe for ProPhoto TIFF export)

    Logged once per call: "Color headroom expansion applied".
    """
    logger.info("Color headroom expansion applied")

    skin_mask = scene_map.skin if scene_map.skin.max() > 0.04 else None

    lab = _rgb_to_lab(image)
    lab = _expand_chroma_lab(lab, skin_mask=skin_mask)
    lab = _limit_highlight_chroma(lab)
    rgb = _lab_to_rgb(lab)

    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _apply_highlight_shoulder(
    Y: np.ndarray,
    threshold: float = _HL_GLOBAL_THRESHOLD,
    curve_power: int = _HL_CURVE_POWER,
) -> np.ndarray:
    """Remap bright luminance through a smooth ease-out shoulder.

    Values below *threshold* are unchanged.
    Values in [threshold, 1] are remapped via the power ease-out curve:

        x = (Y − threshold) / (1 − threshold)
        Y′ = threshold + (1 − threshold) × (1 − (1 − x)^curve_power)

    curve_power controls the knee shape:
      • power=2 (quadratic): slope=2 at knee — gentle, smooth onset
      • power=3 (cubic):     slope=3 at knee — slightly more abrupt
      Quadratic is used by default to avoid chalkiness at the transition.

    Properties
    ──────────
    • Below threshold: identity (no change).
    • Just above threshold: expansion with slope = curve_power — high-lum
      pixels spread into a wider output range, improving tonal separation.
    • Near Y=1.0: slope → 0 — near-clip values cluster just below 1.0,
      creating recovery headroom for the editor's Highlights slider.
    • The earlier threshold (0.74 vs prior 0.78) widens the shoulder zone,
      giving headroom to a larger proportion of the image's highlights.
    """
    span   = 1.0 - threshold + 1e-8
    x      = np.clip((Y - threshold) / span, 0.0, 1.0)

    shoulder   = 1.0 - (1.0 - x) ** curve_power
    Y_shoulder = threshold + (1.0 - threshold) * shoulder

    return np.where(x > 1e-6, Y_shoulder, Y).astype(np.float32)


def _apply_near_clip_saturation_safety(
    image: np.ndarray,
    threshold: float = _SAT_GLOBAL_THRESHOLD,
    amount: float = _SAT_GLOBAL_AMOUNT,
) -> np.ndarray:
    """Reduce saturation brittleness in high-chroma, near-clipped pixels.

    Targets pixels that are simultaneously:
      • Bright (max channel > threshold, approaching clipping)
      • Highly saturated (wide RGB spread)

    The raised threshold (0.78 vs prior 0.75) and lighter amount (0.08 vs
    0.12) mean only genuinely near-clip pixels are affected, preserving
    more colour saturation across the image for the editor to work with.

    Method
    ──────
    Chroma is estimated as (max_channel − min_channel) / max_channel.
    The danger weight combines near-clip proximity and chroma, so only
    brittle colours are affected.
    """
    Y      = _compute_luminance(image)
    max_ch = image.max(axis=2)    # H×W
    min_ch = image.min(axis=2)

    # Normalised RGB chroma (0 = grey, 1 = fully saturated)
    chroma = np.where(max_ch > 1e-5, (max_ch - min_ch) / max_ch, 0.0)

    # Proximity to clipping in [threshold, 1]
    clip_t = np.clip((max_ch - threshold) / (1.0 - threshold + 1e-6), 0.0, 1.0)

    # Danger weight: both conditions must be present
    danger = clip_t * np.clip(chroma * 1.5, 0.0, 1.0)    # H×W

    # Blend toward greyscale (~10–12 % at full danger)
    grey  = Y[..., np.newaxis]
    blend = (danger * amount)[..., np.newaxis]

    return (image * (1.0 - blend) + grey * blend).clip(0.0, 1.0).astype(np.float32)


def _apply_scene_protection(
    image: np.ndarray,
    scene_map: SceneMap,
) -> np.ndarray:
    """Apply region-specific micro-corrections after global transforms.

    Skin
    ────
    The WB and contrast steps may have slightly cooled or flattened skin.
    A small warm-back nudge (boost R channel, suppress B slightly) in
    mid-luminance skin zones preserves believable flesh tones.

    Hair
    ────
    Shadow expansion lifts near-blacks everywhere.  In hair regions this
    can produce muddiness.  A selective pull-back of the shadow lift in
    hair zones keeps blacks intentional and clean.

    Sky
    ───
    Sky highlights benefit from extra near-clip saturation damping to
    maintain smooth blue gradients under exposure adjustments.  A gentle
    luminance-gated desaturation is applied in sky-masked bright regions.

    Foliage
    ───────
    Foliage greens can shift slightly blue-green under grey-world WB
    correction.  A gentle G-channel preservation bias counteracts this,
    keeping greens rich without a visible colour cast.
    """
    img = image.copy()

    # ── Skin: gentle warmth protection ──────────────────────────────────────
    if scene_map.skin.max() > 0.04:
        skin_w = scene_map.skin[..., np.newaxis]    # H×W×1, [0,1]
        Y      = _compute_luminance(img)
        # Only affect mid-luminance skin (not blown-out or very dark)
        mid    = (np.clip((Y - 0.15) / 0.45, 0.0, 1.0)
                  * np.clip((0.75 - Y) / 0.30, 0.0, 1.0))[..., np.newaxis]
        warm        = np.zeros_like(img)
        warm[..., 0] = 0.012    # +R
        warm[..., 2] = -0.008   # −B
        img = img + skin_w * mid * warm

    # ── Hair: counteract shadow lift in very dark pixels ─────────────────────
    if scene_map.hair.max() > 0.04:
        hair_w = scene_map.hair[..., np.newaxis]
        Y      = _compute_luminance(img)
        # Shadow zone: strongest pullback below Y=0.12, zero by Y=0.25
        shadow = np.clip(1.0 - Y / 0.25, 0.0, 1.0)[..., np.newaxis]
        img    = img - hair_w * shadow * 0.012

    # ── Sky: extra near-clip chroma damping for smooth blue gradients ────────
    if scene_map.sky.max() > 0.04:
        sky_w = scene_map.sky[..., np.newaxis]
        Y     = _compute_luminance(img)
        # Only high-luminance sky pixels (upper shoulder zone)
        bright = np.clip((Y - 0.70) / 0.15, 0.0, 1.0)[..., np.newaxis]
        grey   = Y[..., np.newaxis]
        # Gentle push toward grey in bright sky regions — 8 % max
        img    = img - sky_w * bright * 0.08 * (img - grey)

    # ── Foliage: preserve green channel richness ─────────────────────────────
    if scene_map.foliage.max() > 0.04:
        foliage_w = scene_map.foliage[..., np.newaxis]
        Y         = _compute_luminance(img)
        # Mid-luminance only: avoid touching very bright or very dark foliage
        mid       = (np.clip((Y - 0.10) / 0.35, 0.0, 1.0)
                     * np.clip((0.70 - Y) / 0.25, 0.0, 1.0))[..., np.newaxis]
        # Tiny G preservation bias — counteracts grey-world blue-shift in greens
        g_boost        = np.zeros_like(img)
        g_boost[..., 1] = 0.010    # +G
        g_boost[..., 2] = -0.006   # −B
        img = img + foliage_w * mid * g_boost

    return np.clip(img, 0.0, 1.0).astype(np.float32)
