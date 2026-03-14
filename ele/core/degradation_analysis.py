"""Stage 1 — Degradation Analysis.

Analyses an input float32 RGB image and returns a DegradationReport.
All heuristics are lightweight and run on CPU without external ML models.
"""

from __future__ import annotations

import numpy as np

from ele.config import CLIP_LOW_THRESHOLD, CLIP_HIGH_THRESHOLD, CLIP_FRACTION_WARN
from ele.types import DegradationReport
from ele.utils import luminance


def analyse(image: np.ndarray) -> DegradationReport:
    """Analyse degradation in a float32 linear RGB image (H×W×3, values [0,1]).

    Returns:
        DegradationReport with scores in [0, 1].
    """
    notes: list[str] = []

    gray      = luminance(image)
    clipping  = _clipping_score(image, notes)
    compress  = _compression_score(gray, notes)
    ringing   = _estimate_ringing(gray, notes)
    sharpness = _sharpness_score(gray)
    noise     = _noise_score(gray)
    dr        = _dynamic_range_score(image)

    return DegradationReport(
        compression_score=compress,
        clipping_score=clipping,
        sharpness_score=sharpness,
        noise_score=noise,
        dynamic_range_score=dr,
        ringing_score=ringing,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Individual metric estimators
# ---------------------------------------------------------------------------

def _clipping_score(image: np.ndarray, notes: list[str]) -> float:
    """Fraction of pixels that are near-black or near-white."""
    total = image.size / 3  # pixels
    low_clip  = (image < CLIP_LOW_THRESHOLD).any(axis=2).sum() / total
    high_clip = (image > CLIP_HIGH_THRESHOLD).any(axis=2).sum() / total
    score = float(np.clip((low_clip + high_clip) / 2, 0.0, 1.0))

    if high_clip > CLIP_FRACTION_WARN:
        notes.append(f"highlight clipping detected ({high_clip:.1%} of pixels)")
    if low_clip > CLIP_FRACTION_WARN:
        notes.append(f"shadow clipping detected ({low_clip:.1%} of pixels)")

    return score


def _estimate_block_boundary_strength(gray: np.ndarray) -> float:
    """Compare 8×8 grid boundary discontinuities against nearby interior energy.

    Key insight: a genuine JPEG block boundary produces a large luminance jump
    AT the boundary that is disproportionately larger than the gradient 3–4
    pixels away from it.  Real image edges also produce large boundary jumps
    but their nearby interior energy is similarly elevated, so the ratio stays
    close to 1.

    Returns the mean (boundary_diff / nearby_interior_diff) ratio across all
    8-pixel boundaries in both axes.  Values > 1.3 indicate blockiness.
    """
    h, w = gray.shape

    ratios: list[float] = []

    # ── Horizontal block boundaries (column multiples of 8) ──────────────────
    for x in range(8, w - 1, 8):
        # The crossing diff at the boundary
        bd = float(np.abs(gray[:, x] - gray[:, x - 1]).mean())

        # Nearby interior diffs: 3 and 4 pixels away from the boundary
        nd_samples: list[float] = []
        for off in (3, 4):
            if x - off - 1 >= 0:
                nd_samples.append(float(np.abs(gray[:, x - off] - gray[:, x - off - 1]).mean()))
            if x + off < w:
                nd_samples.append(float(np.abs(gray[:, x + off] - gray[:, x + off - 1]).mean()))

        nd = float(np.mean(nd_samples)) + 1e-8 if nd_samples else 1e-8
        ratios.append(bd / nd)

    # ── Vertical block boundaries (row multiples of 8) ───────────────────────
    for y in range(8, h - 1, 8):
        bd = float(np.abs(gray[y, :] - gray[y - 1, :]).mean())

        nd_samples = []
        for off in (3, 4):
            if y - off - 1 >= 0:
                nd_samples.append(float(np.abs(gray[y - off, :] - gray[y - off - 1, :]).mean()))
            if y + off < h:
                nd_samples.append(float(np.abs(gray[y + off, :] - gray[y + off - 1, :]).mean()))

        nd = float(np.mean(nd_samples)) + 1e-8 if nd_samples else 1e-8
        ratios.append(bd / nd)

    return float(np.mean(ratios)) if ratios else 1.0


def _compression_score(gray: np.ndarray, notes: list[str]) -> float:
    """Estimate JPEG block artefacts via boundary-vs-nearby-interior ratio.

    Improvement over the prior implementation:
    - Prior: compared boundary diff to the GLOBAL interior average, which made
      real-content edges inflate the score (false positives on sharp subjects).
    - Now: compares boundary diff to the NEARBY interior diff (3-4 px away),
      which isolates true block jumps from scene edges that naturally taper.

    Mapping: ratio ≥ 1.6 → score = 1.0 (was 1.4 — slightly higher threshold
    because the ratio is now sharper and less diluted by global averages).
    """
    if gray.shape[0] < 16 or gray.shape[1] < 16:
        return 0.0

    ratio = _estimate_block_boundary_strength(gray)

    # Map: ratio 1.0 (no blocking) → 0.0; ratio ≥ 1.6 → 1.0
    score = float(np.clip((ratio - 1.0) / 0.6, 0.0, 1.0))

    if score > 0.35:
        notes.append(
            f"possible JPEG block artefacts (boundary/nearby ratio={ratio:.2f})"
        )

    return score


def _estimate_ringing(gray: np.ndarray, notes: list[str]) -> float:
    """Estimate edge-adjacent oscillation / overshoot (JPEG ringing).

    Method
    ──────
    1. Build a coarse smooth reference (Gaussian σ=3) — this represents what
       the image would look like with no high-frequency artefacts.
    2. Detect strong edges using fine-scale Gaussian difference (σ=0.7).
    3. Expand the edge mask by 4 pixels but exclude the 1-pixel edge core.
       This ring zone is where JPEG ringing actually lives.
    4. Measure the mean deviation |image − coarse_smooth| inside the ring zone.
       Genuine ringing oscillates 0.01–0.05 above/below the smooth reference.
    5. Normalise to [0, 1] against a 0.025 expected ringing magnitude.

    A clean JPEG scores near 0; a heavily compressed or web-resaved image
    can score 0.4–0.8.
    """
    from scipy.ndimage import gaussian_filter, binary_dilation  # type: ignore[import]

    if gray.shape[0] < 16 or gray.shape[1] < 16:
        return 0.0

    # Coarse smooth reference — suppresses all fine structure
    smooth_coarse = gaussian_filter(gray, sigma=3.0)

    # Fine-scale edge detection
    smooth_fine = gaussian_filter(gray, sigma=0.7)
    fine_detail = np.abs(gray - smooth_fine)

    edge_thresh = float(np.percentile(fine_detail, 88))
    if edge_thresh < 1e-5:
        return 0.0  # flat image

    strong_edges = fine_detail > edge_thresh

    # Ring zone: 2–4 px from edges, not on the edges themselves
    near4 = binary_dilation(strong_edges, iterations=4)
    near1 = binary_dilation(strong_edges, iterations=1)
    ring_zone = near4 & ~near1

    if ring_zone.sum() < 50:
        return 0.0

    # Ringing signal = deviation from coarse smooth inside ring zone
    ring_signal = float(np.abs(gray - smooth_coarse)[ring_zone].mean())
    score = float(np.clip(ring_signal / 0.025, 0.0, 1.0))

    if score > 0.35:
        notes.append(f"JPEG ringing detected (edge-adjacent deviation={ring_signal:.4f})")

    return score


def _sharpness_score(gray: np.ndarray) -> float:
    """Estimate sharpness from Laplacian high-frequency energy."""
    lap = (
        -gray[:-2, 1:-1]
        - gray[2:,  1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
        + 4 * gray[1:-1, 1:-1]
    )
    energy = float(np.mean(lap ** 2))
    # Typical sharp images: energy ~0.001–0.01; blurry: <0.0002
    return float(np.clip(energy / 0.005, 0.0, 1.0))


def _noise_score(gray: np.ndarray) -> float:
    """Estimate noise from local variance in smooth regions.

    Uses a 4×4 block variance map and takes the lower percentile,
    which corresponds to flat regions where variance is mostly noise.
    """
    h, w = gray.shape
    block = 4

    gh = (h // block) * block
    gw = (w // block) * block
    blocks = gray[:gh, :gw].reshape(gh // block, block, gw // block, block)
    var_map = blocks.var(axis=(1, 3))

    noise_var = float(np.percentile(var_map, 10))
    return float(np.clip(noise_var / 5e-4, 0.0, 1.0))


def _dynamic_range_score(image: np.ndarray) -> float:
    """Estimate photographic dynamic range from luminance percentile spread.

    This is a heuristic estimate for JPEG sources operating on normalised
    linear values [0, 1].  It does not measure true sensor DR; it measures
    the tonal range present in the image after sRGB linearisation.

    A JPEG that has been contrast-compressed or has large clipped regions
    will read lower than a well-exposed capture.  A high score indicates
    wide usable tonal spread, not necessarily recoverable RAW headroom.

    Method
    ──────
    1. Compute per-pixel Rec. 709 luminance Y.
    2. Take robust percentiles (0.1 % / 99.9 %) to ignore stray pixels.
    3. Clamp p_low to 1e-3 to avoid divide-by-zero and log explosions
       from near-black images or heavy vignetting.
    4. DR in stops = log2(p_high / p_low).
    5. Normalise to [0, 1] against a 12-stop ceiling (typical JPEG range).
    """
    Y     = luminance(image)
    p_low = float(np.percentile(Y, 0.1))
    p_high = float(np.percentile(Y, 99.9))

    p_low = max(p_low, 1e-3)

    if p_high <= p_low:
        return 0.0

    dr = np.log2(p_high / p_low)
    return float(np.clip(dr / 12.0, 0.0, 1.0))
