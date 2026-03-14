"""Color management helpers and pipeline documentation.

ele's internal working space:
  Primaries : ProPhoto RGB (ROMM RGB, ISO 22028-2)
  White point: D50 (CIE illuminant)
  Transfer fn: Linear (gamma 1.0)
  Precision  : float32 (internally), uint16 at export

ProPhoto RGB uses a much wider colour gamut than sRGB or AdobeRGB,
preserving out-of-gamut scene colours that can be recovered during
editing in Lightroom, Photoshop, or Capture One.

ICC profile embedding:
  ele always embeds a ProPhoto RGB ICC profile in TIFF tag 34675.
  Priority order:
    1. Bundled profile at ele/export/profiles/ProPhotoRGB.icc
    2. System profile (macOS ColorSync ROMM RGB, Linux colord, etc.)
    3. Programmatically generated minimal ICC v2 profile (fallback)
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Pipeline documentation constants
# ---------------------------------------------------------------------------

PROPHOTO_PRIMARY_R = (0.7347, 0.2653)   # CIE xy
PROPHOTO_PRIMARY_G = (0.1596, 0.8404)
PROPHOTO_PRIMARY_B = (0.0366, 0.0001)
PROPHOTO_WHITEPOINT = (0.3457, 0.3585)  # D50

PIPELINE_COLOR_PRIMARIES  = "ProPhoto RGB (ROMM RGB)"
PIPELINE_WHITE_POINT      = "D50"
PIPELINE_TRANSFER_FUNCTION = "Linear (gamma 1.0)"
PIPELINE_BIT_DEPTH        = 16

# Candidate system ICC profile paths (searched in order)
_SYSTEM_ICC_CANDIDATES = [
    # macOS ColorSync — ROMM RGB is the canonical name on Apple systems
    Path("/System/Library/ColorSync/Profiles/ROMM RGB.icc"),
    Path("/Library/ColorSync/Profiles/ROMM RGB.icc"),
    # macOS alternative naming
    Path("/System/Library/ColorSync/Profiles/ProPhoto RGB.icc"),
    Path("/Library/ColorSync/Profiles/ProPhoto RGB.icc"),
    # Linux colord / ICC databases
    Path("/usr/share/color/icc/colord/ProPhoto.icc"),
    Path("/usr/share/color/icc/ProPhoto.icc"),
    Path("/usr/share/color/icc/colord/ROMM-RGB.icc"),
]


def find_prophoto_icc() -> bytes | None:
    """Look for a ProPhoto RGB ICC profile on the system.

    Returns the raw bytes of the profile, or None if not found.
    Searches: bundled profile → system ColorSync / colord paths.
    """
    # 1. Bundled profile (expected at ele/export/profiles/ProPhotoRGB.icc)
    bundled = Path(__file__).parent / "profiles" / "ProPhotoRGB.icc"
    if bundled.exists():
        return bundled.read_bytes()

    # 2. System paths
    for p in _SYSTEM_ICC_CANDIDATES:
        if p.exists():
            return p.read_bytes()

    return None


def load_prophoto_icc() -> tuple[bytes, str]:
    """Return (icc_bytes, source_description) for a ProPhoto RGB ICC profile.

    Always returns valid bytes — falls back to a programmatically generated
    minimal ICC v2 profile if no system profile is found.

    Returns:
        (bytes, str): profile bytes and a short description of their origin.
    """
    # 1. Bundled profile
    bundled = Path(__file__).parent / "profiles" / "ProPhotoRGB.icc"
    if bundled.exists():
        return bundled.read_bytes(), f"bundled ({bundled})"

    # 2. System paths
    for p in _SYSTEM_ICC_CANDIDATES:
        if p.exists():
            return p.read_bytes(), f"system ({p})"

    # 3. Generated fallback
    return _build_minimal_prophoto_icc(), "generated (minimal ICC v2 fallback)"


# ---------------------------------------------------------------------------
# Minimal programmatic ICC v2 profile
# ---------------------------------------------------------------------------

def _build_minimal_prophoto_icc() -> bytes:
    """Build a minimal but valid ProPhoto RGB ICC v2 profile.

    Primaries and white point follow ISO 22028-2 (ROMM RGB / ProPhoto RGB).
    Transfer function: linear (gamma 1.0 = identity).
    Profile is self-contained; no external files required.

    Matrix columns (ProPhoto RGB → XYZ D50):
        rXYZ: (0.79767, 0.28804, 0.00000)
        gXYZ: (0.13513, 0.71188, 0.00000)
        bXYZ: (0.03140, 0.00008, 0.82480)
    """

    def s15f16(v: float) -> bytes:
        """Pack float as ICC s15Fixed16Number (big-endian signed 16.16)."""
        return struct.pack(">i", round(v * 65536))

    def xyz_tag(x: float, y: float, z: float) -> bytes:
        return b"XYZ " + b"\x00" * 4 + s15f16(x) + s15f16(y) + s15f16(z)

    def curv_linear() -> bytes:
        """curv tag: count=1, value=256 → gamma 1.0 (linear)."""
        data = b"curv" + b"\x00" * 4 + struct.pack(">I", 1) + struct.pack(">H", 256)
        return data + b"\x00" * 2  # pad to 16 bytes

    def desc_tag(text: str) -> bytes:
        """ICC v2 profileDescriptionTag with ASCII section only."""
        ascii_bytes = text.encode("ascii") + b"\x00"
        data = b"desc"
        data += b"\x00" * 4                          # reserved
        data += struct.pack(">I", len(ascii_bytes))  # ASCII count (inc. null)
        data += ascii_bytes
        data += b"\x00" * 4                          # Unicode language code = 0
        data += struct.pack(">I", 0)                 # Unicode count = 0
        data += b"\x00" * 2                          # Mac: script=0, count=0
        while len(data) % 4:                         # 4-byte alignment
            data += b"\x00"
        return data

    # ── Tag data ──────────────────────────────────────────────────────────────
    t_desc = desc_tag("ProPhoto RGB")
    t_wtpt = xyz_tag(0.96420, 1.00000, 0.82490)   # D50 white point
    t_rXYZ = xyz_tag(0.79767, 0.28804, 0.00000)   # ProPhoto R primary → XYZ
    t_gXYZ = xyz_tag(0.13513, 0.71188, 0.00000)   # ProPhoto G primary → XYZ
    t_bXYZ = xyz_tag(0.03140, 0.00008, 0.82480)   # ProPhoto B primary → XYZ
    t_trc  = curv_linear()                          # shared by rTRC/gTRC/bTRC

    # 8 tags: desc wtpt rXYZ gXYZ bXYZ rTRC gTRC bTRC
    N_TAGS = 8
    DATA_START = 128 + 4 + N_TAGS * 12  # header + count + directory = 228

    # Assign offsets sequentially; TRC tags share the same bytes
    blocks    = [t_desc, t_wtpt, t_rXYZ, t_gXYZ, t_bXYZ, t_trc]
    tag_names = ["desc", "wtpt", "rXYZ", "gXYZ", "bXYZ", "rTRC"]
    offsets: dict[str, tuple[int, int]] = {}
    pos = DATA_START
    for name, blk in zip(tag_names, blocks):
        offsets[name] = (pos, len(blk))
        pos += len(blk)

    offsets["gTRC"] = offsets["rTRC"]
    offsets["bTRC"] = offsets["rTRC"]
    profile_size = pos

    # ── Header (128 bytes, ICC v2 layout) ────────────────────────────────────
    hdr  = struct.pack(">I", profile_size)   # 0-3:   total profile size
    hdr += b"\x00" * 4                        # 4-7:   CMM type
    hdr += b"\x02\x20\x00\x00"               # 8-11:  version 2.2.0.0
    hdr += b"mntr"                            # 12-15: device class (monitor)
    hdr += b"RGB "                            # 16-19: colour space signature
    hdr += b"XYZ "                            # 20-23: PCS
    hdr += b"\x00" * 12                       # 24-35: date/time (zeroed)
    hdr += b"acsp"                            # 36-39: file signature
    hdr += b"APPL"                            # 40-43: primary platform (Apple)
    hdr += b"\x00" * 4                        # 44-47: profile flags
    hdr += b"\x00" * 4                        # 48-51: device manufacturer
    hdr += b"\x00" * 4                        # 52-55: device model
    hdr += b"\x00" * 8                        # 56-63: device attributes
    hdr += struct.pack(">I", 0)               # 64-67: rendering intent (perceptual)
    hdr += s15f16(0.96420)                    # 68-71: PCS illuminant X (D50)
    hdr += s15f16(1.00000)                    # 72-75: PCS illuminant Y
    hdr += s15f16(0.82490)                    # 76-79: PCS illuminant Z
    hdr += b"ele "                            # 80-83: profile creator
    hdr += b"\x00" * 16                       # 84-99: profile ID (MD5, zeroed)
    hdr += b"\x00" * 28                       # 100-127: reserved
    assert len(hdr) == 128, f"ICC header must be 128 bytes, got {len(hdr)}"

    # ── Tag count + directory ──────────────────────────────────────────────────
    tag_dir = struct.pack(">I", N_TAGS)
    for sig, key in [
        (b"desc", "desc"), (b"wtpt", "wtpt"),
        (b"rXYZ", "rXYZ"), (b"gXYZ", "gXYZ"), (b"bXYZ", "bXYZ"),
        (b"rTRC", "rTRC"), (b"gTRC", "gTRC"), (b"bTRC", "bTRC"),
    ]:
        off, sz = offsets[key]
        tag_dir += sig + struct.pack(">II", off, sz)

    body = b"".join(blocks)
    return hdr + tag_dir + body


def build_pipeline_metadata() -> dict[str, str | int]:
    """Return a dictionary describing the ele colour pipeline.

    Intended for embedding in TIFF ImageDescription or similar tags.
    """
    return {
        "Software":             "ele (pseudo-RAW master preprocessing engine)",
        "ColorPrimaries":       PIPELINE_COLOR_PRIMARIES,
        "WhitePoint":           PIPELINE_WHITE_POINT,
        "TransferCharacteristics": PIPELINE_TRANSFER_FUNCTION,
        "BitDepth":             PIPELINE_BIT_DEPTH,
        "Note":                 "This file is an unrendered linear master.",
    }


def prepare_for_export(image: np.ndarray) -> np.ndarray:
    """Validate and clip a float32 image for 16-bit export.

    Args:
        image: H×W×3 float32, nominally [0, 1].

    Returns:
        Same array clipped to [0, 1], float32.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected H×W×3 array, got shape {image.shape}")
    return np.clip(image, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Colorimetric conversion matrices
# ---------------------------------------------------------------------------
#
# The ele internal working space is **linear sRGB primaries** (D65), obtained
# by applying the inverse sRGB EOTF to the input JPEG/PNG.  ProPhoto RGB
# (ROMM RGB) has wider primaries (D50).  The matrices below enable lossless
# round-tripping between these two spaces.
#
# All values are derived from the standard component matrices:
#   sRGB → XYZ D65  (IEC 61966-2-1)
#   Bradford D65→D50 chromatic adaptation
#   XYZ D50 → ProPhoto  (ISO 22028-2 / ICC spec)
#
# The sRGB gamut is a strict subset of the ProPhoto gamut, so no values go
# below 0 or above 1 when converting sRGB [0,1] → ProPhoto.
# ---------------------------------------------------------------------------

def _build_srgb_to_prophoto() -> np.ndarray:
    """Compute the 3×3 linear sRGB (D65) → linear ProPhoto RGB (D50) matrix."""
    M_srgb_to_xyz = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float64)
    M_d65_to_d50 = np.array([              # Bradford chromatic adaptation
        [ 1.0478112,  0.0228866, -0.0501270],
        [ 0.0295424,  0.9904844, -0.0170491],
        [-0.0092345,  0.0150436,  0.7521316],
    ], dtype=np.float64)
    M_xyz_to_prophoto = np.array([
        [ 1.3459433, -0.2556075, -0.0511118],
        [-0.5445989,  1.5081673,  0.0205351],
        [ 0.0000000,  0.0000000,  1.2118128],
    ], dtype=np.float64)
    return (M_xyz_to_prophoto @ M_d65_to_d50 @ M_srgb_to_xyz).astype(np.float32)


_SRGB_TO_PROPHOTO: np.ndarray = _build_srgb_to_prophoto()
_PROPHOTO_TO_SRGB: np.ndarray = np.linalg.inv(_SRGB_TO_PROPHOTO).astype(np.float32)


# ---------------------------------------------------------------------------
# Transfer-curve helpers
# ---------------------------------------------------------------------------

def apply_prophoto_export_trc(linear: np.ndarray) -> np.ndarray:
    """Apply ROMM RGB gamma encoding (ISO 22028-2): linear → encoded.

    E' = 8 × E              for E <  0.001953  (linear foot)
    E' = E^(1 / 1.8)        for E >= 0.001953
    """
    encoded = np.where(
        linear < 0.001953,
        linear * 8.0,
        np.power(np.clip(linear, 0.0, None), 1.0 / 1.8),
    )
    return encoded.clip(0.0, 1.0).astype(np.float32)


def apply_prophoto_inverse_trc(encoded: np.ndarray) -> np.ndarray:
    """Inverse ROMM RGB TRC: encoded → linear.

    E = E' / 8              for E' <  0.015625  (= 8 × 0.001953)
    E = E'^1.8              for E' >= 0.015625
    """
    linear = np.where(
        encoded < 0.015625,
        encoded / 8.0,
        np.power(np.clip(encoded, 0.0, None), 1.8),
    )
    return linear.clip(0.0, None).astype(np.float32)


def apply_srgb_display_trc(linear: np.ndarray) -> np.ndarray:
    """Apply sRGB OETF (IEC 61966-2-1): linear → sRGB gamma-encoded."""
    encoded = np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(np.clip(linear, 0.0031308, None), 1.0 / 2.4) - 0.055,
    )
    return encoded.clip(0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# High-level output transforms
# ---------------------------------------------------------------------------

def convert_linear_working_to_export_space(image: np.ndarray) -> np.ndarray:
    """Convert internal linear sRGB primaries to linear ProPhoto RGB (D50).

    The ele internal image is in linear sRGB primaries (result of decoding
    an sRGB JPEG/PNG input).  ProPhoto has wider primaries; this matrix maps
    the sRGB gamut into the ProPhoto encoding space without clipping.

    Note: the mapping is applied before gamma encoding.  Out-of-gamut
    pixel values (if any, from L3 processing) are clamped at the caller.
    """
    h, w = image.shape[:2]
    flat = image.reshape(-1, 3).astype(np.float64)
    out  = (flat @ _SRGB_TO_PROPHOTO.T.astype(np.float64)).astype(np.float32)
    return out.reshape(h, w, 3)


def to_export_prophoto_tiff(image: np.ndarray) -> np.ndarray:
    """Prepare the internal linear image for ProPhoto TIFF export.

    Pipeline
    ────────
    1. Convert linear sRGB primaries → linear ProPhoto RGB primaries.
    2. Apply ROMM RGB gamma encoding (ISO 22028-2).
    3. Clip to [0, 1].

    The output should be quantised to uint16 and saved with ProPhoto /
    ROMM RGB ICC embedded (TIFF tag 34675).  This ensures Photoshop,
    Lightroom, and Capture One decode the values correctly.

    Colorimetric note
    ─────────────────
    ele processes in linear sRGB primaries, not true linear ProPhoto.
    The primary conversion step (sRGB → ProPhoto matrix) corrects this
    before export, so the embedded ICC matches the stored values.
    """
    prophoto_linear = convert_linear_working_to_export_space(image)
    return apply_prophoto_export_trc(prophoto_linear)


def to_display_srgb_preview(image: np.ndarray) -> np.ndarray:
    """Prepare the internal linear image for sRGB browser display.

    Input is linear sRGB primaries (no primary conversion needed).
    Applies sRGB OETF for 8-bit JPEG/PNG output.
    """
    return apply_srgb_display_trc(image)


def decode_prophoto_tiff_for_preview(encoded: np.ndarray) -> np.ndarray:
    """Convert a ROMM-gamma-encoded ProPhoto array to sRGB for browser display.

    This is the inverse of to_export_prophoto_tiff, followed by a
    ProPhoto → sRGB primary conversion and sRGB OETF.  Used to generate
    accurate browser previews from the exported TIFF without re-running
    the full pipeline.

    Pipeline
    ────────
    1. Inverse ROMM TRC → linear ProPhoto
    2. ProPhoto primaries → linear sRGB primaries
    3. sRGB OETF → sRGB gamma-encoded [0, 1]
    """
    linear_prophoto = apply_prophoto_inverse_trc(encoded)
    h, w = linear_prophoto.shape[:2]
    flat = linear_prophoto.reshape(-1, 3).astype(np.float64)
    linear_srgb = (flat @ _PROPHOTO_TO_SRGB.T.astype(np.float64)).clip(0.0, 1.0)
    linear_srgb = linear_srgb.reshape(h, w, 3).astype(np.float32)
    return apply_srgb_display_trc(linear_srgb)
