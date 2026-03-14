"""Tests for TIFF export."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest


def _synthetic_image(h: int = 32, w: int = 32) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.random((h, w, 3), dtype=np.float64).astype(np.float32)


def test_export_tiff_creates_file() -> None:
    from ele.export.tiff_export import export_tiff

    image = _synthetic_image()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = str(Path(tmpdir) / "test_output.tiff")
        result = export_tiff(image, out)
        assert Path(result).exists(), "Output file was not created"
        assert Path(result).stat().st_size > 0, "Output file is empty"


def test_export_tiff_enforces_extension() -> None:
    from ele.export.tiff_export import export_tiff

    image = _synthetic_image()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = str(Path(tmpdir) / "test_output.jpg")   # wrong extension
        result = export_tiff(image, out)
        assert result.endswith(".tiff"), f"Expected .tiff extension, got {result}"
        assert Path(result).exists()


def test_export_tiff_16bit() -> None:
    """Verify exported TIFF has 16-bit pixel depth."""
    import tifffile
    from ele.export.tiff_export import export_tiff

    image = _synthetic_image()
    with tempfile.TemporaryDirectory() as tmpdir:
        out = str(Path(tmpdir) / "output.tiff")
        export_tiff(image, out)
        loaded = tifffile.imread(out)
        assert loaded.dtype == np.uint16, f"Expected uint16, got {loaded.dtype}"
        assert loaded.shape == (32, 32, 3)


def test_export_tiff_value_range() -> None:
    """Verify pixel values map correctly to 16-bit range after ROMM gamma encoding.

    The export pipeline converts linear sRGB → ProPhoto (matrix) → ROMM gamma.
    A neutral grey [0.5, 0.5, 0.5] is preserved through the matrix (white-point
    preserving), then encoded as 0.5^(1/1.8) ≈ 0.680, giving uint16 ≈ 44590.
    White (1.0) and black (0.0) are invariant under ROMM gamma.
    """
    import tifffile
    from ele.export.tiff_export import export_tiff
    from ele.export.color_management import to_export_prophoto_tiff

    image = np.zeros((4, 4, 3), dtype=np.float32)
    image[0, 0, :] = 1.0   # white pixel
    image[1, 1, :] = 0.0   # black pixel
    image[2, 2, :] = 0.5   # mid-grey (neutral)

    # Compute expected uint16 for mid-grey using the same transform
    mid_enc = to_export_prophoto_tiff(np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32))
    expected_mid = int(round(float(mid_enc[0, 0, 0]) * 65535.0))

    with tempfile.TemporaryDirectory() as tmpdir:
        out = str(Path(tmpdir) / "range_test.tiff")
        export_tiff(image, out)
        loaded = tifffile.imread(out)

        assert loaded[0, 0, 0] == 65535,  "White pixel: expected 65535"
        assert loaded[1, 1, 0] == 0,      "Black pixel: expected 0"
        # Mid-grey should match ROMM-gamma encoded value (±1 for rounding)
        assert abs(int(loaded[2, 2, 0]) - expected_mid) <= 1, (
            f"Mid-grey mismatch: got {loaded[2, 2, 0]}, expected ~{expected_mid} "
            "(ROMM gamma encoded, not linear)"
        )


def test_dng_export_raises() -> None:
    from ele.export.dng_export import export_dng

    image = _synthetic_image()
    with pytest.raises(NotImplementedError):
        export_dng(image, "/tmp/test.dng")
