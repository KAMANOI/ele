"""CLI smoke tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from typer.testing import CliRunner

from ele.cli.main import app

runner = CliRunner()


def _make_jpeg(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    arr = (np.random.default_rng(0).random((*size, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path, format="JPEG", quality=85)


def test_cli_help_runs() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ele" in result.output.lower() or "pseudo" in result.output.lower()


def test_cli_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1" in result.output


def test_cli_free_mode_creates_tiff() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "input.jpg"
        out = Path(tmpdir) / "output.tiff"
        _make_jpeg(src)

        result = runner.invoke(app, [str(src), "--mode", "free", "--output", str(out)])
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert out.exists(), "Output file not created"
        assert out.stat().st_size > 0


def test_cli_creator_mode_with_report() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "input.jpg"
        out = Path(tmpdir) / "output.tiff"
        _make_jpeg(src)

        result = runner.invoke(
            app,
            [str(src), "--mode", "creator", "--output", str(out), "--report"],
        )
        assert result.exit_code == 0, f"CLI failed: {result.output}"
        assert out.exists()


def test_cli_missing_input_fails() -> None:
    result = runner.invoke(app, ["nonexistent.jpg", "--output", "out.tiff"])
    assert result.exit_code != 0
