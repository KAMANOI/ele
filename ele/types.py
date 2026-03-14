"""ele shared data types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Mode(str, Enum):
    FREE    = "free"
    CREATOR = "creator"
    PRO     = "pro"
    PRINT   = "print"


@dataclass
class DegradationReport:
    """Lightweight analysis of input image quality."""

    compression_score:    float        # 0–1, higher = more blocking artefacts
    clipping_score:       float        # 0–1, higher = more clipped pixels
    sharpness_score:      float        # 0–1, higher = sharper image
    noise_score:          float        # 0–1, higher = more noise
    dynamic_range_score:  float        # 0–1, higher = wider DR
    ringing_score:        float = 0.0 # 0–1, higher = more edge-adjacent ringing
    notes:                list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"  compression    : {self.compression_score:.3f}",
            f"  clipping       : {self.clipping_score:.3f}",
            f"  sharpness      : {self.sharpness_score:.3f}",
            f"  noise          : {self.noise_score:.3f}",
            f"  dynamic range  : {self.dynamic_range_score:.3f}",
            f"  ringing        : {self.ringing_score:.3f}",
        ]
        if self.notes:
            lines.append("  notes          : " + "; ".join(self.notes))
        return "\n".join(lines)


@dataclass
class SceneMap:
    """Per-pixel region masks (float32, 0–1 soft membership)."""

    sky:          np.ndarray
    foliage:      np.ndarray
    skin:         np.ndarray
    architecture: np.ndarray
    hair:         np.ndarray
    fabric:       np.ndarray

    def __post_init__(self) -> None:
        # All masks must be 2-D float32
        for name in ("sky", "foliage", "skin", "architecture", "hair", "fabric"):
            arr = getattr(self, name)
            if arr.ndim != 2:
                raise ValueError(f"SceneMap.{name} must be 2-D, got shape {arr.shape}")


@dataclass
class PipelineResult:
    """Final output of any pipeline run."""

    image:       np.ndarray
    report:      DegradationReport
    output_path: str | None
    metadata:    dict[str, str | int | float] = field(default_factory=dict)
