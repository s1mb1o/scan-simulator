"""Base transform class."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TransformResult:
    """Result of applying a transform."""
    image: np.ndarray
    mask: np.ndarray | None = None


class BaseTransform(abc.ABC):
    """Abstract base for all degradation transforms.

    Each transform declares whether it affects geometry (and thus must also
    warp the GT mask) or is color-only.
    """

    affects_geometry: bool = False

    def __init__(self, rng: np.random.Generator | None = None, **params: Any):
        self.rng = rng or np.random.default_rng()
        self.params = params

    def _sample(self, key: str, default: float) -> float:
        """Sample a parameter value from [lo, hi] range or use scalar."""
        val = self.params.get(key, default)
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return float(self.rng.uniform(val[0], val[1]))
        return float(val)

    def _sample_int(self, key: str, default: int) -> int:
        val = self.params.get(key, default)
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return int(self.rng.integers(val[0], val[1] + 1))
        return int(val)

    @abc.abstractmethod
    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        """Apply the transform. Image is BGR uint8, mask is grayscale uint8."""
        ...

    def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        return self.apply(image, mask)
