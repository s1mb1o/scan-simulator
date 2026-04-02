"""Transform pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import PipelineConfig
from .presets import get_preset
from .transforms import REGISTRY
from .transforms.base import TransformResult


class ScanSimulator:
    """Main pipeline: applies a sequence of random degradation transforms."""

    def __init__(self, config: PipelineConfig, seed: int | None = None):
        self.config = config
        self.seed = seed or config.seed
        self._rng = np.random.default_rng(self.seed)

    @classmethod
    def from_preset(cls, name: str, seed: int | None = None) -> ScanSimulator:
        config = get_preset(name)
        return cls(config, seed=seed)

    @classmethod
    def from_yaml(cls, path: str | Path, seed: int | None = None) -> ScanSimulator:
        config = PipelineConfig.from_yaml(path)
        return cls(config, seed=seed)

    def __call__(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Apply random degradation pipeline.

        Args:
            image: BGR uint8 image
            mask: Optional grayscale uint8 GT mask

        Returns:
            (degraded_image, degraded_mask) — mask is None if not provided
        """
        result = TransformResult(image.copy(), mask.copy() if mask is not None else None)

        for tc in self.config.transforms:
            if tc.name not in REGISTRY:
                raise ValueError(f"Unknown transform '{tc.name}'. "
                                 f"Available: {', '.join(sorted(REGISTRY.keys()))}")

            # Probability gate
            if self._rng.random() > tc.p:
                continue

            # Instantiate with fresh per-call RNG (derived from pipeline RNG)
            child_seed = int(self._rng.integers(0, 2**31))
            child_rng = np.random.default_rng(child_seed)
            transform = REGISTRY[tc.name](rng=child_rng, **tc.params)

            result = transform(result.image, result.mask)

        return result.image, result.mask

    def preview_grid(
        self,
        image: np.ndarray,
        rows: int = 3,
        cols: int = 3,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate a grid of random variants for visual inspection."""
        import cv2

        cell_h, cell_w = image.shape[:2]
        # Scale down if too large
        max_cell = 400
        if max(cell_h, cell_w) > max_cell:
            scale = max_cell / max(cell_h, cell_w)
            cell_w = int(cell_w * scale)
            cell_h = int(cell_h * scale)
            image = cv2.resize(image, (cell_w, cell_h))
            if mask is not None:
                mask = cv2.resize(mask, (cell_w, cell_h),
                                  interpolation=cv2.INTER_NEAREST)

        grid = np.zeros((cell_h * rows, cell_w * cols, 3), dtype=np.uint8)

        for r in range(rows):
            for c in range(cols):
                degraded, _ = self(image, mask)
                y0, x0 = r * cell_h, c * cell_w
                grid[y0:y0 + cell_h, x0:x0 + cell_w] = degraded

        return grid
