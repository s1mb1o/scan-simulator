"""Physical damage transforms: fold marks, wrinkles, edge wear."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from .base import BaseTransform, TransformResult


class FoldMark(BaseTransform):
    """Straight crease lines with slight offset and shadow."""

    affects_geometry = True

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        count = self._sample_int("count", 1)
        shadow_width = self._sample_int("shadow_width", 3)
        h, w = image.shape[:2]
        result = image.copy()
        result_mask = mask.copy() if mask is not None else None

        for _ in range(count):
            # Random line across the image (mostly horizontal or vertical)
            vertical = self.rng.random() < 0.5
            if vertical:
                x = int(self.rng.integers(w // 5, 4 * w // 5))
                pt1, pt2 = (x, 0), (x + int(self.rng.integers(-w // 20, w // 20)), h)
            else:
                y = int(self.rng.integers(h // 5, 4 * h // 5))
                pt1, pt2 = (0, y), (w, y + int(self.rng.integers(-h // 20, h // 20)))

            # Draw shadow line (dark)
            shadow = np.zeros((h, w), dtype=np.float32)
            cv2.line(shadow, pt1, pt2, 1.0, shadow_width, lineType=cv2.LINE_AA)
            shadow = gaussian_filter(shadow, sigma=shadow_width * 0.6)

            # Dark shadow on one side, slight brightening on other
            darkness = self._sample("darkness", 0.25)
            result = result.astype(np.float32)
            result -= shadow[..., np.newaxis] * 255 * darkness

            # Slight highlight next to shadow
            if vertical:
                shadow_shift = np.roll(shadow, shadow_width + 1, axis=1)
            else:
                shadow_shift = np.roll(shadow, shadow_width + 1, axis=0)
            result += shadow_shift[..., np.newaxis] * 255 * darkness * 0.3

            result = np.clip(result, 0, 255).astype(np.uint8)

            # Apply slight geometric offset along fold
            if result_mask is not None:
                offset_px = max(1, shadow_width // 3)
                displacement = (shadow * offset_px).astype(np.float32)
                rows, cols = np.mgrid[:h, :w].astype(np.float32)
                if vertical:
                    map_x = cols + displacement
                    map_y = rows
                else:
                    map_x = cols
                    map_y = rows + displacement
                result_mask = cv2.remap(result_mask, map_x, map_y,
                                        cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_REPLICATE)

        return TransformResult(result, result_mask)


class Wrinkle(BaseTransform):
    """Local elastic deformation simulating paper wrinkles."""

    affects_geometry = True

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        count = self._sample_int("count", 2)
        strength = self._sample("strength", 5.0)
        h, w = image.shape[:2]

        # Accumulate displacement field
        dx = np.zeros((h, w), dtype=np.float32)
        dy = np.zeros((h, w), dtype=np.float32)

        for _ in range(count):
            # Random wrinkle center and direction
            cx = self.rng.integers(w // 6, 5 * w // 6)
            cy = self.rng.integers(h // 6, 5 * h // 6)
            length = self.rng.integers(min(h, w) // 6, min(h, w) // 2)
            angle = self.rng.uniform(0, np.pi)

            # Create wrinkle displacement along a line
            y, x = np.ogrid[:h, :w]
            # Distance from wrinkle line
            nx, ny = np.cos(angle), np.sin(angle)
            dist_along = (x - cx) * nx + (y - cy) * ny
            dist_perp = np.abs((x - cx) * (-ny) + (y - cy) * nx)

            # Wrinkle profile: displacement perpendicular to fold, falloff with distance
            along_mask = np.exp(-(dist_along ** 2) / (2 * (length / 3) ** 2))
            perp_profile = np.exp(-(dist_perp ** 2) / (2 * (strength * 2) ** 2))
            wrinkle = along_mask * perp_profile * strength

            dx += (wrinkle * (-ny)).astype(np.float32)
            dy += (wrinkle * nx).astype(np.float32)

        # Apply displacement
        rows, cols = np.mgrid[:h, :w].astype(np.float32)
        map_x = cols + dx
        map_y = rows + dy

        result = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)

        # Add highlight/shadow along wrinkles
        gradient = np.gradient(dy, axis=0) + np.gradient(dx, axis=1)
        gradient = gaussian_filter(gradient, sigma=1.5)
        gradient = gradient / (np.abs(gradient).max() + 1e-8) * 30
        result = np.clip(result.astype(np.float32) + gradient[..., np.newaxis],
                         0, 255).astype(np.uint8)

        result_mask = mask
        if mask is not None:
            result_mask = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_REPLICATE)

        return TransformResult(result, result_mask)


class EdgeWear(BaseTransform):
    """Darkened/damaged margins simulating handling wear."""

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        width_frac = self._sample("width", 0.06)
        darkness = self._sample("darkness", 0.5)
        h, w = image.shape[:2]

        width_px = int(max(w, h) * width_frac)

        # Create margin mask (smooth falloff)
        margin = np.ones((h, w), dtype=np.float32)
        for i in range(width_px):
            alpha = (i / width_px) ** 1.5
            if i < h and i < w:
                margin[i, :] = min(margin[i, 0], alpha)
                margin[h - 1 - i, :] = np.minimum(margin[h - 1 - i, :], alpha)
                margin[:, i] = np.minimum(margin[:, i], alpha)
                margin[:, w - 1 - i] = np.minimum(margin[:, w - 1 - i], alpha)

        # Add noise to margin
        noise = self.rng.standard_normal((h, w)).astype(np.float32) * 0.1
        noise = gaussian_filter(noise, sigma=3)
        margin = np.clip(margin + noise * (1 - margin), 0, 1)

        darken = 1.0 - (1.0 - margin) * darkness
        result = (image.astype(np.float32) * darken[..., np.newaxis])
        return TransformResult(np.clip(result, 0, 255).astype(np.uint8), mask)
