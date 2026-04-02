"""Paper & aging transforms: color, texture, ink fading, stains."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from .base import BaseTransform, TransformResult


class PaperColor(BaseTransform):
    """Non-uniform paper yellowing/aging via smooth noise in sepia colorspace."""

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        intensity = self._sample("intensity", 0.2)
        color = self.params.get("color", [230, 210, 170])  # warm sepia BGR

        h, w = image.shape[:2]
        # Generate smooth spatial variation
        noise = self.rng.standard_normal((h // 8 + 1, w // 8 + 1)).astype(np.float32)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
        noise = gaussian_filter(noise, sigma=w // 10)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

        # Create tint overlay
        tint = np.full_like(image, color, dtype=np.float32)
        alpha = (noise * intensity)[..., np.newaxis]

        result = image.astype(np.float32) * (1.0 - alpha) + tint * alpha
        return TransformResult(np.clip(result, 0, 255).astype(np.uint8), mask)


class PaperTexture(BaseTransform):
    """Procedural paper grain/fiber texture overlay."""

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        strength = self._sample("strength", 0.1)
        h, w = image.shape[:2]

        # High-frequency noise for grain
        grain = self.rng.standard_normal((h, w)).astype(np.float32)
        grain = gaussian_filter(grain, sigma=0.8)

        # Low-frequency variation for fibers
        fibers = self.rng.standard_normal((h // 4 + 1, w // 4 + 1)).astype(np.float32)
        fibers = cv2.resize(fibers, (w, h), interpolation=cv2.INTER_CUBIC)
        fibers = gaussian_filter(fibers, sigma=3)

        texture = 0.7 * grain + 0.3 * fibers
        texture = texture / (np.abs(texture).max() + 1e-8)

        result = image.astype(np.float32) + texture[..., np.newaxis] * 255 * strength
        return TransformResult(np.clip(result, 0, 255).astype(np.uint8), mask)


class InkFading(BaseTransform):
    """Spatially varying contrast reduction on dark pixels (ink fading)."""

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        strength = self._sample("strength", 0.25)
        h, w = image.shape[:2]

        # Spatial fade map (smooth)
        fade_map = self.rng.standard_normal((h // 16 + 1, w // 16 + 1)).astype(np.float32)
        fade_map = cv2.resize(fade_map, (w, h), interpolation=cv2.INTER_CUBIC)
        fade_map = gaussian_filter(fade_map, sigma=w // 8)
        fade_map = (fade_map - fade_map.min()) / (fade_map.max() - fade_map.min() + 1e-8)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # Dark pixels get faded more
        darkness = 1.0 - gray  # 1.0 for black ink, 0.0 for white paper
        fade_amount = darkness * fade_map * strength

        result = image.astype(np.float32)
        result += fade_amount[..., np.newaxis] * 255
        return TransformResult(np.clip(result, 0, 255).astype(np.uint8), mask)


class CoffeeStain(BaseTransform):
    """Elliptical brownish blotches with ring edges."""

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        count = self._sample_int("count", 1)
        h, w = image.shape[:2]
        result = image.astype(np.float32)

        for _ in range(count):
            cx = self.rng.integers(w // 4, 3 * w // 4)
            cy = self.rng.integers(h // 4, 3 * h // 4)
            rx = self.rng.integers(20, min(100, w // 4))
            ry = self.rng.integers(int(rx * 0.7), int(rx * 1.3) + 1)

            y, x = np.ogrid[:h, :w]
            dist = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2

            # Ring effect: stronger at edge
            ring = np.exp(-((dist - 0.8) ** 2) / 0.1)
            fill = np.exp(-dist / 0.5)
            stain_alpha = np.clip(0.3 * fill + 0.5 * ring, 0, 1).astype(np.float32)

            # Brown color
            brown = np.array([60, 120, 180], dtype=np.float32)  # BGR
            opacity = self._sample("opacity", 0.3)
            alpha = (stain_alpha * opacity)[..., np.newaxis]
            result = result * (1 - alpha) + brown * alpha

        return TransformResult(np.clip(result, 0, 255).astype(np.uint8), mask)


class Foxing(BaseTransform):
    """Small brown spots simulating aged paper fungal damage."""

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        count = self._sample_int("count", 30)
        h, w = image.shape[:2]
        result = image.copy()

        for _ in range(count):
            cx = int(self.rng.integers(0, w))
            cy = int(self.rng.integers(0, h))
            radius = self._sample_int("radius", 4)
            # Brownish color with variation
            color = (
                int(40 + self.rng.integers(0, 40)),
                int(80 + self.rng.integers(0, 60)),
                int(140 + self.rng.integers(0, 60)),
            )
            cv2.circle(result, (cx, cy), radius, color, -1, lineType=cv2.LINE_AA)

        # Soften spots
        blurred = cv2.GaussianBlur(result, (3, 3), 0.8)
        # Only blend where spots were drawn
        diff = cv2.absdiff(result, image)
        spot_mask = (diff.sum(axis=2) > 10).astype(np.float32)[..., np.newaxis]
        result = (image.astype(np.float32) * (1 - spot_mask * 0.7) +
                  blurred.astype(np.float32) * spot_mask * 0.7)
        return TransformResult(np.clip(result, 0, 255).astype(np.uint8), mask)
