"""Scanner artifact transforms: rotation, perspective, illumination, moiré, rollers."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from .base import BaseTransform, TransformResult


class Rotation(BaseTransform):
    """Slight rotation simulating misaligned paper in scanner."""

    affects_geometry = True

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        angle = self._sample("angle", 2.0)
        if self.rng.random() < 0.5 and not isinstance(self.params.get("angle"), (list, tuple)):
            angle = -angle

        h, w = image.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2

        bg_color = (245, 245, 240)
        result = cv2.warpAffine(image, M, (new_w, new_h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=bg_color)
        start_x = (new_w - w) // 2
        start_y = (new_h - h) // 2
        result = result[start_y:start_y + h, start_x:start_x + w]

        result_mask = mask
        if mask is not None:
            result_mask = cv2.warpAffine(mask, M, (new_w, new_h),
                                         flags=cv2.INTER_NEAREST,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=0)
            result_mask = result_mask[start_y:start_y + h, start_x:start_x + w]

        return TransformResult(result, result_mask)


class Perspective(BaseTransform):
    """Mild trapezoid distortion simulating photographing at an angle."""

    affects_geometry = True

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        strength = self._sample("strength", 0.03)
        h, w = image.shape[:2]

        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        max_shift = strength * min(h, w)
        dst = src.copy()
        for i in range(4):
            dst[i, 0] += self.rng.uniform(-max_shift, max_shift)
            dst[i, 1] += self.rng.uniform(-max_shift, max_shift)

        M = cv2.getPerspectiveTransform(src, dst)
        result = cv2.warpPerspective(image, M, (w, h),
                                     borderMode=cv2.BORDER_REPLICATE)

        result_mask = mask
        if mask is not None:
            result_mask = cv2.warpPerspective(mask, M, (w, h),
                                              flags=cv2.INTER_NEAREST,
                                              borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=0)

        return TransformResult(result, result_mask)


class UnevenLight(BaseTransform):
    """Uneven illumination: vignetting, directional gradient, or flash hotspot."""

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        falloff = self._sample("falloff", 0.2)
        h, w = image.shape[:2]

        mode = self.rng.choice(["vignette", "directional", "hotspot"])

        if mode == "vignette":
            y, x = np.ogrid[:h, :w]
            cy, cx = h / 2, w / 2
            dist = np.sqrt(((x - cx) / (w / 2)) ** 2 + ((y - cy) / (h / 2)) ** 2)
            light = 1.0 - falloff * np.clip(dist - 0.3, 0, None)
        elif mode == "directional":
            angle = self.rng.uniform(0, 2 * np.pi)
            y, x = np.mgrid[:h, :w].astype(np.float32)
            grad = (x / w * np.cos(angle) + y / h * np.sin(angle))
            grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
            light = 1.0 - falloff * grad
        else:
            cx = self.rng.uniform(0.2, 0.8) * w
            cy = self.rng.uniform(0.2, 0.8) * h
            y, x = np.ogrid[:h, :w]
            dist = np.sqrt(((x - cx) / (w / 2)) ** 2 + ((y - cy) / (h / 2)) ** 2)
            light = 1.0 + falloff * 0.5 * np.exp(-dist ** 2 / 0.3) - falloff * 0.3 * dist

        light = np.clip(light, 0.3, 1.5).astype(np.float32)
        result = np.clip(image.astype(np.float32) * light[..., np.newaxis],
                         0, 255).astype(np.uint8)
        return TransformResult(result, mask)


class ScannerShadow(BaseTransform):
    """Dark gradient along one edge (scanner lid shadow)."""

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        width_frac = self._sample("width", 0.08)
        darkness = self._sample("darkness", 0.45)
        h, w = image.shape[:2]

        edge = self.rng.choice(["top", "bottom", "left", "right"])
        shadow = np.ones((h, w), dtype=np.float32)

        width_px = int(max(h, w) * width_frac)
        ramp = np.linspace(1.0 - darkness, 1.0, width_px)

        if edge == "top":
            shadow[:width_px, :] = ramp[:, np.newaxis]
        elif edge == "bottom":
            shadow[-width_px:, :] = ramp[::-1, np.newaxis]
        elif edge == "left":
            shadow[:, :width_px] = ramp[np.newaxis, :]
        else:
            shadow[:, -width_px:] = ramp[::-1][np.newaxis, :]

        result = (image.astype(np.float32) * shadow[..., np.newaxis])
        return TransformResult(np.clip(result, 0, 255).astype(np.uint8), mask)


class Moire(BaseTransform):
    """Additive sinusoidal moiré interference pattern."""

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        frequency = self._sample("frequency", 0.1)
        strength = self._sample("strength", 0.05)
        h, w = image.shape[:2]

        angle = self.rng.uniform(0, np.pi)
        y, x = np.mgrid[:h, :w].astype(np.float32)
        phase = x * np.cos(angle) + y * np.sin(angle)

        f1 = frequency * 2 * np.pi
        f2 = frequency * 2 * np.pi * 1.07
        pattern = np.sin(phase * f1) * np.sin(phase * f2)

        result = image.astype(np.float32) + pattern[..., np.newaxis] * 255 * strength
        return TransformResult(np.clip(result, 0, 255).astype(np.uint8), mask)


class DirtyRollers(BaseTransform):
    """Horizontal/vertical banding from dirty scanner rollers.

    Inspired by Augraphy's DirtyRollers: creates gradient bars tiled across
    the image with a large-scale meta-mask for non-uniform roller pressure.
    """

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        intensity = self._sample("intensity", 0.15)
        line_width = self._sample_int("line_width", 8)
        h, w = image.shape[:2]
        vertical = self.rng.random() < 0.5

        # Build bar templates (bright center → dark edges, with variations)
        bar_len = line_width * 2
        bars = []
        for _ in range(6):
            hi = self.rng.uniform(0.88, 0.99)
            lo = self.rng.uniform(0.70, 0.87)
            half = np.linspace(lo, hi, bar_len // 2)
            bar = np.concatenate([half, half[::-1]])
            bars.append(bar)

        # Tile bars across the scan direction
        scan_dim = w if not vertical else h
        scanline = np.ones(scan_dim, dtype=np.float32)
        pos = 0
        while pos < scan_dim:
            bar = bars[int(self.rng.integers(0, len(bars)))]
            end = min(pos + len(bar), scan_dim)
            scanline[pos:end] = bar[:end - pos]
            pos = end + int(self.rng.integers(0, line_width))

        # Meta-mask: large-scale pressure variation
        meta_len = max(8, scan_dim // (line_width * 3))
        meta = self.rng.uniform(0.5, 1.0, meta_len).astype(np.float32)
        meta = np.interp(np.linspace(0, meta_len - 1, scan_dim), np.arange(meta_len), meta)
        meta = gaussian_filter(meta, sigma=line_width * 2)

        # Combine scanline × meta
        combined = 1.0 - (1.0 - scanline) * meta * intensity

        # Apply as multiplicative mask
        if vertical:
            mask_2d = np.tile(combined[:, np.newaxis], (1, w))
        else:
            mask_2d = np.tile(combined[np.newaxis, :], (h, 1))

        # Slight random blur for realism
        mask_2d = gaussian_filter(mask_2d, sigma=1.0)

        result = np.clip(image.astype(np.float32) * mask_2d[..., np.newaxis],
                         0, 255).astype(np.uint8)
        return TransformResult(result, mask)


class BookBinding(BaseTransform):
    """Page curvature near book spine — darkening + slight geometric warp.

    Simulates a page photographed/scanned while still bound in a book.
    One edge (spine side) curves inward with progressive darkening and
    mild horizontal compression.
    """

    affects_geometry = True

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        curve_depth = self._sample("curve_depth", 0.04)
        darkness = self._sample("darkness", 0.4)
        h, w = image.shape[:2]

        spine_side = self.rng.choice(["left", "right"])

        # Create curvature profile: cosine falloff from spine edge
        curve_width = int(w * self._sample("curve_width", 0.25))
        x = np.arange(w, dtype=np.float32)
        if spine_side == "left":
            # Spine at x=0, falloff toward right
            curve = np.clip(1.0 - x / curve_width, 0, 1)
        else:
            curve = np.clip((x - (w - curve_width)) / curve_width, 0, 1)

        # Smooth cosine shape
        curve = (1.0 - np.cos(curve * np.pi)) / 2.0

        # Darkening mask (shadow in spine fold)
        dark_mask = curve * darkness
        result = image.astype(np.float32) * (1.0 - dark_mask[np.newaxis, :, np.newaxis])

        # Geometric warp: horizontal compression near spine
        displacement = curve * curve_depth * w
        map_x = np.tile(x[np.newaxis, :], (h, 1)).astype(np.float32)
        map_y = np.tile(np.arange(h, dtype=np.float32)[:, np.newaxis], (1, w))

        if spine_side == "left":
            map_x += displacement[np.newaxis, :]
        else:
            map_x -= displacement[np.newaxis, :]

        # Slight vertical stretch near spine (page lifts)
        v_stretch = 1.0 + curve * 0.02
        center_y = h / 2
        map_y = center_y + (map_y - center_y) / v_stretch[np.newaxis, :]

        result = cv2.remap(np.clip(result, 0, 255).astype(np.uint8),
                           map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)

        result_mask = mask
        if mask is not None:
            result_mask = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        return TransformResult(result, result_mask)
