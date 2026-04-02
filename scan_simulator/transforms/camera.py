"""Camera/photo artifact transforms: blur, noise, compression, aberration."""

from __future__ import annotations

import cv2
import numpy as np

from .base import BaseTransform, TransformResult


class DefocusBlur(BaseTransform):
    """Spatially varying Gaussian blur (stronger at edges, simulating DOF)."""

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        sigma = self._sample("sigma", 1.0)
        h, w = image.shape[:2]

        # Center is sharp, edges are blurry
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2)
        dist = np.clip(dist, 0, 1.5)

        # Two blur levels: light center + heavy edge, blend by distance
        light = cv2.GaussianBlur(image, (0, 0), sigma * 0.3)
        heavy = cv2.GaussianBlur(image, (0, 0), sigma * 1.5)

        alpha = np.clip((dist - 0.4) / 0.6, 0, 1).astype(np.float32)[..., np.newaxis]
        result = light.astype(np.float32) * (1 - alpha) + heavy.astype(np.float32) * alpha
        return TransformResult(np.clip(result, 0, 255).astype(np.uint8), mask)


class MotionBlur(BaseTransform):
    """Directional motion blur from hand shake."""

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        size = self._sample_int("size", 5)
        if size < 3:
            return TransformResult(image, mask)
        if size % 2 == 0:
            size += 1

        angle = self.rng.uniform(0, 180)

        # Create motion blur kernel
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        # Draw a line in the kernel at the given angle
        cos_a = np.cos(np.radians(angle))
        sin_a = np.sin(np.radians(angle))
        for i in range(size):
            offset = i - center
            x = int(round(center + offset * cos_a))
            y = int(round(center + offset * sin_a))
            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1.0
        kernel /= kernel.sum() + 1e-8

        result = cv2.filter2D(image, -1, kernel)
        return TransformResult(result, mask)


class GaussianNoise(BaseTransform):
    """Additive Gaussian noise + optional salt-and-pepper."""

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        sigma = self._sample("sigma", 12.0)
        sp_prob = self._sample("sp_prob", 0.001)

        noise = self.rng.standard_normal(image.shape).astype(np.float32) * sigma
        result = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if sp_prob > 0:
            salt_mask = self.rng.random(image.shape[:2]) < sp_prob / 2
            pepper_mask = self.rng.random(image.shape[:2]) < sp_prob / 2
            result[salt_mask] = 255
            result[pepper_mask] = 0

        return TransformResult(result, mask)


class JPEGArtifacts(BaseTransform):
    """JPEG compression artifacts via encode/decode cycle."""

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        quality = self._sample_int("quality", 35)
        quality = max(5, min(95, quality))

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buf = cv2.imencode(".jpg", image, encode_param)
        result = cv2.imdecode(buf, cv2.IMREAD_COLOR)

        return TransformResult(result, mask)


class ChromaticAberration(BaseTransform):
    """Color fringing by shifting R and B channels in opposite directions."""

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        shift = self._sample_int("shift", 2)
        if shift < 1:
            return TransformResult(image, mask)

        result = image.copy()
        # Shift blue channel left, red channel right (BGR format)
        angle = self.rng.uniform(0, 2 * np.pi)
        dx = int(round(shift * np.cos(angle)))
        dy = int(round(shift * np.sin(angle)))

        h, w = image.shape[:2]
        M_blue = np.float32([[1, 0, -dx], [0, 1, -dy]])
        M_red = np.float32([[1, 0, dx], [0, 1, dy]])

        result[:, :, 0] = cv2.warpAffine(image[:, :, 0], M_blue, (w, h),
                                          borderMode=cv2.BORDER_REPLICATE)
        result[:, :, 2] = cv2.warpAffine(image[:, :, 2], M_red, (w, h),
                                          borderMode=cv2.BORDER_REPLICATE)

        return TransformResult(result, mask)
