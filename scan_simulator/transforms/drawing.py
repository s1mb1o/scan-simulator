"""Drawing artifact transforms: pen bleed, annotations, watermarks."""

from __future__ import annotations

import cv2
import numpy as np

from .base import BaseTransform, TransformResult


class PenBleed(BaseTransform):
    """Ink bleeding: dilate dark lines + blur edges."""

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        kernel_size = self._sample_int("kernel", 3)
        blur_sigma = self._sample("blur", 1.5)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect dark pixels (ink)
        ink_mask = (gray < 128).astype(np.uint8)

        # Dilate ink
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (kernel_size, kernel_size))
        dilated = cv2.dilate(ink_mask, kernel, iterations=1)

        # New ink pixels (bleed area)
        bleed = dilated & ~ink_mask

        # Apply: darken bleed area with some blur
        result = image.copy().astype(np.float32)
        bleed_f = cv2.GaussianBlur(bleed.astype(np.float32),
                                    (0, 0), blur_sigma)
        bleed_f = np.clip(bleed_f, 0, 1)

        # Darken proportionally
        darkness = self._sample("darkness", 0.6)
        result *= (1.0 - bleed_f[..., np.newaxis] * darkness)

        return TransformResult(np.clip(result, 0, 255).astype(np.uint8), mask)


class HandAnnotation(BaseTransform):
    """Random pen marks/scribbles in margins."""

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        count = self._sample_int("count", 3)
        h, w = image.shape[:2]
        result = image.copy()

        margin = int(min(h, w) * 0.08)

        for _ in range(count):
            # Choose margin area
            side = self.rng.choice(["top", "bottom", "left", "right"])
            color = (
                int(self.rng.integers(0, 80)),
                int(self.rng.integers(0, 80)),
                int(self.rng.integers(80, 200)),
            )
            thickness = int(self.rng.integers(1, 3))

            if side == "top":
                y0 = int(self.rng.integers(5, margin))
                x0 = int(self.rng.integers(margin, w - margin))
            elif side == "bottom":
                y0 = int(self.rng.integers(h - margin, h - 5))
                x0 = int(self.rng.integers(margin, w - margin))
            elif side == "left":
                y0 = int(self.rng.integers(margin, h - margin))
                x0 = int(self.rng.integers(5, margin))
            else:
                y0 = int(self.rng.integers(margin, h - margin))
                x0 = int(self.rng.integers(w - margin, w - 5))

            # Draw a short scribble (polyline with 3-6 points)
            n_pts = int(self.rng.integers(3, 7))
            pts = [(x0, y0)]
            for _ in range(n_pts - 1):
                dx = int(self.rng.integers(-20, 20))
                dy = int(self.rng.integers(-10, 10))
                pts.append((pts[-1][0] + dx, pts[-1][1] + dy))

            pts_arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(result, [pts_arr], False, color, thickness,
                          lineType=cv2.LINE_AA)

        return TransformResult(result, mask)


class Watermark(BaseTransform):
    """Semi-transparent diagonal text watermark."""

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        opacity = self._sample("opacity", 0.08)
        h, w = image.shape[:2]

        texts = ["COPY", "DRAFT", "ARCHIVE", "SAMPLE", "REVIEW",
                 "CONFIDENTIAL", "DO NOT COPY"]
        text = self.rng.choice(texts)

        # Create text overlay
        overlay = np.zeros_like(image, dtype=np.float32)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = min(w, h) / 300
        thickness = max(2, int(font_scale * 2))

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        cx = (w - text_size[0]) // 2
        cy = (h + text_size[1]) // 2

        cv2.putText(overlay, text, (cx, cy), font, font_scale,
                    (128, 128, 128), thickness, cv2.LINE_AA)

        # Rotate overlay
        angle = self.rng.uniform(-45, -30)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        overlay = cv2.warpAffine(overlay, M, (w, h))

        # Blend
        text_mask = (overlay.sum(axis=2) > 0).astype(np.float32)[..., np.newaxis]
        result = image.astype(np.float32)
        result = result * (1 - text_mask * opacity) + overlay * text_mask * opacity * 2
        return TransformResult(np.clip(result, 0, 255).astype(np.uint8), mask)
