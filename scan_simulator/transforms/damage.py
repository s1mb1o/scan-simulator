"""Physical damage transforms: fold marks, wrinkles, edge wear, holes."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline

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
            vertical = self.rng.random() < 0.5
            if vertical:
                x = int(self.rng.integers(w // 5, 4 * w // 5))
                pt1, pt2 = (x, 0), (x + int(self.rng.integers(-w // 20, w // 20)), h)
            else:
                y = int(self.rng.integers(h // 5, 4 * h // 5))
                pt1, pt2 = (0, y), (w, y + int(self.rng.integers(-h // 20, h // 20)))

            shadow = np.zeros((h, w), dtype=np.float32)
            cv2.line(shadow, pt1, pt2, 1.0, shadow_width, lineType=cv2.LINE_AA)
            shadow = gaussian_filter(shadow, sigma=shadow_width * 0.6)

            darkness = self._sample("darkness", 0.25)
            result = result.astype(np.float32)
            result -= shadow[..., np.newaxis] * 255 * darkness

            if vertical:
                shadow_shift = np.roll(shadow, shadow_width + 1, axis=1)
            else:
                shadow_shift = np.roll(shadow, shadow_width + 1, axis=0)
            result += shadow_shift[..., np.newaxis] * 255 * darkness * 0.3

            result = np.clip(result, 0, 255).astype(np.uint8)

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

        dx = np.zeros((h, w), dtype=np.float32)
        dy = np.zeros((h, w), dtype=np.float32)

        for _ in range(count):
            cx = self.rng.integers(w // 6, 5 * w // 6)
            cy = self.rng.integers(h // 6, 5 * h // 6)
            length = self.rng.integers(min(h, w) // 6, min(h, w) // 2)
            angle = self.rng.uniform(0, np.pi)

            y, x = np.ogrid[:h, :w]
            nx, ny = np.cos(angle), np.sin(angle)
            dist_along = (x - cx) * nx + (y - cy) * ny
            dist_perp = np.abs((x - cx) * (-ny) + (y - cy) * nx)

            along_mask = np.exp(-(dist_along ** 2) / (2 * (length / 3) ** 2))
            perp_profile = np.exp(-(dist_perp ** 2) / (2 * (strength * 2) ** 2))
            wrinkle = along_mask * perp_profile * strength

            dx += (wrinkle * (-ny)).astype(np.float32)
            dy += (wrinkle * nx).astype(np.float32)

        rows, cols = np.mgrid[:h, :w].astype(np.float32)
        map_x = cols + dx
        map_y = rows + dy

        result = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)

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


def _wavy_edge_profile(length: int, n_points: int, amplitude: float,
                       rng: np.random.Generator) -> np.ndarray:
    """Generate a smooth wavy edge profile via cubic spline.

    Returns an array of shape (length,) with values in [0, amplitude].
    """
    xs = np.linspace(0, length - 1, n_points)
    ys = rng.uniform(0, amplitude, n_points)
    # Clamp endpoints to small values for natural look
    ys[0] = rng.uniform(0, amplitude * 0.3)
    ys[-1] = rng.uniform(0, amplitude * 0.3)
    cs = CubicSpline(xs, ys, bc_type='natural')
    return np.clip(cs(np.arange(length)), 0, amplitude).astype(np.float32)


class EdgeWear(BaseTransform):
    """Organic edge wear with wavy boundary, darkening, and scattered noise.

    Uses spline-interpolated wavy edge profiles (inspired by Augraphy's
    BadPhotoCopy) instead of a uniform rectangular gradient. Each edge gets
    an independent random profile.
    """

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        width_frac = self._sample("width", 0.06)
        darkness = self._sample("darkness", 0.5)
        noise_strength = self._sample("noise_strength", 0.15)
        h, w = image.shape[:2]

        max_width = int(max(w, h) * width_frac)
        n_ctrl = max(6, max_width // 10)  # control points for spline

        # Build per-edge wavy width profiles
        wear_mask = np.zeros((h, w), dtype=np.float32)

        # Top edge: profile along x-axis, wear extends downward
        profile_top = _wavy_edge_profile(w, n_ctrl, max_width, self.rng)
        for x in range(w):
            depth = int(profile_top[x])
            if depth > 0:
                ramp = np.linspace(1.0, 0.0, depth)
                wear_mask[:depth, x] = np.maximum(wear_mask[:depth, x], ramp)

        # Bottom edge
        profile_bot = _wavy_edge_profile(w, n_ctrl, max_width, self.rng)
        for x in range(w):
            depth = int(profile_bot[x])
            if depth > 0:
                ramp = np.linspace(1.0, 0.0, depth)
                wear_mask[h - depth:h, x] = np.maximum(
                    wear_mask[h - depth:h, x], ramp[::-1])

        # Left edge
        profile_left = _wavy_edge_profile(h, n_ctrl, max_width, self.rng)
        for y in range(h):
            depth = int(profile_left[y])
            if depth > 0:
                ramp = np.linspace(1.0, 0.0, depth)
                wear_mask[y, :depth] = np.maximum(wear_mask[y, :depth], ramp)

        # Right edge
        profile_right = _wavy_edge_profile(h, n_ctrl, max_width, self.rng)
        for y in range(h):
            depth = int(profile_right[y])
            if depth > 0:
                ramp = np.linspace(1.0, 0.0, depth)
                wear_mask[y, w - depth:w] = np.maximum(
                    wear_mask[y, w - depth:w], ramp[::-1])

        # Add clustered noise in wear zone (darker speckles where paper is worn)
        noise = self.rng.standard_normal((h, w)).astype(np.float32)
        noise = gaussian_filter(noise, sigma=2.0)
        noise = np.clip(noise, -1, 1) * noise_strength
        wear_mask = np.clip(wear_mask + noise * wear_mask, 0, 1)

        # Smooth the mask for organic blending
        wear_mask = gaussian_filter(wear_mask, sigma=max(1.5, max_width * 0.08))

        # Apply darkening + slight yellowing in worn areas
        result = image.astype(np.float32)
        # Darken
        result *= (1.0 - wear_mask[..., np.newaxis] * darkness)
        # Slight warm tint in worn areas
        tint = np.array([0.92, 0.97, 1.0], dtype=np.float32)  # BGR: slightly warm
        result *= (1.0 + wear_mask[..., np.newaxis] * (tint - 1.0) * 0.3)

        return TransformResult(np.clip(result, 0, 255).astype(np.uint8), mask)


class SurfaceWear(BaseTransform):
    """Surface abrasion and scuffing — scattered light patches and scratches.

    Simulates paper being rubbed, folded repeatedly, or dragged across a
    surface. Creates light scratches and faded patches across the document.
    """

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        count = self._sample_int("count", 8)
        intensity = self._sample("intensity", 0.25)
        h, w = image.shape[:2]

        wear = np.zeros((h, w), dtype=np.float32)

        for _ in range(count):
            kind = self.rng.choice(["scratch", "scuff", "rub"])

            if kind == "scratch":
                # Thin linear scratch
                x0 = int(self.rng.integers(0, w))
                y0 = int(self.rng.integers(0, h))
                angle = self.rng.uniform(0, np.pi)
                length = int(self.rng.integers(min(h, w) // 8, min(h, w) // 2))
                x1 = int(x0 + length * np.cos(angle))
                y1 = int(y0 + length * np.sin(angle))
                thickness = int(self.rng.integers(1, 3))
                scratch_mask = np.zeros((h, w), dtype=np.float32)
                cv2.line(scratch_mask, (x0, y0), (x1, y1), 1.0,
                         thickness, lineType=cv2.LINE_AA)
                scratch_mask = gaussian_filter(scratch_mask, sigma=1.0)
                wear += scratch_mask * self.rng.uniform(0.5, 1.0)

            elif kind == "scuff":
                # Elliptical faded patch
                cx = int(self.rng.integers(w // 8, 7 * w // 8))
                cy = int(self.rng.integers(h // 8, 7 * h // 8))
                rx = int(self.rng.integers(10, max(11, w // 6)))
                ry = int(self.rng.integers(int(rx * 0.4), int(rx * 1.2) + 1))
                angle = self.rng.uniform(0, 360)

                scuff_mask = np.zeros((h, w), dtype=np.float32)
                cv2.ellipse(scuff_mask, (cx, cy), (rx, ry), angle,
                            0, 360, 1.0, -1, lineType=cv2.LINE_AA)
                scuff_mask = gaussian_filter(scuff_mask, sigma=max(rx, ry) * 0.3)
                wear += scuff_mask * self.rng.uniform(0.3, 0.8)

            else:  # rub — broad directional fade
                y_start = int(self.rng.integers(0, h))
                band_h = int(self.rng.integers(h // 10, h // 3))
                y_end = min(h, y_start + band_h)
                rub_mask = np.zeros((h, w), dtype=np.float32)
                rub_mask[y_start:y_end, :] = 1.0
                rub_mask = gaussian_filter(rub_mask, sigma=band_h * 0.4)
                wear += rub_mask * self.rng.uniform(0.15, 0.4)

        wear = np.clip(wear, 0, 1)

        # Wear lightens the image (fades ink)
        result = image.astype(np.float32)
        result += wear[..., np.newaxis] * (255 - result) * intensity

        return TransformResult(np.clip(result, 0, 255).astype(np.uint8), mask)


class Holes(BaseTransform):
    """Punch holes, worm holes, and torn spots.

    Generates realistic holes of various types: clean punch holes (binder),
    small worm/insect holes, and irregular torn spots. Holes are rendered
    as white (paper removed) with optional shadow rings.
    """

    def apply(self, image: np.ndarray, mask: np.ndarray | None = None) -> TransformResult:
        hole_type = self.params.get("type", "mixed")
        count = self._sample_int("count", 3)
        h, w = image.shape[:2]

        result = image.copy()
        result_mask = mask.copy() if mask is not None else None

        # Background color visible through holes (scanner bed = dark gray/black)
        bg_color = np.array(self.params.get("bg_color", [30, 30, 30]),
                            dtype=np.uint8)

        for _ in range(count):
            if hole_type == "mixed":
                kind = self.rng.choice(["punch", "worm", "torn"])
            else:
                kind = hole_type

            if kind == "punch":
                self._draw_punch_hole(result, result_mask, h, w, bg_color)
            elif kind == "worm":
                self._draw_worm_hole(result, result_mask, h, w, bg_color)
            else:
                self._draw_torn_spot(result, result_mask, h, w, bg_color)

        return TransformResult(result, result_mask)

    def _draw_punch_hole(self, image: np.ndarray, mask: np.ndarray | None,
                         h: int, w: int, bg: np.ndarray):
        """Clean circular punch hole, typically near left/top margin."""
        # Punch holes are usually along one edge
        edge = self.rng.choice(["left", "top"])
        margin = int(min(h, w) * 0.06)
        radius = int(self.rng.integers(
            max(3, min(h, w) // 50), max(4, min(h, w) // 25)))

        if edge == "left":
            cx = int(self.rng.integers(margin, margin + radius * 3))
            cy = int(self.rng.integers(h // 6, 5 * h // 6))
        else:
            cx = int(self.rng.integers(w // 6, 5 * w // 6))
            cy = int(self.rng.integers(margin, margin + radius * 3))

        # Ring shadow (slightly larger, offset)
        shadow_offset = max(1, radius // 8)
        shadow = np.zeros((h, w), dtype=np.float32)
        cv2.circle(shadow, (cx + shadow_offset, cy + shadow_offset),
                   radius + 2, 1.0, -1, lineType=cv2.LINE_AA)
        shadow = gaussian_filter(shadow, sigma=radius * 0.3)
        image[:] = np.clip(
            image.astype(np.float32) - shadow[..., np.newaxis] * 60,
            0, 255).astype(np.uint8)

        # Ring impression (raised edge around hole)
        ring = np.zeros((h, w), dtype=np.float32)
        cv2.circle(ring, (cx, cy), radius + max(1, radius // 5),
                   1.0, max(1, radius // 4), lineType=cv2.LINE_AA)
        ring = gaussian_filter(ring, sigma=0.8)
        image[:] = np.clip(
            image.astype(np.float32) - ring[..., np.newaxis] * 30,
            0, 255).astype(np.uint8)

        # The hole itself
        cv2.circle(image, (cx, cy), radius, bg.tolist(), -1, lineType=cv2.LINE_AA)
        if mask is not None:
            cv2.circle(mask, (cx, cy), radius, 0, -1, lineType=cv2.LINE_AA)

    def _draw_worm_hole(self, image: np.ndarray, mask: np.ndarray | None,
                        h: int, w: int, bg: np.ndarray):
        """Small irregular hole from insects/worms — tiny, slightly ragged."""
        cx = int(self.rng.integers(w // 10, 9 * w // 10))
        cy = int(self.rng.integers(h // 10, 9 * h // 10))
        radius = int(self.rng.integers(2, max(3, min(h, w) // 80)))

        # Irregular shape: draw multiple overlapping tiny circles
        n_blobs = int(self.rng.integers(2, 5))
        hole_mask = np.zeros((h, w), dtype=np.uint8)
        for _ in range(n_blobs):
            dx = int(self.rng.integers(-radius, radius + 1))
            dy = int(self.rng.integers(-radius, radius + 1))
            r = max(1, int(self.rng.integers(1, radius + 1)))
            cv2.circle(hole_mask, (cx + dx, cy + dy), r, 255, -1)

        # Slight brown edge around hole
        dilated = cv2.dilate(hole_mask, np.ones((3, 3), np.uint8), iterations=1)
        edge_ring = dilated & ~hole_mask
        brown = np.array([40, 90, 140], dtype=np.float32)
        edge_f = (edge_ring > 0).astype(np.float32)
        image[:] = np.clip(
            image.astype(np.float32) * (1 - edge_f[..., np.newaxis] * 0.5)
            + brown * edge_f[..., np.newaxis] * 0.5,
            0, 255).astype(np.uint8)

        # The hole
        image[hole_mask > 0] = bg
        if mask is not None:
            mask[hole_mask > 0] = 0

    def _draw_torn_spot(self, image: np.ndarray, mask: np.ndarray | None,
                        h: int, w: int, bg: np.ndarray):
        """Irregular torn/ripped area — jagged outline, paper fibers visible."""
        cx = int(self.rng.integers(w // 8, 7 * w // 8))
        cy = int(self.rng.integers(h // 8, 7 * h // 8))
        base_radius = int(self.rng.integers(
            max(4, min(h, w) // 40), max(5, min(h, w) // 15)))

        # Jagged polygon via angular sampling with random radii
        n_verts = int(self.rng.integers(8, 16))
        angles = np.sort(self.rng.uniform(0, 2 * np.pi, n_verts))
        radii = base_radius * self.rng.uniform(0.4, 1.0, n_verts)
        pts = np.array([
            (int(cx + r * np.cos(a)), int(cy + r * np.sin(a)))
            for a, r in zip(angles, radii)
        ], dtype=np.int32)

        # Fiber/fringe effect: dilate the torn region slightly with noise
        torn_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(torn_mask, [pts], 255)

        # Ragged edge — erode + noise for paper fiber look
        fringe = cv2.dilate(torn_mask, np.ones((3, 3), np.uint8), iterations=1)
        fringe = fringe & ~torn_mask
        fringe_noise = self.rng.random((h, w)) < 0.5
        fringe = fringe & (fringe_noise.astype(np.uint8) * 255)

        # Darken fringe (torn paper fibers)
        fringe_f = (fringe > 0).astype(np.float32)
        image[:] = np.clip(
            image.astype(np.float32) * (1 - fringe_f[..., np.newaxis] * 0.4),
            0, 255).astype(np.uint8)

        # Shadow under torn area
        shadow = gaussian_filter((torn_mask > 0).astype(np.float32), sigma=3)
        shadow_shifted = np.roll(np.roll(shadow, 2, axis=0), 2, axis=1)
        image[:] = np.clip(
            image.astype(np.float32)
            - shadow_shifted[..., np.newaxis] * 40 * (1 - (torn_mask > 0).astype(np.float32))[..., np.newaxis],
            0, 255).astype(np.uint8)

        # The hole
        image[torn_mask > 0] = bg
        if mask is not None:
            mask[torn_mask > 0] = 0
