"""Generate a visual showcase of all transforms and presets.

Creates:
  - Individual before/after pairs for each of 21 transforms
  - Full pipeline results for each of 6 presets
  - Composite grid image for quick review
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from scan_simulator.transforms import REGISTRY
from scan_simulator.presets import PRESETS
from scan_simulator.pipeline import ScanSimulator


SEED = 42
TARGET_SIZE = 800  # resize long edge for manageable output


def resize_to_fit(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def add_label(img: np.ndarray, text: str, font_scale: float = 0.7) -> np.ndarray:
    """Add a label bar at the top of the image."""
    h, w = img.shape[:2]
    bar_h = 32
    labeled = np.full((h + bar_h, w, 3), 255, dtype=np.uint8)
    labeled[bar_h:, :] = img

    # Dark background for label
    labeled[:bar_h, :] = (40, 40, 40)
    cv2.putText(labeled, text, (8, bar_h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    return labeled


def make_pair(original: np.ndarray, transformed: np.ndarray, label: str) -> np.ndarray:
    """Create side-by-side original | transformed with label."""
    left = add_label(original, "Original")
    right = add_label(transformed, label)
    # Ensure same height
    max_h = max(left.shape[0], right.shape[0])
    if left.shape[0] < max_h:
        left = np.pad(left, ((0, max_h - left.shape[0]), (0, 0), (0, 0)),
                       constant_values=255)
    if right.shape[0] < max_h:
        right = np.pad(right, ((0, max_h - right.shape[0]), (0, 0), (0, 0)),
                       constant_values=255)
    sep = np.full((max_h, 4, 3), 180, dtype=np.uint8)
    return np.hstack([left, sep, right])


def main():
    if len(sys.argv) < 2:
        print("Usage: python showcase.py <input_image> [output_dir]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("showcase_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    original = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if original is None:
        print(f"Cannot read {input_path}")
        sys.exit(1)

    original = resize_to_fit(original, TARGET_SIZE)
    h, w = original.shape[:2]
    print(f"Input: {input_path} ({w}x{h})")

    # ── Part 1: Individual transforms ──────────────────────────────
    print("\n=== Individual Transforms ===")
    transforms_dir = output_dir / "transforms"
    transforms_dir.mkdir(exist_ok=True)

    # Strong params to make effect clearly visible
    strong_params: dict[str, dict] = {
        "PaperColor":          {"intensity": 0.4, "color": [210, 195, 155]},
        "PaperTexture":        {"strength": 0.2},
        "InkFading":           {"strength": 0.35},
        "CoffeeStain":         {"count": 2, "opacity": 0.4},
        "Foxing":              {"count": 60, "radius": 4},
        "FoldMark":            {"count": 2, "shadow_width": 4, "darkness": 0.35},
        "Wrinkle":             {"count": 3, "strength": 8.0},
        "EdgeWear":            {"width": 0.08, "darkness": 0.6, "noise_strength": 0.2},
        "SurfaceWear":         {"count": 10, "intensity": 0.3},
        "Holes":               {"count": 5, "type": "mixed"},
        "Rotation":            {"angle": 4.0},
        "Perspective":         {"strength": 0.04},
        "UnevenLight":         {"falloff": 0.35},
        "ScannerShadow":       {"width": 0.12, "darkness": 0.5},
        "Moire":               {"frequency": 0.12, "strength": 0.08},
        "DefocusBlur":         {"sigma": 1.8},
        "MotionBlur":          {"size": 9},
        "GaussianNoise":       {"sigma": 22, "sp_prob": 0.003},
        "JPEGArtifacts":       {"quality": 15},
        "ChromaticAberration": {"shift": 3},
        "PenBleed":            {"kernel": 4, "darkness": 0.65},
        "HandAnnotation":      {"count": 5},
        "Watermark":           {"opacity": 0.12},
    }

    # Category grouping for the composite
    categories = {
        "Paper & Aging":     ["PaperColor", "PaperTexture", "InkFading", "CoffeeStain", "Foxing"],
        "Physical Damage":   ["FoldMark", "Wrinkle", "EdgeWear", "SurfaceWear", "Holes"],
        "Scanner Artifacts": ["Rotation", "Perspective", "UnevenLight", "ScannerShadow", "Moire"],
        "Camera Artifacts":  ["DefocusBlur", "MotionBlur", "GaussianNoise", "JPEGArtifacts", "ChromaticAberration"],
        "Drawing Artifacts": ["PenBleed", "HandAnnotation", "Watermark"],
    }

    all_pairs = []

    for cat_name, transform_names in categories.items():
        print(f"\n  {cat_name}:")
        for name in transform_names:
            cls = REGISTRY[name]
            params = strong_params.get(name, {})
            rng = np.random.default_rng(SEED)
            transform = cls(rng=rng, **params)

            result = transform(original.copy())
            out_path = transforms_dir / f"{name}.png"
            pair = make_pair(original, result.image, name)
            cv2.imwrite(str(out_path), pair)
            all_pairs.append((cat_name, name, result.image))
            print(f"    {name:25s} -> {out_path.name}")

    # ── Part 2: Preset pipelines ──────────────────────────────────
    print("\n=== Presets (3 variants each) ===")
    presets_dir = output_dir / "presets"
    presets_dir.mkdir(exist_ok=True)

    for preset_name in sorted(PRESETS.keys()):
        sim = ScanSimulator.from_preset(preset_name, seed=SEED)
        grid = sim.preview_grid(original, rows=2, cols=3)
        out_path = presets_dir / f"{preset_name.replace('-', '_')}.png"

        # Add preset name label
        labeled = add_label(grid, f"Preset: {preset_name} (6 random variants)")
        cv2.imwrite(str(out_path), labeled)
        print(f"  {preset_name:20s} -> {out_path.name}")

    # ── Part 3: Composite overview grid ────────────────────────────
    print("\n=== Composite Grid ===")
    # One row per category, showing original + all transforms in that category
    thumb_size = 300
    original_thumb = resize_to_fit(original, thumb_size)
    th, tw = original_thumb.shape[:2]

    rows = []
    for cat_name, transform_names in categories.items():
        row_images = [add_label(original_thumb, "Original", 0.5)]
        for name in transform_names:
            # Find the result
            for c, n, img in all_pairs:
                if n == name:
                    thumb = resize_to_fit(img, thumb_size)
                    # Pad to same size as original thumb
                    if thumb.shape[:2] != (th, tw):
                        canvas = np.full((th, tw, 3), 255, dtype=np.uint8)
                        ph, pw = min(th, thumb.shape[0]), min(tw, thumb.shape[1])
                        canvas[:ph, :pw] = thumb[:ph, :pw]
                        thumb = canvas
                    row_images.append(add_label(thumb, name, 0.45))
                    break

        # Pad row to max columns (6: original + 5 transforms)
        max_cols = 6
        while len(row_images) < max_cols:
            row_images.append(np.full_like(row_images[0], 255))

        # Add category label on left
        row = np.hstack([
            np.full((row_images[0].shape[0], 2, 3), 180, dtype=np.uint8).astype(np.uint8)
        ] + [np.hstack([img, np.full((img.shape[0], 2, 3), 180, dtype=np.uint8)])
             for img in row_images])

        # Category banner
        banner_h = 28
        banner = np.full((banner_h, row.shape[1], 3), 60, dtype=np.uint8)
        cv2.putText(banner, cat_name, (8, banner_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(np.vstack([banner, row]))

    composite = np.vstack(rows)
    composite_path = output_dir / "composite_all_transforms.png"
    cv2.imwrite(str(composite_path), composite)
    print(f"  Composite -> {composite_path.name} ({composite.shape[1]}x{composite.shape[0]})")

    print(f"\nAll output in: {output_dir}/")
    print(f"  transforms/  — 21 side-by-side before/after pairs")
    print(f"  presets/     — 6 preset grids (2x3 variants each)")
    print(f"  composite_all_transforms.png — overview grid")


if __name__ == "__main__":
    main()
