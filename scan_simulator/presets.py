"""Built-in preset configurations."""

from __future__ import annotations

from .config import PipelineConfig

PRESETS: dict[str, dict] = {
    "scan-clean": {
        "transforms": {
            "PaperColor":    {"p": 0.3, "intensity": [0.05, 0.15]},
            "Rotation":      {"p": 0.5, "angle": [-1.0, 1.0]},
            "UnevenLight":   {"p": 0.4, "falloff": [0.05, 0.15]},
            "GaussianNoise": {"p": 0.3, "sigma": [3, 10]},
            "JPEGArtifacts": {"p": 0.4, "quality": [40, 70]},
        }
    },
    "scan-heavy": {
        "transforms": {
            "PaperColor":     {"p": 0.8, "intensity": [0.2, 0.5], "color": [220, 200, 160]},
            "PaperTexture":   {"p": 0.5, "strength": [0.1, 0.2]},
            "InkFading":      {"p": 0.5, "strength": [0.15, 0.35]},
            "FoldMark":       {"p": 0.6, "count": [1, 3]},
            "EdgeWear":       {"p": 0.4, "width": [0.04, 0.08]},
            "SurfaceWear":    {"p": 0.3, "count": [3, 6], "intensity": [0.1, 0.2]},
            "Rotation":       {"p": 0.7, "angle": [-3.0, 3.0]},
            "Perspective":    {"p": 0.3, "strength": [0.01, 0.03]},
            "UnevenLight":    {"p": 0.6, "falloff": [0.15, 0.35]},
            "ScannerShadow":  {"p": 0.4, "width": [0.05, 0.12]},
            "GaussianNoise":  {"p": 0.5, "sigma": [8, 20]},
            "JPEGArtifacts":  {"p": 0.6, "quality": [20, 45]},
            "CoffeeStain":    {"p": 0.2, "count": [1, 2]},
            "Foxing":         {"p": 0.15, "count": [10, 50]},
        }
    },
    "photo-indoor": {
        "transforms": {
            "PaperColor":          {"p": 0.4, "intensity": [0.05, 0.2]},
            "Perspective":         {"p": 0.7, "strength": [0.02, 0.05]},
            "Rotation":            {"p": 0.6, "angle": [-5.0, 5.0]},
            "UnevenLight":         {"p": 0.8, "falloff": [0.15, 0.35]},
            "DefocusBlur":         {"p": 0.5, "sigma": [0.5, 1.5]},
            "MotionBlur":          {"p": 0.3, "size": [3, 7]},
            "GaussianNoise":       {"p": 0.6, "sigma": [10, 25]},
            "ChromaticAberration": {"p": 0.3, "shift": [1, 2]},
            "JPEGArtifacts":       {"p": 0.5, "quality": [25, 55]},
        }
    },
    "photo-outdoor": {
        "transforms": {
            "Perspective":         {"p": 0.5, "strength": [0.01, 0.03]},
            "Rotation":            {"p": 0.5, "angle": [-3.0, 3.0]},
            "UnevenLight":         {"p": 0.6, "falloff": [0.1, 0.25]},
            "DefocusBlur":         {"p": 0.3, "sigma": [0.3, 1.0]},
            "GaussianNoise":       {"p": 0.4, "sigma": [5, 15]},
            "ChromaticAberration": {"p": 0.2, "shift": [1, 2]},
            "JPEGArtifacts":       {"p": 0.4, "quality": [35, 65]},
        }
    },
    "photocopy": {
        "transforms": {
            "InkFading":      {"p": 0.7, "strength": [0.2, 0.4]},
            "PenBleed":       {"p": 0.6, "kernel": [2, 4], "darkness": [0.4, 0.7]},
            "PaperTexture":   {"p": 0.6, "strength": [0.1, 0.2]},
            "Rotation":       {"p": 0.5, "angle": [-2.0, 2.0]},
            "UnevenLight":    {"p": 0.5, "falloff": [0.1, 0.25]},
            "Moire":          {"p": 0.4, "frequency": [0.08, 0.15], "strength": [0.03, 0.08]},
            "GaussianNoise":  {"p": 0.5, "sigma": [5, 15]},
            "JPEGArtifacts":  {"p": 0.5, "quality": [25, 50]},
        }
    },
    "archive": {
        "transforms": {
            "PaperColor":    {"p": 0.9, "intensity": [0.3, 0.5], "color": [200, 190, 150]},
            "Foxing":        {"p": 0.6, "count": [20, 80], "radius": [2, 6]},
            "CoffeeStain":   {"p": 0.3, "count": [1, 3]},
            "InkFading":     {"p": 0.7, "strength": [0.2, 0.4]},
            "FoldMark":      {"p": 0.5, "count": [1, 2]},
            "EdgeWear":      {"p": 0.7, "width": [0.05, 0.1], "darkness": [0.4, 0.7]},
            "SurfaceWear":   {"p": 0.5, "count": [5, 12], "intensity": [0.15, 0.3]},
            "Holes":         {"p": 0.3, "count": [1, 4], "type": "worm"},
            "PaperTexture":  {"p": 0.5, "strength": [0.1, 0.2]},
            "Rotation":      {"p": 0.4, "angle": [-2.0, 2.0]},
            "GaussianNoise": {"p": 0.3, "sigma": [5, 12]},
            "JPEGArtifacts": {"p": 0.5, "quality": [30, 55]},
        }
    },
}


def get_preset(name: str) -> PipelineConfig:
    """Load a built-in preset by name."""
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PipelineConfig.from_dict(PRESETS[name])
