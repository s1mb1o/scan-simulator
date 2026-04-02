"""Transform registry."""

from .base import BaseTransform
from .paper import PaperColor, PaperTexture, InkFading, CoffeeStain, Foxing
from .damage import FoldMark, Wrinkle, EdgeWear
from .scanner import Rotation, Perspective, UnevenLight, ScannerShadow, Moire
from .camera import DefocusBlur, MotionBlur, GaussianNoise, JPEGArtifacts, ChromaticAberration
from .drawing import PenBleed, HandAnnotation, Watermark

REGISTRY: dict[str, type[BaseTransform]] = {
    # Paper & Aging
    "PaperColor": PaperColor,
    "PaperTexture": PaperTexture,
    "InkFading": InkFading,
    "CoffeeStain": CoffeeStain,
    "Foxing": Foxing,
    # Physical Damage
    "FoldMark": FoldMark,
    "Wrinkle": Wrinkle,
    "EdgeWear": EdgeWear,
    # Scanner
    "Rotation": Rotation,
    "Perspective": Perspective,
    "UnevenLight": UnevenLight,
    "ScannerShadow": ScannerShadow,
    "Moire": Moire,
    # Camera
    "DefocusBlur": DefocusBlur,
    "MotionBlur": MotionBlur,
    "GaussianNoise": GaussianNoise,
    "JPEGArtifacts": JPEGArtifacts,
    "ChromaticAberration": ChromaticAberration,
    # Drawing
    "PenBleed": PenBleed,
    "HandAnnotation": HandAnnotation,
    "Watermark": Watermark,
}

__all__ = ["BaseTransform", "REGISTRY"]
