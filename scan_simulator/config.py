"""Configuration schema and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TransformConfig:
    """Configuration for a single transform."""
    name: str
    p: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    transforms: list[TransformConfig] = field(default_factory=list)
    seed: int | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        with open(path) as f:
            data = yaml.safe_load(f)

        transforms = []
        for name, params in data.get("transforms", {}).items():
            p = params.pop("p", 1.0)
            transforms.append(TransformConfig(name=name, p=p, params=params))

        return cls(transforms=transforms, seed=data.get("seed"))

    @classmethod
    def from_dict(cls, data: dict) -> PipelineConfig:
        transforms = []
        for name, params in data.get("transforms", {}).items():
            params = dict(params)  # copy
            p = params.pop("p", 1.0)
            transforms.append(TransformConfig(name=name, p=p, params=params))
        return cls(transforms=transforms, seed=data.get("seed"))
