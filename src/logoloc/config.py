from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclasses.dataclass
class PathsConfig:
    raw_dir: str = "data/raw/FlickrLogos-32"
    processed_dir: str = "data/processed"
    cache_dir: str = "data/cache"
    outputs_dir: str = "outputs"


@dataclasses.dataclass
class CropConfig:
    strategy: str = "tight_pad"
    padding_ratio: float = 0.10
    output_size: int = 224


@dataclasses.dataclass
class AugmentationConfig:
    enabled: bool = True
    horizontal_flip_prob: float = 0.5
    scale_range: tuple[float, float] = (0.8, 1.2)
    brightness_delta: float = 0.2
    contrast_delta: float = 0.2
    saturation_delta: float = 0.2
    hue_delta: float = 0.02
    copies_per_image: int = 3


@dataclasses.dataclass
class MetricsConfig:
    iou_thresholds: tuple[float, ...] = (0.5, 0.75)
    primary_iou_threshold: float = 0.5
    background_iou_floor: float = 0.1


@dataclasses.dataclass
class DatasetConfig:
    name: str = "FlickrLogos-32"
    num_classes: int = 32
    no_logo_label: str = "no-logo"
    images_per_class_train: int = 10
    images_per_class_val: int = 30
    images_per_class_test: int = 30
    no_logo_images_val: int = 3000
    no_logo_images_test: int = 3000


@dataclasses.dataclass
class ProjectConfig:
    seed: int = 42
    paths: PathsConfig = dataclasses.field(default_factory=PathsConfig)
    dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)
    crop: CropConfig = dataclasses.field(default_factory=CropConfig)
    augmentation: AugmentationConfig = dataclasses.field(default_factory=AugmentationConfig)
    metrics: MetricsConfig = dataclasses.field(default_factory=MetricsConfig)

    def resolved_path(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else REPO_ROOT / p


def _merge_dataclass(instance: Any, data: dict) -> Any:
    for key, value in data.items():
        if not hasattr(instance, key):
            raise ValueError(f"Campo de config desconhecido: '{key}' em {type(instance).__name__}")
        current = getattr(instance, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _merge_dataclass(current, value)
        elif isinstance(current, tuple) and isinstance(value, list):
            setattr(instance, key, tuple(value))
        else:
            setattr(instance, key, value)
    return instance


def load_yaml_into(instance: Any, yaml_path: str | Path) -> Any:
    yaml_path = Path(yaml_path)
    if not yaml_path.is_absolute():
        yaml_path = REPO_ROOT / yaml_path
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _merge_dataclass(instance, data)


def load_project_config(*yaml_paths: str | Path) -> ProjectConfig:
    cfg = ProjectConfig()
    for p in yaml_paths:
        load_yaml_into(cfg, p)
    return cfg


def load_dataclass_yaml(cls: type, *yaml_paths: str | Path) -> Any:
    instance = cls()
    for p in yaml_paths:
        load_yaml_into(instance, p)
    return instance
