from __future__ import annotations

import json
from pathlib import Path


def build_label_map(class_names: list[str]) -> dict[str, int]:
    ordered = sorted(set(class_names))
    return {name: idx for idx, name in enumerate(ordered)}


def save_label_map(label_map: dict[str, int], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2, sort_keys=False)


def load_label_map(path: str | Path) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def inverse(label_map: dict[str, int]) -> dict[int, str]:
    return {v: k for k, v in label_map.items()}
