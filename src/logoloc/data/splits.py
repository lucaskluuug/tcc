from __future__ import annotations

import logging
from pathlib import Path

from ..config import ProjectConfig
from . import raw_adapter
from .cache import cached
from .label_map import build_label_map, save_label_map
from .records import ImageRecord

logger = logging.getLogger(__name__)


class SplitValidationError(RuntimeError):
    pass


def validate_split(records: list[ImageRecord], split: str, cfg: ProjectConfig) -> None:
    ds = cfg.dataset
    positive = [r for r in records if not r.is_no_logo]
    no_logo = [r for r in records if r.is_no_logo]

    per_class: dict[str, int] = {}
    for r in positive:
        for name in r.class_names:
            per_class[name] = per_class.get(name, 0) + 1

    expected_per_class = {
        "P1": ds.images_per_class_train,
        "P2": ds.images_per_class_val,
        "P3": ds.images_per_class_test,
    }[split]
    expected_no_logo = {"P1": 0, "P2": ds.no_logo_images_val, "P3": ds.no_logo_images_test}[split]

    problems = []
    if len(per_class) not in (0, ds.num_classes):
        problems.append(f"esperava {ds.num_classes} classes com exemplos positivos, achei {len(per_class)}")
    for name, count in per_class.items():
        if count != expected_per_class:
            problems.append(f"classe '{name}': {count} imagens (esperado {expected_per_class})")
    if expected_no_logo and len(no_logo) != expected_no_logo:
        problems.append(f"{len(no_logo)} imagens sem logotipo (esperado {expected_no_logo})")

    if problems:
        msg = f"Split {split} nao bate com a particao oficial esperada:\n  - " + "\n  - ".join(problems)
        logger.warning(msg)
    else:
        logger.info("Split %s validado: %d classes, %d no-logo, tudo conforme esperado.", split, len(per_class), len(no_logo))


def prepare_dataset(cfg: ProjectConfig, force: bool = False) -> dict[str, list[ImageRecord]]:
    raw_dir = cfg.resolved_path(cfg.paths.raw_dir)
    cache_dir = cfg.resolved_path(cfg.paths.cache_dir)
    processed_dir = cfg.resolved_path(cfg.paths.processed_dir)

    records_by_split = cached(
        cache_path=cache_dir / "records_by_split.pkl",
        compute_fn=lambda: raw_adapter.load_all_splits(raw_dir),
        force=force,
    )

    for split, records in records_by_split.items():
        validate_split(records, split, cfg)

    class_names = raw_adapter.discover_class_names(raw_dir)
    label_map = build_label_map(class_names)
    save_label_map(label_map, processed_dir / "label_map.json")
    logger.info("label_map.json salvo com %d classes.", len(label_map))

    return records_by_split
