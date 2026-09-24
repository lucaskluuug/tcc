#!/usr/bin/env python
"""Parseia, valida particoes oficiais, aumenta P1 e gera crops de P3.

Uso:
    python scripts/01_prepare_dataset.py --base-config configs/base.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from logoloc.config import load_project_config
from logoloc.data.augment import generate_augmented_p1
from logoloc.data.crops import generate_gt_crops, tight_variant
from logoloc.data.splits import prepare_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--force", action="store_true", help="recomputa tudo, ignora cache existente")
    args = parser.parse_args()

    cfg = load_project_config(args.base_config)

    records_by_split = prepare_dataset(cfg, force=args.force)
    for split, records in records_by_split.items():
        n_positive = sum(1 for r in records if not r.is_no_logo)
        logger.info("%s: %d imagens (%d com logotipo, %d sem)", split, len(records), n_positive, len(records) - n_positive)

    if cfg.augmentation.enabled:
        p1_augmented = generate_augmented_p1(
            records_by_split["P1"],
            cfg.augmentation,
            output_dir=cfg.resolved_path(cfg.paths.processed_dir) / "p1_augmented",
            seed=cfg.seed,
            force=args.force,
        )
        logger.info("P1 apos augmentation: %d imagens", len(p1_augmented))

    crops_p3 = generate_gt_crops(
        records_by_split["P3"],
        cfg.crop,
        output_dir=cfg.resolved_path(cfg.paths.processed_dir) / "p3_gt_crops",
        force=args.force,
    )
    logger.info("Crops do Fluxo 1 (P3, %s + %.0f%% padding): %d gerados", cfg.crop.strategy, cfg.crop.padding_ratio * 100, len(crops_p3))

    crops_p3_tight = generate_gt_crops(
        records_by_split["P3"],
        tight_variant(cfg.crop),
        output_dir=cfg.resolved_path(cfg.paths.processed_dir) / "p3_gt_crops_tight",
        force=args.force,
    )
    logger.info("Crops do Fluxo 1, variante sem padding (sensibilidade): %d gerados", len(crops_p3_tight))

    logger.info("Etapa 1 concluida. Proximo passo: scripts/04_train_faster_rcnn.py ou scripts/05_train_yolo.py")


if __name__ == "__main__":
    main()
