#!/usr/bin/env python
"""Converte os records preparados (script 01) para o layout YOLO-txt.

Uso:
    python scripts/02_build_yolo_dataset.py --base-config configs/base.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from logoloc.train.train_yolo import prepare_yolo_data
from logoloc.config import load_project_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_project_config(args.base_config)
    data_yaml_path, label_map = prepare_yolo_data(cfg, force_prepare=args.force, force_convert=args.force)
    logger.info("Dataset YOLO pronto. data.yaml em: %s", data_yaml_path)
    logger.info("Classes (%d): %s", len(label_map), list(label_map.keys()))


if __name__ == "__main__":
    main()
