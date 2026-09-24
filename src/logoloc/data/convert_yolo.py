from __future__ import annotations

import logging
import shutil
from pathlib import Path

import yaml

from .label_map import inverse
from .records import ImageRecord

logger = logging.getLogger(__name__)

SPLIT_DIRNAME = {"P1": "train", "P2": "val", "P3": "test"}


def _flat_name(record: ImageRecord) -> str:
    return record.image_id.replace("/", "__").replace("\\", "__")


def convert_split_to_yolo(
    records: list[ImageRecord],
    split: str,
    label_map: dict[str, int],
    output_dir: str | Path,
    force: bool = False,
) -> None:
    output_dir = Path(output_dir)
    subdir = SPLIT_DIRNAME[split]
    images_dir = output_dir / "images" / subdir
    labels_dir = output_dir / "labels" / subdir

    if images_dir.exists() and not force:
        logger.info("YOLO split '%s' ja' convertido em %s (force=False), pulando.", split, images_dir)
        return

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for record in records:
        stem = Path(_flat_name(record)).stem
        dst_image = images_dir / f"{stem}.jpg"
        shutil.copy2(record.image_path, dst_image)

        lines = []
        for inst in record.instances:
            b = inst.bbox
            cx = (b.x + b.w / 2.0) / record.width
            cy = (b.y + b.h / 2.0) / record.height
            w = b.w / record.width
            h = b.h / record.height
            class_id = label_map[inst.class_name]
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        with open(labels_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    logger.info("YOLO split '%s': %d imagens convertidas em %s", split, len(records), images_dir)


def write_data_yaml(label_map: dict[str, int], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    id_to_name = inverse(label_map)
    names = [id_to_name[i] for i in range(len(id_to_name))]
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(names)},
    }
    data_yaml_path = output_dir / "data.yaml"
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    logger.info("data.yaml escrito em %s", data_yaml_path)
    return data_yaml_path


def convert_all_to_yolo(
    records_by_split: dict[str, list[ImageRecord]],
    label_map: dict[str, int],
    output_dir: str | Path,
    force: bool = False,
) -> Path:
    for split, records in records_by_split.items():
        convert_split_to_yolo(records, split, label_map, output_dir, force=force)
    return write_data_yaml(label_map, output_dir)
