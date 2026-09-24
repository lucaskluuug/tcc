from __future__ import annotations

import dataclasses
import logging
from pathlib import Path

from PIL import Image

from ..config import CropConfig
from .cache import cached
from .records import BBox, ImageRecord

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class CropRecord:
    crop_path: Path
    class_name: str
    source_image_id: str
    instance_index: int
    source_bbox: BBox
    source_image_size: tuple[int, int]


def _compute_crop_box(bbox: BBox, img_w: int, img_h: int, cfg: CropConfig) -> tuple[int, int, int, int]:
    if cfg.strategy == "tight":
        x1, y1, x2, y2 = bbox.x, bbox.y, bbox.x2, bbox.y2
    elif cfg.strategy == "tight_pad":
        pad = cfg.padding_ratio * max(bbox.w, bbox.h)
        x1, y1, x2, y2 = bbox.x - pad, bbox.y - pad, bbox.x2 + pad, bbox.y2 + pad
    elif cfg.strategy == "square":
        side = max(bbox.w, bbox.h) * (1.0 + 2.0 * cfg.padding_ratio)
        cx, cy = bbox.x + bbox.w / 2.0, bbox.y + bbox.h / 2.0
        x1, y1, x2, y2 = cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0
    else:
        raise ValueError(f"Estrategia de crop desconhecida: {cfg.strategy}")

    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(img_w, int(round(x2)))
    y2 = min(img_h, int(round(y2)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Bbox invalida apos clamp: ({x1},{y1},{x2},{y2}) para imagem {img_w}x{img_h}")
    return x1, y1, x2, y2


def tight_variant(cfg: CropConfig) -> CropConfig:
    return dataclasses.replace(cfg, strategy="tight", padding_ratio=0.0)


def generate_gt_crops(
    records: list[ImageRecord],
    crop_cfg: CropConfig,
    output_dir: str | Path,
    force: bool = False,
) -> list[CropRecord]:
    output_dir = Path(output_dir)

    def _compute() -> list[CropRecord]:
        output_dir.mkdir(parents=True, exist_ok=True)
        crops: list[CropRecord] = []
        for record in records:
            if record.is_no_logo:
                continue
            with Image.open(record.image_path) as im:
                im = im.convert("RGB")
                for idx, inst in enumerate(record.instances):
                    try:
                        x1, y1, x2, y2 = _compute_crop_box(inst.bbox, record.width, record.height, crop_cfg)
                    except ValueError:
                        logger.warning("Pulando instancia invalida em %s (idx %d)", record.image_id, idx)
                        continue
                    crop_img = im.crop((x1, y1, x2, y2)).resize(
                        (crop_cfg.output_size, crop_cfg.output_size), Image.BILINEAR
                    )
                    out_name = f"{Path(record.image_id).name}__inst{idx}.jpg"
                    out_path = output_dir / out_name
                    crop_img.save(out_path, quality=95)
                    crops.append(
                        CropRecord(
                            crop_path=out_path,
                            class_name=inst.class_name,
                            source_image_id=record.image_id,
                            instance_index=idx,
                            source_bbox=inst.bbox,
                            source_image_size=(record.width, record.height),
                        )
                    )
        logger.info("Gerados %d crops (Fluxo 1) em %s", len(crops), output_dir)
        return crops

    cache_path = output_dir.parent / f"{output_dir.name}_crop_records.pkl"
    return cached(cache_path, _compute, force=force)
