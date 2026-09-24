from __future__ import annotations

import logging
from pathlib import Path

import albumentations as A
import cv2

from ..config import AugmentationConfig
from ..seed import set_global_seed
from .cache import cached
from .records import BBox, ImageRecord, LogoInstance

logger = logging.getLogger(__name__)


def build_augmentation_pipeline(cfg: AugmentationConfig) -> A.Compose:
    scale_min, scale_max = cfg.scale_range
    return A.Compose(
        [
            A.HorizontalFlip(p=cfg.horizontal_flip_prob),
            A.RandomScale(scale_limit=(scale_min - 1.0, scale_max - 1.0), p=0.8),
            A.ColorJitter(
                brightness=cfg.brightness_delta,
                contrast=cfg.contrast_delta,
                saturation=cfg.saturation_delta,
                hue=cfg.hue_delta,
                p=0.8,
            ),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"], min_visibility=0.3),
    )


def _augment_one(pipeline: A.Compose, record: ImageRecord) -> tuple:
    image = cv2.cvtColor(cv2.imread(str(record.image_path)), cv2.COLOR_BGR2RGB)
    bboxes = [[inst.bbox.x, inst.bbox.y, inst.bbox.w, inst.bbox.h] for inst in record.instances]
    labels = [inst.class_name for inst in record.instances]
    result = pipeline(image=image, bboxes=bboxes, class_labels=labels)
    return result["image"], result["bboxes"], result["class_labels"]


def generate_augmented_p1(
    p1_records: list[ImageRecord],
    aug_cfg: AugmentationConfig,
    output_dir: str | Path,
    seed: int,
    force: bool = False,
) -> list[ImageRecord]:
    output_dir = Path(output_dir)

    def _compute() -> list[ImageRecord]:
        set_global_seed(seed)
        pipeline = build_augmentation_pipeline(aug_cfg)
        output_dir.mkdir(parents=True, exist_ok=True)
        augmented: list[ImageRecord] = []
        for record in p1_records:
            for copy_idx in range(aug_cfg.copies_per_image):
                try:
                    image, bboxes, labels = _augment_one(pipeline, record)
                except Exception:
                    logger.exception("Falha ao aumentar %s (copia %d), pulando.", record.image_path, copy_idx)
                    continue
                if not bboxes:
                    continue
                out_name = f"{Path(record.image_id).stem}__aug{copy_idx}.jpg"
                out_path = output_dir / out_name
                cv2.imwrite(str(out_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                h, w = image.shape[:2]
                instances = [
                    LogoInstance(class_name=label, bbox=BBox(*bbox))
                    for bbox, label in zip(bboxes, labels)
                ]
                augmented.append(
                    ImageRecord(
                        image_path=out_path,
                        split="P1",
                        width=w,
                        height=h,
                        instances=instances,
                        image_id=f"{record.image_id}__aug{copy_idx}",
                    )
                )
        logger.info("Augmentation de P1: %d imagens originais -> +%d copias aumentadas.", len(p1_records), len(augmented))
        return p1_records + augmented

    cache_path = output_dir.parent / "p1_augmented_records.pkl"
    return cached(cache_path, _compute, force=force)
