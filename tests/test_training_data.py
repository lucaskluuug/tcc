from pathlib import Path

import cv2
import numpy as np

from logoloc.config import AugmentationConfig, DatasetConfig
from logoloc.data.augment import generate_augmented_p1
from logoloc.data.records import BBox, ImageRecord, LogoInstance
from logoloc.data.splits import select_training_records


def _record(split, image_id, with_logo=True, image_path=None, size=100):
    instances = [LogoInstance(class_name="adidas", bbox=BBox(16, 16, 32, 32))] if with_logo else []
    return ImageRecord(
        image_path=image_path or Path(f"{image_id}.jpg"),
        split=split,
        width=size,
        height=size,
        instances=instances,
        image_id=image_id,
    )


def _records_by_split():
    return {
        "P1": [_record("P1", "a"), _record("P1", "b")],
        "P2": [_record("P2", "c"), _record("P2", "sem_logo", with_logo=False), _record("P2", "d")],
        "P3": [_record("P3", "e")],
    }


def test_default_training_split_is_p1():
    assert DatasetConfig().train_splits == ("P1",)


def test_select_training_records_keeps_p1_only_by_default():
    selected = select_training_records(_records_by_split(), ("P1",))

    assert [r.image_id for r in selected] == ["a", "b"]


def test_select_training_records_joins_splits_and_drops_no_logo_images():
    selected = select_training_records(_records_by_split(), ("P1", "P2"))

    assert [r.image_id for r in selected] == ["a", "b", "c", "d"]


def test_augmentation_cache_follows_output_dir(tmp_path):
    image_path = tmp_path / "img.jpg"
    cv2.imwrite(str(image_path), np.full((64, 64, 3), 127, dtype=np.uint8))
    record = _record("P2", "classes/jpg/adidas/img", image_path=image_path, size=64)

    result = generate_augmented_p1([record], AugmentationConfig(copies_per_image=1), output_dir=tmp_path / "p1_p2_augmented", seed=0)

    assert (tmp_path / "p1_p2_augmented_records.pkl").exists()
    assert not (tmp_path / "p1_augmented_records.pkl").exists()
    assert [r.split for r in result] == ["P2", "P2"]
