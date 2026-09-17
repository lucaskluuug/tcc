from __future__ import annotations

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

from .records import ImageRecord

BACKGROUND_OFFSET = 1


class FlickrLogosDetectionDataset(Dataset):
    def __init__(
        self,
        records: list[ImageRecord],
        label_map: dict[str, int],
        transforms=None,
    ):
        self.records = records
        self.label_map = label_map
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image = Image.open(record.image_path).convert("RGB")

        boxes, labels = [], []
        for inst in record.instances:
            b = inst.bbox
            boxes.append([b.x, b.y, b.x2, b.y2])
            labels.append(self.label_map[inst.class_name] + BACKGROUND_OFFSET)

        boxes_t = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
            "image_id_str": record.image_id,
        }

        image_tensor = TF.to_tensor(image)
        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)
        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def num_classes_with_background(label_map: dict[str, int]) -> int:
    return len(label_map) + BACKGROUND_OFFSET
