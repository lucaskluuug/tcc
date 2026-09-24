from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

FullPredictor = Callable[[Path], list["Detection"]]
Top1Predictor = Callable[[Path], "str | None"]


@dataclasses.dataclass(frozen=True)
class Detection:
    class_name: str
    box_xyxy: tuple[float, float, float, float]
    score: float


def faster_rcnn_predictor(model, device, id_to_name: dict[int, str], background_offset: int = 1) -> FullPredictor:
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image

    model.eval()

    def _predict(image_path: Path) -> list[Detection]:
        image = Image.open(image_path).convert("RGB")
        tensor = TF.to_tensor(image).to(device)
        with torch.no_grad():
            output = model([tensor])[0]
        detections = []
        for box, label, score in zip(output["boxes"].tolist(), output["labels"].tolist(), output["scores"].tolist()):
            class_idx = label - background_offset
            if class_idx not in id_to_name:
                continue
            detections.append(Detection(class_name=id_to_name[class_idx], box_xyxy=tuple(box), score=score))
        return detections

    return _predict


def yolo_predictor(model, predict_kwargs: dict) -> FullPredictor:
    def _predict(image_path: Path) -> list[Detection]:
        result = model.predict(str(image_path), verbose=False, **predict_kwargs)[0]
        detections = []
        if result.boxes is None:
            return detections
        for box_xyxy, cls_id, score in zip(
            result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()
        ):
            class_name = result.names[int(cls_id)]
            detections.append(Detection(class_name=class_name, box_xyxy=tuple(box_xyxy), score=score))
        return detections

    return _predict


def top1_from_full(full_predictor: FullPredictor) -> Top1Predictor:
    def _predict(image_path: Path) -> str | None:
        detections = full_predictor(image_path)
        if not detections:
            return None
        return max(detections, key=lambda d: d.score).class_name

    return _predict
