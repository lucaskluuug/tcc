from __future__ import annotations

import dataclasses
from collections import OrderedDict
from pathlib import Path
from typing import Callable

Box = tuple[float, float, float, float]


@dataclasses.dataclass(frozen=True)
class RoIPrediction:
    class_name: str | None
    score: float
    background_score: float


RoIClassifier = Callable[[Path, list[Box]], list[RoIPrediction]]


def roi_classifier(model, device, id_to_name: dict[int, str], background_offset: int = 1) -> RoIClassifier:
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    from torchvision.models.detection.transform import resize_boxes

    model.eval()

    def _classify(image_path: Path, boxes: list[Box]) -> list[RoIPrediction]:
        if not boxes:
            return []

        image = Image.open(image_path).convert("RGB")
        tensor = TF.to_tensor(image).to(device)
        original_size = (tensor.shape[-2], tensor.shape[-1])

        with torch.no_grad():
            images, _ = model.transform([tensor], None)
            features = model.backbone(images.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([("0", features)])

            new_size = images.image_sizes[0]
            box_tensor = torch.as_tensor(boxes, dtype=torch.float32, device=device)
            box_tensor = resize_boxes(box_tensor, original_size, new_size)

            pooled = model.roi_heads.box_roi_pool(features, [box_tensor], [new_size])
            pooled = model.roi_heads.box_head(pooled)
            class_logits, _ = model.roi_heads.box_predictor(pooled)
            probs = torch.softmax(class_logits, dim=-1)

        predictions = []
        foreground = probs[:, 1:]
        best_scores, best_idx = foreground.max(dim=1)
        for i in range(probs.shape[0]):
            label = int(best_idx[i]) + 1
            class_idx = label - background_offset
            predictions.append(
                RoIPrediction(
                    class_name=id_to_name.get(class_idx),
                    score=float(best_scores[i]),
                    background_score=float(probs[i, 0]),
                )
            )
        return predictions

    return _classify
