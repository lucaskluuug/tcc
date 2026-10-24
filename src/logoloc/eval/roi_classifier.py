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
    second_class_name: str | None = None
    second_score: float = 0.0
    true_class_score: float | None = None
    true_class_rank: int | None = None


RoIClassifier = Callable[..., list[RoIPrediction]]


def roi_classifier(model, device, id_to_name: dict[int, str], background_offset: int = 1) -> RoIClassifier:
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    from torchvision.models.detection.transform import resize_boxes

    model.eval()
    name_to_column = {name: idx + background_offset - 1 for idx, name in id_to_name.items()}

    def _classify(image_path: Path, boxes: list[Box], true_classes: list[str] | None = None) -> list[RoIPrediction]:
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
        without_best = foreground.clone()
        without_best[torch.arange(foreground.shape[0], device=foreground.device), best_idx] = -1.0
        second_scores, second_idx = without_best.max(dim=1)
        for i in range(probs.shape[0]):
            label = int(best_idx[i]) + 1
            class_idx = label - background_offset
            true_score, true_rank = None, None
            if true_classes is not None and true_classes[i] in name_to_column:
                column = name_to_column[true_classes[i]]
                true_score = float(foreground[i, column])
                true_rank = int((foreground[i] > foreground[i, column]).sum()) + 1
            predictions.append(
                RoIPrediction(
                    class_name=id_to_name.get(class_idx),
                    score=float(best_scores[i]),
                    background_score=float(probs[i, 0]),
                    second_class_name=id_to_name.get(int(second_idx[i]) + 1 - background_offset),
                    second_score=float(second_scores[i]),
                    true_class_score=true_score,
                    true_class_rank=true_rank,
                )
            )
        return predictions

    return _classify
