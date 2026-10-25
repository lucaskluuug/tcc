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
    proposal_xyxy: tuple[float, float, float, float] | None = None


def detect_with_proposals(model, tensor) -> dict:
    from collections import OrderedDict

    import torch
    import torch.nn.functional as F
    from torchvision.models.detection.transform import resize_boxes
    from torchvision.ops import boxes as box_ops

    heads = model.roi_heads
    original_size = (tensor.shape[-2], tensor.shape[-1])
    images, _ = model.transform([tensor])
    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    proposals, _ = model.rpn(images, features)
    image_size = images.image_sizes[0]
    pooled = heads.box_head(heads.box_roi_pool(features, proposals, images.image_sizes))
    class_logits, box_regression = heads.box_predictor(pooled)

    boxes = box_ops.clip_boxes_to_image(heads.box_coder.decode(box_regression, proposals), image_size)
    scores = F.softmax(class_logits, -1)
    labels = torch.arange(scores.shape[-1], device=scores.device).view(1, -1).expand_as(scores)
    origin = torch.arange(scores.shape[0], device=scores.device).view(-1, 1).expand_as(scores)

    boxes, scores = boxes[:, 1:].reshape(-1, 4), scores[:, 1:].reshape(-1)
    labels, origin = labels[:, 1:].reshape(-1), origin[:, 1:].reshape(-1)
    keep = torch.where(scores > heads.score_thresh)[0]
    boxes, scores, labels, origin = boxes[keep], scores[keep], labels[keep], origin[keep]
    keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
    boxes, scores, labels, origin = boxes[keep], scores[keep], labels[keep], origin[keep]
    keep = box_ops.batched_nms(boxes, scores, labels, heads.nms_thresh)[: heads.detections_per_img]
    boxes, scores, labels, origin = boxes[keep], scores[keep], labels[keep], origin[keep]

    return {
        "boxes": resize_boxes(boxes, image_size, original_size),
        "labels": labels,
        "scores": scores,
        "proposals": resize_boxes(proposals[0][origin], image_size, original_size),
    }


def faster_rcnn_predictor(
    model, device, id_to_name: dict[int, str], background_offset: int = 1, return_proposals: bool = False
) -> FullPredictor:
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image

    model.eval()

    def _predict(image_path: Path) -> list[Detection]:
        image = Image.open(image_path).convert("RGB")
        tensor = TF.to_tensor(image).to(device)
        with torch.no_grad():
            output = detect_with_proposals(model, tensor) if return_proposals else model([tensor])[0]
        proposals = output["proposals"].tolist() if return_proposals else [None] * len(output["boxes"])
        detections = []
        for box, label, score, proposal in zip(
            output["boxes"].tolist(), output["labels"].tolist(), output["scores"].tolist(), proposals
        ):
            class_idx = label - background_offset
            if class_idx not in id_to_name:
                continue
            detections.append(
                Detection(
                    class_name=id_to_name[class_idx],
                    box_xyxy=tuple(box),
                    score=score,
                    proposal_xyxy=tuple(proposal) if proposal is not None else None,
                )
            )
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
