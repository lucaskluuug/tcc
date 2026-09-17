from __future__ import annotations

import dataclasses

from ..data.records import ImageRecord
from .metrics import (
    DetectionPRF1,
    ErrorDecomposition,
    GTBox,
    PredBox,
    compute_detection_prf1,
    compute_map,
    error_decomposition,
    iou_classification_pairs,
)
from .predictors import FullPredictor


@dataclasses.dataclass
class Flow2IouResult:
    iou_threshold: float
    mean_ap: float
    ap_per_class: dict[str, float]
    prf1: DetectionPRF1


@dataclasses.dataclass
class Flow2Result:
    results_by_iou: dict[float, Flow2IouResult]
    iou_classification_pairs: list[tuple[float, bool]]
    error_decomposition: ErrorDecomposition
    all_gts: list[GTBox]
    all_preds: list[PredBox]


def _flatten_ground_truth(records: list[ImageRecord]) -> list[GTBox]:
    gts = []
    for record in records:
        for inst in record.instances:
            gts.append(GTBox(image_id=record.image_id, class_name=inst.class_name, box=inst.bbox.as_xyxy()))
    return gts


def evaluate_flow2(
    records: list[ImageRecord],
    predict_fn: FullPredictor,
    class_names: list[str],
    iou_thresholds: tuple[float, ...],
    primary_iou_threshold: float,
    background_iou_floor: float,
) -> Flow2Result:
    all_gts = _flatten_ground_truth(records)
    all_preds: list[PredBox] = []
    for record in records:
        for det in predict_fn(record.image_path):
            all_preds.append(PredBox(image_id=record.image_id, class_name=det.class_name, box=det.box_xyxy, score=det.score))

    results_by_iou: dict[float, Flow2IouResult] = {}
    for thr in iou_thresholds:
        mean_ap, ap_per_class = compute_map(all_gts, all_preds, class_names, thr)
        prf1 = compute_detection_prf1(all_gts, all_preds, thr)
        results_by_iou[thr] = Flow2IouResult(iou_threshold=thr, mean_ap=mean_ap, ap_per_class=ap_per_class, prf1=prf1)

    pairs = iou_classification_pairs(all_gts, all_preds)
    errors = error_decomposition(all_gts, all_preds, fg_iou_threshold=primary_iou_threshold, bg_iou_floor=background_iou_floor)
    return Flow2Result(
        results_by_iou=results_by_iou,
        iou_classification_pairs=pairs,
        error_decomposition=errors,
        all_gts=all_gts,
        all_preds=all_preds,
    )
