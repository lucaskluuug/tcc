from __future__ import annotations

import dataclasses

from ..data.crops import CropRecord
from .classification_metrics import ClassificationReport, classification_report
from .predictors import Top1Predictor

NO_DETECTION_LABEL = "__no_detection__"


@dataclasses.dataclass
class Flow1Result:
    report: ClassificationReport
    detection_failure_rate: float
    y_true: list[str]
    y_pred: list[str]


def evaluate_flow1(crops: list[CropRecord], predict_fn: Top1Predictor, class_names: list[str]) -> Flow1Result:
    y_true: list[str] = []
    y_pred: list[str] = []
    n_no_detection = 0

    for crop in crops:
        predicted_class = predict_fn(crop.crop_path)
        if predicted_class is None:
            n_no_detection += 1
            predicted_class = NO_DETECTION_LABEL
        y_true.append(crop.class_name)
        y_pred.append(predicted_class)

    report = classification_report(y_true, y_pred, class_names=class_names)
    detection_failure_rate = n_no_detection / len(crops) if crops else 0.0
    return Flow1Result(report=report, detection_failure_rate=detection_failure_rate, y_true=y_true, y_pred=y_pred)
