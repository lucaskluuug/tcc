from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass(frozen=True)
class GTBox:
    image_id: str
    class_name: str
    box: tuple[float, float, float, float]


@dataclasses.dataclass(frozen=True)
class PredBox:
    image_id: str
    class_name: str
    box: tuple[float, float, float, float]
    score: float


def iou_xyxy(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    ax1, ay1, ax2, ay2 = boxes_a[:, 0:1], boxes_a[:, 1:2], boxes_a[:, 2:3], boxes_a[:, 3:4]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter_area = inter_w * inter_h

    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a + area_b - inter_area
    return np.where(union > 0, inter_area / np.where(union > 0, union, 1), 0.0)


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    if len(recall) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def compute_ap_for_class(
    gts: list[GTBox], preds: list[PredBox], iou_threshold: float
) -> tuple[float, np.ndarray, np.ndarray]:
    n_gt = len(gts)
    if n_gt == 0:
        return 0.0, np.array([]), np.array([])

    gt_by_image: dict[str, list[int]] = {}
    for i, gt in enumerate(gts):
        gt_by_image.setdefault(gt.image_id, []).append(i)
    matched = np.zeros(n_gt, dtype=bool)

    preds_sorted = sorted(preds, key=lambda p: p.score, reverse=True)
    tp = np.zeros(len(preds_sorted))
    fp = np.zeros(len(preds_sorted))

    for i, pred in enumerate(preds_sorted):
        candidate_idx = gt_by_image.get(pred.image_id, [])
        best_iou, best_gt_i = 0.0, -1
        for gi in candidate_idx:
            if matched[gi]:
                continue
            iou = iou_xyxy(pred.box, gts[gi].box)
            if iou > best_iou:
                best_iou, best_gt_i = iou, gi
        if best_iou >= iou_threshold and best_gt_i >= 0:
            tp[i] = 1
            matched[best_gt_i] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.clip(tp_cum + fp_cum, 1e-12, None)
    ap = average_precision(recall, precision)
    return ap, recall, precision


def compute_map(
    all_gts: list[GTBox],
    all_preds: list[PredBox],
    class_names: list[str],
    iou_threshold: float,
) -> tuple[float, dict[str, float]]:
    ap_per_class: dict[str, float] = {}
    for class_name in class_names:
        gts_c = [g for g in all_gts if g.class_name == class_name]
        preds_c = [p for p in all_preds if p.class_name == class_name]
        if len(gts_c) == 0:
            continue
        ap, _, _ = compute_ap_for_class(gts_c, preds_c, iou_threshold)
        ap_per_class[class_name] = ap
    mean_ap = float(np.mean(list(ap_per_class.values()))) if ap_per_class else 0.0
    return mean_ap, ap_per_class


@dataclasses.dataclass
class DetectionPRF1:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    mean_iou_tp: float


def compute_detection_prf1(
    all_gts: list[GTBox], all_preds: list[PredBox], iou_threshold: float
) -> DetectionPRF1:
    gt_by_image: dict[str, list[int]] = {}
    for i, gt in enumerate(all_gts):
        gt_by_image.setdefault(gt.image_id, []).append(i)
    matched = np.zeros(len(all_gts), dtype=bool)

    preds_sorted = sorted(all_preds, key=lambda p: p.score, reverse=True)
    tp, fp = 0, 0
    matched_ious: list[float] = []

    for pred in preds_sorted:
        candidate_idx = gt_by_image.get(pred.image_id, [])
        best_iou, best_gt_i = 0.0, -1
        for gi in candidate_idx:
            if matched[gi] or all_gts[gi].class_name != pred.class_name:
                continue
            iou = iou_xyxy(pred.box, all_gts[gi].box)
            if iou > best_iou:
                best_iou, best_gt_i = iou, gi
        if best_iou >= iou_threshold and best_gt_i >= 0:
            tp += 1
            matched[best_gt_i] = True
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = int((~matched).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou_tp = float(np.mean(matched_ious)) if matched_ious else 0.0
    return DetectionPRF1(precision=precision, recall=recall, f1=f1, tp=tp, fp=fp, fn=fn, mean_iou_tp=mean_iou_tp)


def iou_classification_pairs(all_gts: list[GTBox], all_preds: list[PredBox]) -> list[tuple[float, bool]]:
    gt_by_image: dict[str, list[int]] = {}
    for i, gt in enumerate(all_gts):
        gt_by_image.setdefault(gt.image_id, []).append(i)
    matched = np.zeros(len(all_gts), dtype=bool)

    preds_sorted = sorted(all_preds, key=lambda p: p.score, reverse=True)
    pairs: list[tuple[float, bool]] = []
    for pred in preds_sorted:
        candidate_idx = [gi for gi in gt_by_image.get(pred.image_id, []) if not matched[gi]]
        if not candidate_idx:
            continue
        ious = [iou_xyxy(pred.box, all_gts[gi].box) for gi in candidate_idx]
        best_local = int(np.argmax(ious))
        best_iou, best_gt_i = ious[best_local], candidate_idx[best_local]
        if best_iou <= 0:
            continue
        matched[best_gt_i] = True
        class_correct = all_gts[best_gt_i].class_name == pred.class_name
        pairs.append((best_iou, class_correct))
    return pairs


@dataclasses.dataclass
class ErrorDecomposition:
    true_positive: int
    classification_error: int
    localization_error: int
    both_error: int
    duplicate: int
    background_error: int
    missed: int

    @property
    def total_errors(self) -> int:
        return (
            self.classification_error
            + self.localization_error
            + self.both_error
            + self.duplicate
            + self.background_error
            + self.missed
        )

    def fractions(self) -> dict[str, float]:
        total = self.total_errors
        if total == 0:
            return {k: 0.0 for k in ("classification_error", "localization_error", "both_error", "duplicate", "background_error", "missed")}
        return {
            "classification_error": self.classification_error / total,
            "localization_error": self.localization_error / total,
            "both_error": self.both_error / total,
            "duplicate": self.duplicate / total,
            "background_error": self.background_error / total,
            "missed": self.missed / total,
        }


def error_decomposition(
    all_gts: list[GTBox],
    all_preds: list[PredBox],
    fg_iou_threshold: float,
    bg_iou_floor: float,
) -> ErrorDecomposition:
    gt_by_image: dict[str, list[int]] = {}
    for i, gt in enumerate(all_gts):
        gt_by_image.setdefault(gt.image_id, []).append(i)
    consumed = np.zeros(len(all_gts), dtype=bool)

    preds_sorted = sorted(all_preds, key=lambda p: p.score, reverse=True)
    tp = cls_err = loc_err = both_err = dup = bkg_err = 0

    for pred in preds_sorted:
        candidate_idx = gt_by_image.get(pred.image_id, [])
        best_iou, best_gi = 0.0, -1
        for gi in candidate_idx:
            iou = iou_xyxy(pred.box, all_gts[gi].box)
            if iou > best_iou:
                best_iou, best_gi = iou, gi

        if best_gi == -1 or best_iou < bg_iou_floor:
            bkg_err += 1
            continue

        class_correct = all_gts[best_gi].class_name == pred.class_name
        if best_iou >= fg_iou_threshold:
            if not class_correct:
                cls_err += 1
            elif not consumed[best_gi]:
                tp += 1
                consumed[best_gi] = True
            else:
                dup += 1
        else:
            if class_correct:
                loc_err += 1
            else:
                both_err += 1

    missed = int((~consumed).sum())
    return ErrorDecomposition(
        true_positive=tp,
        classification_error=cls_err,
        localization_error=loc_err,
        both_error=both_err,
        duplicate=dup,
        background_error=bkg_err,
        missed=missed,
    )
