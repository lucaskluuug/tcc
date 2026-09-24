import numpy as np
import pytest

from logoloc.eval.metrics import (
    GTBox,
    PredBox,
    average_precision,
    compute_ap_for_class,
    compute_detection_prf1,
    compute_map,
    error_decomposition,
    iou_matrix,
    iou_xyxy,
)


def test_iou_xyxy_identical_boxes():
    box = (0, 0, 10, 10)
    assert iou_xyxy(box, box) == pytest.approx(1.0)


def test_iou_xyxy_no_overlap():
    assert iou_xyxy((0, 0, 5, 5), (10, 10, 15, 15)) == pytest.approx(0.0)


def test_iou_xyxy_known_value():
    iou = iou_xyxy((0, 0, 10, 10), (5, 5, 15, 15))
    assert iou == pytest.approx(25 / 175)


def test_iou_matrix_matches_pairwise():
    boxes_a = np.array([[0, 0, 10, 10], [0, 0, 5, 5]], dtype=float)
    boxes_b = np.array([[5, 5, 15, 15]], dtype=float)
    mat = iou_matrix(boxes_a, boxes_b)
    assert mat.shape == (2, 1)
    assert mat[0, 0] == pytest.approx(iou_xyxy((0, 0, 10, 10), (5, 5, 15, 15)))
    assert mat[1, 0] == pytest.approx(iou_xyxy((0, 0, 5, 5), (5, 5, 15, 15)))


def test_average_precision_perfect_ranking():
    recall = np.array([0.5, 1.0])
    precision = np.array([1.0, 1.0])
    assert average_precision(recall, precision) == pytest.approx(1.0)


def test_average_precision_known_pascal_example():
    tp = np.array([1, 1, 0, 1, 0])
    fp = 1 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    n_gt = 5
    recall = tp_cum / n_gt
    precision = tp_cum / (tp_cum + fp_cum)
    ap = average_precision(recall, precision)
    assert ap == pytest.approx(0.55, abs=1e-6)


def test_compute_ap_for_class_single_perfect_detection():
    gts = [GTBox(image_id="img1", class_name="adidas", box=(0, 0, 10, 10))]
    preds = [PredBox(image_id="img1", class_name="adidas", box=(0, 0, 10, 10), score=0.9)]

    ap, recall, precision = compute_ap_for_class(gts, preds, iou_threshold=0.5)

    assert ap == pytest.approx(1.0)
    assert recall[-1] == pytest.approx(1.0)
    assert precision[-1] == pytest.approx(1.0)


def test_compute_ap_for_class_false_positive_hurts_precision_not_recall():
    gts = [GTBox(image_id="img1", class_name="adidas", box=(0, 0, 10, 10))]
    preds = [
        PredBox(image_id="img1", class_name="adidas", box=(0, 0, 10, 10), score=0.9),
        PredBox(image_id="img2", class_name="adidas", box=(0, 0, 10, 10), score=0.8),
    ]

    ap, recall, precision = compute_ap_for_class(gts, preds, iou_threshold=0.5)

    assert recall[-1] == pytest.approx(1.0)
    assert precision[-1] == pytest.approx(0.5)


def test_compute_ap_for_class_no_gt_returns_zero():
    ap, recall, precision = compute_ap_for_class([], [], iou_threshold=0.5)
    assert ap == 0.0
    assert len(recall) == 0


def test_compute_map_averages_only_classes_with_gt():
    gts = [
        GTBox(image_id="img1", class_name="adidas", box=(0, 0, 10, 10)),
        GTBox(image_id="img2", class_name="bmw", box=(0, 0, 10, 10)),
    ]
    preds = [
        PredBox(image_id="img1", class_name="adidas", box=(0, 0, 10, 10), score=0.9),
        PredBox(image_id="img2", class_name="bmw", box=(20, 20, 30, 30), score=0.9),
    ]

    mean_ap, ap_per_class = compute_map(gts, preds, class_names=["adidas", "bmw", "chimay"], iou_threshold=0.5)

    assert "chimay" not in ap_per_class
    assert ap_per_class["adidas"] == pytest.approx(1.0)
    assert ap_per_class["bmw"] == pytest.approx(0.0)
    assert mean_ap == pytest.approx(0.5)


def test_compute_detection_prf1_counts_tp_fp_fn():
    gts = [
        GTBox(image_id="img1", class_name="adidas", box=(0, 0, 10, 10)),
        GTBox(image_id="img2", class_name="bmw", box=(0, 0, 10, 10)),
    ]
    preds = [
        PredBox(image_id="img1", class_name="adidas", box=(0, 0, 10, 10), score=0.9),
        PredBox(image_id="img2", class_name="bmw", box=(50, 50, 60, 60), score=0.8),
    ]

    result = compute_detection_prf1(gts, preds, iou_threshold=0.5)

    assert result.tp == 1
    assert result.fp == 1
    assert result.fn == 1
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.mean_iou_tp == pytest.approx(1.0)


def test_compute_detection_prf1_wrong_class_is_fp_and_fn():
    gts = [GTBox(image_id="img1", class_name="adidas", box=(0, 0, 10, 10))]
    preds = [PredBox(image_id="img1", class_name="bmw", box=(0, 0, 10, 10), score=0.9)]

    result = compute_detection_prf1(gts, preds, iou_threshold=0.5)

    assert result.tp == 0
    assert result.fp == 1


def test_error_decomposition_all_categories():
    gts = [
        GTBox(image_id="img1", class_name="x", box=(0, 0, 10, 10)),
        GTBox(image_id="img1", class_name="y", box=(20, 20, 30, 30)),
    ]
    preds = [
        PredBox(image_id="img1", class_name="x", box=(0, 0, 10, 10), score=0.9),
        PredBox(image_id="img1", class_name="y", box=(20, 20, 25, 25), score=0.8),
        PredBox(image_id="img1", class_name="x", box=(20, 20, 30, 30), score=0.7),
        PredBox(image_id="img1", class_name="z", box=(100, 100, 110, 110), score=0.6),
    ]

    result = error_decomposition(gts, preds, fg_iou_threshold=0.5, bg_iou_floor=0.1)

    assert result.true_positive == 1
    assert result.localization_error == 1
    assert result.classification_error == 1
    assert result.background_error == 1
    assert result.both_error == 0
    assert result.duplicate == 0
    assert result.missed == 1
    assert result.true_positive + result.missed == len(gts)


def test_error_decomposition_duplicate_detection():
    gts = [GTBox(image_id="img1", class_name="x", box=(0, 0, 10, 10))]
    preds = [
        PredBox(image_id="img1", class_name="x", box=(0, 0, 10, 10), score=0.9),
        PredBox(image_id="img1", class_name="x", box=(0, 0, 10, 10), score=0.5),
    ]

    result = error_decomposition(gts, preds, fg_iou_threshold=0.5, bg_iou_floor=0.1)

    assert result.true_positive == 1
    assert result.duplicate == 1
    assert result.missed == 0


def test_error_decomposition_both_error_wrong_class_and_bad_box():
    gts = [GTBox(image_id="img1", class_name="x", box=(0, 0, 10, 10))]
    preds = [PredBox(image_id="img1", class_name="y", box=(4, 4, 14, 14), score=0.9)]

    result = error_decomposition(gts, preds, fg_iou_threshold=0.5, bg_iou_floor=0.1)

    assert result.both_error == 1
    assert result.missed == 1


def test_error_decomposition_fractions_sum_to_one():
    gts = [GTBox(image_id="img1", class_name="x", box=(0, 0, 10, 10))]
    preds = [PredBox(image_id="img1", class_name="y", box=(100, 100, 110, 110), score=0.9)]

    result = error_decomposition(gts, preds, fg_iou_threshold=0.5, bg_iou_floor=0.1)

    assert result.missed == 1
    fractions = result.fractions()
    assert sum(fractions.values()) == pytest.approx(1.0)
