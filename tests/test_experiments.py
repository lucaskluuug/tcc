from pathlib import Path

import pandas as pd
import pytest

from logoloc.data.records import BBox, ImageRecord, LogoInstance
from logoloc.eval.experiments import Limiares, experimento3, resposta_por_logotipo, tabela_deteccoes
from logoloc.eval.metrics import PredBox
from logoloc.eval.roi_classifier import RoIPrediction


def test_experimento3_passes_true_classes_and_records_second_class():
    record = ImageRecord(
        image_path=Path("img.jpg"),
        split="P3",
        width=200,
        height=200,
        instances=[LogoInstance(class_name="adidas", bbox=BBox(50, 50, 60, 40))],
        image_id="img",
    )
    calls = []

    def fake_classify(image_path, boxes, true_classes=None):
        calls.append((len(boxes), true_classes))
        return [
            RoIPrediction(
                class_name="adidas",
                score=0.6,
                background_score=0.3,
                second_class_name="bmw",
                second_score=0.05,
                true_class_score=0.6,
                true_class_rank=1,
            )
            for _ in boxes
        ]

    result = experimento3([record], fake_classify, Limiares(), modos=("reducao",), niveis=(1.0, 0.5))

    n_boxes, true_classes = calls[0]
    assert n_boxes == len(result) == 2
    assert true_classes == ["adidas", "adidas"]
    assert (result["classe_segunda"] == "bmw").all()
    assert (result["score_segunda"] == 0.05).all()
    assert (result["prob_verdadeira"] == 0.6).all()
    assert (result["posicao_verdadeira"] == 1).all()


def test_roi_prediction_defaults_keep_old_fields_valid():
    pred = RoIPrediction(class_name="adidas", score=0.6, background_score=0.3)

    assert pred.second_class_name is None
    assert pred.true_class_rank is None


COLUNAS = ["image_id", "classe", "score", "x1", "y1", "x2", "y2", "px1", "py1", "px2", "py2"]


def test_tabela_deteccoes_keeps_every_detection():
    preds = [
        PredBox(image_id="a", class_name="adidas", box=(1.0, 2.0, 3.0, 4.0), score=0.9, proposal=(0.5, 1.5, 3.5, 4.5)),
        PredBox(image_id="b", class_name="bmw", box=(5.0, 6.0, 7.0, 8.0), score=0.1),
    ]

    table = tabela_deteccoes(preds)

    assert list(table.columns) == COLUNAS
    assert table["image_id"].tolist() == ["a", "b"]
    assert table.iloc[1]["x2"] == 7.0
    assert table.iloc[0]["px1"] == 0.5
    assert pd.isna(table.iloc[1]["px1"])


def test_tabela_deteccoes_without_detections_keeps_columns():
    assert list(tabela_deteccoes([]).columns) == COLUNAS


def test_resposta_por_logotipo_records_iou_of_classified_proposal():
    record = ImageRecord(
        image_path=Path("img.jpg"),
        split="P3",
        width=200,
        height=200,
        instances=[
            LogoInstance(class_name="adidas", bbox=BBox(50, 50, 100, 100)),
            LogoInstance(class_name="adidas", bbox=BBox(0, 170, 20, 20)),
        ],
        image_id="img",
    )
    preds = [PredBox(image_id="img", class_name="adidas", box=(50, 50, 150, 150), score=0.9, proposal=(60, 60, 150, 150))]

    result = resposta_por_logotipo([record], preds, Limiares())

    assert result.iloc[0]["iou"] == 1.0
    assert result.iloc[0]["iou_proposta"] == pytest.approx(0.81)
    assert not result.iloc[1]["localizado"]
    assert result.iloc[1]["iou_proposta"] == 0.0


def test_resposta_por_logotipo_without_proposals_leaves_proposal_iou_empty():
    record = ImageRecord(
        image_path=Path("img.jpg"),
        split="P3",
        width=200,
        height=200,
        instances=[LogoInstance(class_name="adidas", bbox=BBox(50, 50, 100, 100))],
        image_id="img",
    )
    preds = [PredBox(image_id="img", class_name="adidas", box=(50, 50, 150, 150), score=0.9)]

    result = resposta_por_logotipo([record], preds, Limiares())

    assert result.iloc[0]["localizado"]
    assert pd.isna(result.iloc[0]["iou_proposta"])
