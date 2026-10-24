from pathlib import Path

from logoloc.data.records import BBox, ImageRecord, LogoInstance
from logoloc.eval.experiments import Limiares, experimento3, tabela_deteccoes
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


def test_tabela_deteccoes_keeps_every_detection():
    preds = [
        PredBox(image_id="a", class_name="adidas", box=(1.0, 2.0, 3.0, 4.0), score=0.9),
        PredBox(image_id="b", class_name="bmw", box=(5.0, 6.0, 7.0, 8.0), score=0.1),
    ]

    table = tabela_deteccoes(preds)

    assert list(table.columns) == ["image_id", "classe", "score", "x1", "y1", "x2", "y2"]
    assert table["image_id"].tolist() == ["a", "b"]
    assert table.iloc[1]["x2"] == 7.0


def test_tabela_deteccoes_without_detections_keeps_columns():
    assert list(tabela_deteccoes([]).columns) == ["image_id", "classe", "score", "x1", "y1", "x2", "y2"]
