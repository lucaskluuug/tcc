#!/usr/bin/env python
"""Avalia um checkpoint de Faster R-CNN: Fluxo 1 (crops de P3) e Fluxo 2
(imagens completas de P3), resultado salvo em outputs/results/<run-name>/.

Uso:
    python scripts/06_evaluate_faster_rcnn.py \
        --checkpoint outputs/runs/faster_rcnn_resnet50_run1/model_last.pt \
        --model-config configs/faster_rcnn/resnet50.yaml \
        --run-name faster_rcnn_resnet50_run1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from logoloc.config import load_dataclass_yaml, load_project_config
from logoloc.data.crops import generate_gt_crops, tight_variant
from logoloc.data.label_map import inverse, load_label_map
from logoloc.data.splits import prepare_dataset
from logoloc.eval.consolidate import evaluate_model, save_evaluation
from logoloc.eval.predictors import faster_rcnn_predictor
from logoloc.models.faster_rcnn import FasterRCNNConfig, build_faster_rcnn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    project_cfg = load_project_config(args.base_config)
    model_cfg = load_dataclass_yaml(FasterRCNNConfig, args.model_config)

    records_by_split = prepare_dataset(project_cfg)
    records_p3 = records_by_split["P3"]

    label_map = load_label_map(project_cfg.resolved_path(project_cfg.paths.processed_dir) / "label_map.json")
    id_to_name = inverse(label_map)
    class_names = list(label_map.keys())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_faster_rcnn(len(label_map) + 1, model_cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Checkpoint carregado (epoca %s)", checkpoint.get("epoch", "?"))

    processed_dir = project_cfg.resolved_path(project_cfg.paths.processed_dir)
    crops_p3 = generate_gt_crops(records_p3, project_cfg.crop, output_dir=processed_dir / "p3_gt_crops")
    crops_p3_tight = generate_gt_crops(
        records_p3, tight_variant(project_cfg.crop), output_dir=processed_dir / "p3_gt_crops_tight"
    )

    predict_fn = faster_rcnn_predictor(model, device, id_to_name)
    evaluation = evaluate_model(
        args.run_name,
        predict_fn,
        crops_p3,
        records_p3,
        class_names,
        project_cfg.metrics.iou_thresholds,
        primary_iou_threshold=project_cfg.metrics.primary_iou_threshold,
        background_iou_floor=project_cfg.metrics.background_iou_floor,
        crops_p3_tight=crops_p3_tight,
    )
    output_dir = project_cfg.resolved_path(project_cfg.paths.outputs_dir) / "results" / args.run_name
    save_evaluation(evaluation, output_dir)
    logger.info("Avaliacao salva em %s", output_dir)


if __name__ == "__main__":
    main()
