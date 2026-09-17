#!/usr/bin/env python
"""
Mede como a classificacao degrada conforme a caixa de proposta piora, usando as
caixas anotadas perturbadas em niveis controlados de IoU e injetadas direto na
RoI head do Faster R-CNN.

Uso:
    python scripts/10_localization_sensitivity.py \
        --checkpoint outputs/runs/frcnn_r50_v1/model_last.pt \
        --model-config configs/faster_rcnn/resnet50.yaml \
        --run-name frcnn_r50_v1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from logoloc.config import load_dataclass_yaml, load_project_config
from logoloc.data.label_map import inverse, load_label_map
from logoloc.data.splits import prepare_dataset
from logoloc.eval.roi_classifier import roi_classifier
from logoloc.eval.sensitivity import DEFAULT_LEVELS, plot_sensitivity, run_sensitivity, save_sensitivity
from logoloc.models.faster_rcnn import FasterRCNNConfig, build_faster_rcnn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--levels", nargs="*", type=float, default=list(DEFAULT_LEVELS))
    parser.add_argument("--limit", type=int, default=None, help="usa apenas as N primeiras imagens, para teste rapido")
    args = parser.parse_args()

    project_cfg = load_project_config(args.base_config)
    model_cfg = load_dataclass_yaml(FasterRCNNConfig, args.model_config)

    records_by_split = prepare_dataset(project_cfg)
    records_p3 = records_by_split["P3"]
    if args.limit:
        com_logo = [r for r in records_p3 if not r.is_no_logo][: args.limit]
        records_p3 = com_logo

    processed_dir = project_cfg.resolved_path(project_cfg.paths.processed_dir)
    label_map = load_label_map(processed_dir / "label_map.json")
    id_to_name = inverse(label_map)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_faster_rcnn(len(label_map) + 1, model_cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Checkpoint carregado (epoca %s), dispositivo %s", checkpoint.get("epoch", "?"), device)

    classify = roi_classifier(model, device, id_to_name)
    resultado = run_sensitivity(
        args.run_name, classify, records_p3, levels=tuple(args.levels), seed=project_cfg.seed
    )

    output_dir = project_cfg.resolved_path(project_cfg.paths.outputs_dir) / "results" / args.run_name
    caminho = save_sensitivity(resultado, output_dir)
    plot_sensitivity([resultado], output_dir / "localization_sensitivity.png")

    print(resultado.by_level().to_string(index=False))
    logger.info("Resultados salvos em %s", caminho.parent)


if __name__ == "__main__":
    main()
