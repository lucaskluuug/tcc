#!/usr/bin/env python
"""
Roda os tres experimentos sobre P3 para um checkpoint de Faster R-CNN.

Experimento 1: deteccao convencional (mAP, precisao, revocacao, TIDE) e a marca
que o detector atribui a cada logotipo anotado, sem limiar de confianca.
Experimentos 2 e 3: a caixa anotada, intacta ou degradada por reducao e por
deslocamento ate niveis exatos de IoU, injetada direto na rede de deteccao.

Uso:
    python scripts/11_run_experiments.py \
        --checkpoint outputs/runs/frcnn_r50_v1/model_last.pt \
        --model-config configs/faster_rcnn/resnet50.yaml \
        --run-name frcnn_r50_v1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from logoloc.config import load_dataclass_yaml, load_project_config
from logoloc.data.label_map import inverse, load_label_map
from logoloc.data.splits import prepare_dataset
from logoloc.eval.experiments import Limiares, detectar, experimento1, experimento3, salvar
from logoloc.eval.predictors import faster_rcnn_predictor
from logoloc.eval.roi_classifier import roi_classifier
from logoloc.models.faster_rcnn import FasterRCNNConfig, build_faster_rcnn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--limit", type=int, default=None, help="usa apenas as N primeiras imagens, para teste rapido")
    args = parser.parse_args()

    project_cfg = load_project_config(args.base_config)
    model_cfg = load_dataclass_yaml(FasterRCNNConfig, args.model_config)

    records_p3 = prepare_dataset(project_cfg)["P3"]
    if args.limit:
        com_logo = [r for r in records_p3 if not r.is_no_logo][: args.limit]
        sem_logo = [r for r in records_p3 if r.is_no_logo][: args.limit]
        records_p3 = com_logo + sem_logo

    label_map = load_label_map(project_cfg.resolved_path(project_cfg.paths.processed_dir) / "label_map.json")
    id_to_name = inverse(label_map)
    class_names = list(label_map.keys())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_faster_rcnn(len(label_map) + 1, model_cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.roi_heads.score_thresh = 0.0
    logger.info("Checkpoint carregado (epoca %s), dispositivo %s", checkpoint.get("epoch", "?"), device)

    limiares = Limiares(
        operacao=model_cfg.box_score_thresh,
        piso_fundo=project_cfg.metrics.background_iou_floor,
        ious_map=tuple(project_cfg.metrics.iou_thresholds),
        iou_principal=project_cfg.metrics.primary_iou_threshold,
    )

    logger.info("Experimento 1: deteccao sobre %d imagens, sem limiar de confianca", len(records_p3))
    preds = detectar(records_p3, faster_rcnn_predictor(model, device, id_to_name))
    resumo_exp1, por_logotipo = experimento1(records_p3, preds, class_names, limiares)

    logger.info("Experimentos 2 e 3: injecao das caixas anotadas e degradadas")
    por_caixa = experimento3(records_p3, roi_classifier(model, device, id_to_name), limiares)

    output_dir = project_cfg.resolved_path(project_cfg.paths.outputs_dir) / "results" / args.run_name / "experimentos"
    caminho = salvar(output_dir, args.run_name, limiares, resumo_exp1, por_logotipo, por_caixa)

    with open(caminho, encoding="utf-8") as f:
        resumo = json.load(f)
    print(json.dumps({k: resumo[k] for k in ("experimento2",)}, indent=2, ensure_ascii=False))
    print(json.dumps(resumo["experimento1"]["por_logotipo"], indent=2, ensure_ascii=False))
    logger.info("Resultados salvos em %s", output_dir)


if __name__ == "__main__":
    main()
