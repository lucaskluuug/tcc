#!/usr/bin/env python
"""Consolida os resultados dos modelos ja' avaliados (scripts 06/07) em uma
tabela comparativa Fluxo 1 x Fluxo 2 e um grafico IoU x acerto de
classificacao. Le' de outputs/results/<run>/.

Uso:
    python scripts/08_consolidate_results.py
    python scripts/08_consolidate_results.py --runs faster_rcnn_resnet50_run1 yolov8s_run1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from logoloc.config import load_project_config
from logoloc.eval.consolidate import build_comparison_table, load_saved_evaluation, plot_iou_vs_classification
from logoloc.eval.sensitivity import plot_sensitivity_from_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--runs", nargs="*", default=None, help="nomes das runs em outputs/results/; default = todas")
    args = parser.parse_args()

    cfg = load_project_config(args.base_config)
    results_dir = cfg.resolved_path(cfg.paths.outputs_dir) / "results"

    run_names = args.runs or [p.name for p in results_dir.iterdir() if p.is_dir()]
    if not run_names:
        logger.error("Nenhum resultado encontrado em %s. Rode scripts/06 ou 07 primeiro.", results_dir)
        return

    evaluations = [load_saved_evaluation(results_dir / name) for name in run_names]
    logger.info("Consolidando %d modelos: %s", len(evaluations), run_names)

    table = build_comparison_table(evaluations, primary_iou=cfg.metrics.primary_iou_threshold)
    table_path = results_dir / "comparison_table.csv"
    table.to_csv(table_path, index=False)
    print(table.to_string(index=False))
    logger.info("Tabela comparativa salva em %s", table_path)

    plot_path = plot_iou_vs_classification(evaluations, results_dir / "iou_vs_classification.png")
    logger.info("Grafico IoU x classificacao salvo em %s", plot_path)

    sens_path = plot_sensitivity_from_dirs([results_dir / name for name in run_names], results_dir / "localization_sensitivity.png")
    if sens_path:
        logger.info("Grafico de sensibilidade salvo em %s", sens_path)


if __name__ == "__main__":
    main()
