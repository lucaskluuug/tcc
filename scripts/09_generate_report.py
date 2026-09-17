#!/usr/bin/env python
"""
Uso:
    python scripts/09_generate_report.py
    python scripts/09_generate_report.py --runs frcnn_r50_v1 frcnn_r101_v1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from logoloc.config import load_project_config
from logoloc.eval.consolidate import build_comparison_table, load_saved_evaluation
from logoloc.eval.report import generate_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_project_config(args.base_config)
    results_dir = cfg.resolved_path(cfg.paths.outputs_dir) / "results"
    runs_dir = cfg.resolved_path(cfg.paths.outputs_dir) / "runs"

    run_names = args.runs or sorted(p.name for p in results_dir.iterdir() if p.is_dir())
    if not run_names:
        logger.error("Nenhum resultado encontrado em %s.", results_dir)
        return

    evaluations = [load_saved_evaluation(results_dir / name) for name in run_names]
    comparison = build_comparison_table(evaluations, primary_iou=cfg.metrics.primary_iou_threshold)

    output_path = Path(args.output) if args.output else results_dir / "relatorio.md"
    generate_report(cfg, comparison, run_names, results_dir, runs_dir, output_path)
    logger.info("Relatorio gerado em %s", output_path)


if __name__ == "__main__":
    main()
