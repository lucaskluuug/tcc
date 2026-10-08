#!/usr/bin/env python
"""
Consolida os resultados de scripts/11_run_experiments.py das runs informadas em
tabelas CSV, um grafico de degradacao e um relatorio em markdown.

Uso:
    python scripts/12_experiments_report.py --runs frcnn_r50_v1 frcnn_r101_v1
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from logoloc.config import load_project_config
from logoloc.eval.experiments_report import carregar, gerar_relatorio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--runs", nargs="*", default=None, help="default = todas as runs com resultados de experimentos")
    args = parser.parse_args()

    cfg = load_project_config(args.base_config)
    results_dir = cfg.resolved_path(cfg.paths.outputs_dir) / "results"
    nomes = args.runs or sorted(p.parent.name for p in results_dir.glob("*/experimentos/resumo.json"))
    if not nomes:
        logger.error("Nenhum resultado em %s/*/experimentos. Rode scripts/11 primeiro.", results_dir)
        return

    runs = [carregar(results_dir / nome) for nome in nomes]
    caminho = gerar_relatorio(runs, results_dir / "experimentos")
    print(caminho.read_text(encoding="utf-8"))
    logger.info("Relatorio salvo em %s", caminho)


if __name__ == "__main__":
    main()
