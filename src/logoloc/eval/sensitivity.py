from __future__ import annotations

import dataclasses
import json
import logging
import random
from pathlib import Path

import pandas as pd

from ..data.records import ImageRecord
from .jitter import jitter_to_iou
from .roi_classifier import RoIClassifier

logger = logging.getLogger(__name__)

DEFAULT_LEVELS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)


@dataclasses.dataclass
class SensitivityResult:
    run_name: str
    levels: tuple[float, ...]
    rows: list[dict]

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def by_level(self) -> pd.DataFrame:
        df = self.as_frame()
        if df.empty:
            return df
        grouped = df.groupby("nivel_alvo").agg(
            n=("acerto", "size"),
            iou_medio=("iou_obtido", "mean"),
            acuracia=("acerto", "mean"),
            score_medio=("score", "mean"),
            taxa_fundo_vence=("fundo_vence", "mean"),
        )
        return grouped.reset_index().sort_values("nivel_alvo", ascending=False)


def run_sensitivity(
    run_name: str,
    classify: RoIClassifier,
    records: list[ImageRecord],
    levels: tuple[float, ...] = DEFAULT_LEVELS,
    seed: int = 42,
    log_every: int = 100,
) -> SensitivityResult:
    rng = random.Random(seed)
    rows: list[dict] = []
    com_logo = [r for r in records if not r.is_no_logo]
    logger.info("Sensibilidade de localizacao: %d imagens, %d niveis", len(com_logo), len(levels))

    for pos, record in enumerate(com_logo, 1):
        caixas: list[tuple[float, float, float, float]] = []
        meta: list[tuple[int, float, float]] = []

        for idx, inst in enumerate(record.instances):
            gt = inst.bbox.as_xyxy()
            for nivel in levels:
                caixa, obtido = jitter_to_iou(gt, nivel, record.width, record.height, rng)
                caixas.append(caixa)
                meta.append((idx, nivel, obtido))

        predicoes = classify(record.image_path, caixas)
        for (idx, nivel, obtido), pred in zip(meta, predicoes):
            verdadeira = record.instances[idx].class_name
            rows.append(
                {
                    "image_id": record.image_id,
                    "instancia": idx,
                    "classe_verdadeira": verdadeira,
                    "nivel_alvo": nivel,
                    "iou_obtido": obtido,
                    "classe_predita": pred.class_name,
                    "score": pred.score,
                    "fundo_vence": pred.background_score > pred.score,
                    "acerto": pred.class_name == verdadeira,
                }
            )

        if log_every and pos % log_every == 0:
            logger.info("  %d/%d imagens", pos, len(com_logo))

    return SensitivityResult(run_name=run_name, levels=levels, rows=rows)


def save_sensitivity(result: SensitivityResult, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result.as_frame().to_csv(output_dir / "localization_sensitivity_raw.csv", index=False)
    resumo = result.by_level()
    resumo.to_csv(output_dir / "localization_sensitivity.csv", index=False)

    with open(output_dir / "localization_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": result.run_name,
                "levels": list(result.levels),
                "por_nivel": resumo.to_dict(orient="records"),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return output_dir / "localization_sensitivity.csv"


def plot_sensitivity(results: list[SensitivityResult], output_path: str | Path) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for result in results:
        resumo = result.by_level()
        if resumo.empty:
            continue
        ax.plot(resumo["iou_medio"], resumo["acuracia"], marker="o", label=result.run_name)
        for _, linha in resumo.iterrows():
            if linha["nivel_alvo"] == 1.0:
                ax.annotate(
                    "caixa anotada",
                    xy=(linha["iou_medio"], linha["acuracia"]),
                    xytext=(-6, 8),
                    textcoords="offset points",
                    fontsize=7,
                    ha="right",
                )

    ax.set_xlabel("IoU da caixa usada como proposta")
    ax.set_ylabel("Taxa de acerto de classificacao")
    ax.set_title("Degradacao da classificacao conforme a caixa piora")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
