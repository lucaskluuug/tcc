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
    seed: int = 42
    score_threshold: float = 0.25

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def by_level(self) -> pd.DataFrame:
        df = self.as_frame()
        if df.empty:
            return df
        df = df.assign(acima_limiar=df["score"] > self.score_threshold)
        grouped = df.groupby("nivel_alvo").agg(
            n=("acerto", "size"),
            iou_medio=("iou_obtido", "mean"),
            acuracia=("acerto", "mean"),
            score_medio=("score", "mean"),
            taxa_acima_limiar=("acima_limiar", "mean"),
            taxa_fundo_vence=("fundo_vence", "mean"),
        )
        return grouped.reset_index().sort_values("nivel_alvo", ascending=False)


def run_sensitivity(
    run_name: str,
    classify: RoIClassifier,
    records: list[ImageRecord],
    levels: tuple[float, ...] = DEFAULT_LEVELS,
    seed: int = 42,
    score_threshold: float = 0.25,
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

    return SensitivityResult(
        run_name=run_name, levels=levels, rows=rows, seed=seed, score_threshold=score_threshold
    )


def save_sensitivity(result: SensitivityResult, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    caminho = output_dir / f"localization_sensitivity_seed{result.seed}.csv"
    result.by_level().to_csv(caminho, index=False)
    result.as_frame().to_csv(output_dir / f"localization_sensitivity_seed{result.seed}_raw.csv", index=False)
    return caminho


def aggregate_sensitivity(output_dir: str | Path, run_name: str) -> pd.DataFrame | None:
    output_dir = Path(output_dir)
    arquivos = sorted(p for p in output_dir.glob("localization_sensitivity_seed*.csv") if not p.stem.endswith("_raw"))
    if not arquivos:
        return None

    seeds = []
    frames = []
    for caminho in arquivos:
        seed = int(caminho.stem.rsplit("seed", 1)[1])
        seeds.append(seed)
        frames.append(pd.read_csv(caminho).assign(seed=seed))
    todos = pd.concat(frames, ignore_index=True)

    colunas = ["iou_medio", "acuracia", "score_medio", "taxa_acima_limiar", "taxa_fundo_vence"]
    colunas = [c for c in colunas if c in todos.columns]
    media = todos.groupby("nivel_alvo")[colunas].mean()
    desvio = todos.groupby("nivel_alvo")[["acuracia"]].std().rename(columns={"acuracia": "acuracia_std"})
    n = todos.groupby("nivel_alvo")["n"].first()
    resumo = media.join(desvio).join(n).reset_index().sort_values("nivel_alvo", ascending=False)
    resumo["acuracia_std"] = resumo["acuracia_std"].fillna(0.0)
    resumo["n_seeds"] = len(seeds)

    resumo.to_csv(output_dir / "localization_sensitivity.csv", index=False)
    with open(output_dir / "localization_sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_name": run_name,
                "seeds": seeds,
                "levels": sorted(resumo["nivel_alvo"].tolist(), reverse=True),
                "por_nivel": resumo.to_dict(orient="records"),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return resumo


def plot_sensitivity_from_dirs(run_dirs: list[Path], output_path: str | Path) -> Path | None:
    import matplotlib.pyplot as plt

    curvas = []
    for run_dir in run_dirs:
        caminho = Path(run_dir) / "localization_sensitivity.csv"
        if caminho.exists():
            curvas.append((Path(run_dir).name, pd.read_csv(caminho)))
    if not curvas:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    for nome, resumo in curvas:
        resumo = resumo.sort_values("iou_medio")
        n_seeds = int(resumo["n_seeds"].iloc[0]) if "n_seeds" in resumo.columns else 1
        rotulo = f"{nome} ({n_seeds} sementes)" if n_seeds > 1 else nome
        (linha,) = ax.plot(resumo["iou_medio"], resumo["acuracia"], marker="o", label=rotulo)
        if n_seeds > 1 and "acuracia_std" in resumo.columns:
            ax.fill_between(
                resumo["iou_medio"],
                resumo["acuracia"] - resumo["acuracia_std"],
                resumo["acuracia"] + resumo["acuracia_std"],
                color=linha.get_color(),
                alpha=0.15,
                linewidth=0,
            )

    ax.axvline(0.5, color="gray", linestyle=":", linewidth=1)
    ax.annotate("limiar de deteccao", xy=(0.5, 0.02), xytext=(4, 0), textcoords="offset points", fontsize=7, color="gray")
    ax.set_xlabel("IoU da caixa usada como proposta")
    ax.set_ylabel("Taxa de acerto de classificacao")
    ax.set_title("Degradacao da classificacao conforme a caixa piora (mesmas instancias)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
