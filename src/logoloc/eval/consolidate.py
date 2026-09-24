from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.crops import CropRecord
from ..data.records import ImageRecord
from .confusion_matrix import plot_confusion_matrix
from .flow1_classification import Flow1Result, evaluate_flow1
from .flow2_detection import Flow2Result, evaluate_flow2
from .metrics import DetectionPRF1, ErrorDecomposition
from .predictors import FullPredictor, top1_from_full


@dataclasses.dataclass
class ModelEvaluation:
    run_name: str
    flow1: Flow1Result
    flow2: Flow2Result
    flow1_tight: Flow1Result | None = None


def evaluate_model(
    run_name: str,
    predict_fn: FullPredictor,
    crops_p3: list[CropRecord],
    records_p3: list[ImageRecord],
    class_names: list[str],
    iou_thresholds: tuple[float, ...],
    primary_iou_threshold: float,
    background_iou_floor: float,
    crops_p3_tight: list[CropRecord] | None = None,
) -> ModelEvaluation:
    flow1 = evaluate_flow1(crops_p3, top1_from_full(predict_fn), class_names)
    flow2 = evaluate_flow2(
        records_p3, predict_fn, class_names, iou_thresholds, primary_iou_threshold, background_iou_floor
    )
    flow1_tight = (
        evaluate_flow1(crops_p3_tight, top1_from_full(predict_fn), class_names) if crops_p3_tight is not None else None
    )
    return ModelEvaluation(run_name=run_name, flow1=flow1, flow2=flow2, flow1_tight=flow1_tight)


def _binned_iou_classification(pairs: list[tuple[float, bool]], n_bins: int = 10) -> pd.DataFrame:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ious = np.array([p[0] for p in pairs]) if pairs else np.array([])
    correct = np.array([p[1] for p in pairs], dtype=float) if pairs else np.array([])
    rows = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (ious >= lo) & (ious < hi)
        n = int(mask.sum())
        rows.append({"iou_min": lo, "iou_max": hi, "n": n, "taxa_acerto_classificacao": correct[mask].mean() if n else float("nan")})
    return pd.DataFrame(rows)


def save_evaluation(evaluation: ModelEvaluation, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(evaluation.flow1.report, output_dir / "flow1_confusion_matrix.png")
    evaluation.flow1.report.as_table().to_csv(output_dir / "flow1_per_class_report.csv", index=False)

    summary = {
        "run_name": evaluation.run_name,
        "flow1": {
            "accuracy": evaluation.flow1.report.accuracy,
            "macro_f1": evaluation.flow1.report.macro_f1(),
            "detection_failure_rate": evaluation.flow1.detection_failure_rate,
        },
        "flow2": {
            str(thr): {
                "mAP": r.mean_ap,
                "precision": r.prf1.precision,
                "recall": r.prf1.recall,
                "f1": r.prf1.f1,
                "tp": r.prf1.tp,
                "fp": r.prf1.fp,
                "fn": r.prf1.fn,
                "mean_iou_tp": r.prf1.mean_iou_tp,
                "ap_per_class": r.ap_per_class,
            }
            for thr, r in evaluation.flow2.results_by_iou.items()
        },
        "flow2_error_decomposition": dataclasses.asdict(evaluation.flow2.error_decomposition),
    }
    if evaluation.flow1_tight is not None:
        summary["flow1_tight"] = {
            "accuracy": evaluation.flow1_tight.report.accuracy,
            "macro_f1": evaluation.flow1_tight.report.macro_f1(),
            "detection_failure_rate": evaluation.flow1_tight.detection_failure_rate,
        }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pd.DataFrame(evaluation.flow2.iou_classification_pairs, columns=["iou", "class_correct"]).to_csv(
        output_dir / "flow2_iou_classification_pairs.csv", index=False
    )
    _binned_iou_classification(evaluation.flow2.iou_classification_pairs).to_csv(
        output_dir / "flow2_iou_classification_bins.csv", index=False
    )


@dataclasses.dataclass
class _LoadedFlow1Report:
    accuracy: float
    _macro_f1: float

    def macro_f1(self) -> float:
        return self._macro_f1


@dataclasses.dataclass
class LoadedFlow1Result:
    report: _LoadedFlow1Report
    detection_failure_rate: float


@dataclasses.dataclass
class LoadedFlow2IouResult:
    mean_ap: float
    prf1: DetectionPRF1


@dataclasses.dataclass
class LoadedFlow2Result:
    results_by_iou: dict[float, LoadedFlow2IouResult]
    iou_classification_pairs: list[tuple[float, bool]]
    error_decomposition: ErrorDecomposition | None


@dataclasses.dataclass
class LoadedModelEvaluation:
    run_name: str
    flow1: LoadedFlow1Result
    flow2: LoadedFlow2Result
    flow1_tight: LoadedFlow1Result | None = None


def load_saved_evaluation(output_dir: str | Path) -> LoadedModelEvaluation:
    output_dir = Path(output_dir)
    with open(output_dir / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    flow1 = LoadedFlow1Result(
        report=_LoadedFlow1Report(accuracy=summary["flow1"]["accuracy"], _macro_f1=summary["flow1"]["macro_f1"]),
        detection_failure_rate=summary["flow1"]["detection_failure_rate"],
    )

    results_by_iou = {}
    for thr_str, r in summary["flow2"].items():
        prf1 = DetectionPRF1(
            precision=r["precision"], recall=r["recall"], f1=r["f1"],
            tp=r["tp"], fp=r["fp"], fn=r["fn"], mean_iou_tp=r["mean_iou_tp"],
        )
        results_by_iou[float(thr_str)] = LoadedFlow2IouResult(mean_ap=r["mAP"], prf1=prf1)

    pairs_path = output_dir / "flow2_iou_classification_pairs.csv"
    pairs_df = pd.read_csv(pairs_path)
    pairs = list(zip(pairs_df["iou"].tolist(), pairs_df["class_correct"].astype(bool).tolist()))

    errors = ErrorDecomposition(**summary["flow2_error_decomposition"]) if "flow2_error_decomposition" in summary else None
    flow2 = LoadedFlow2Result(results_by_iou=results_by_iou, iou_classification_pairs=pairs, error_decomposition=errors)

    flow1_tight = None
    if "flow1_tight" in summary:
        flow1_tight = LoadedFlow1Result(
            report=_LoadedFlow1Report(accuracy=summary["flow1_tight"]["accuracy"], _macro_f1=summary["flow1_tight"]["macro_f1"]),
            detection_failure_rate=summary["flow1_tight"]["detection_failure_rate"],
        )

    return LoadedModelEvaluation(run_name=summary["run_name"], flow1=flow1, flow2=flow2, flow1_tight=flow1_tight)


def build_comparison_table(evaluations: list[ModelEvaluation], primary_iou: float) -> pd.DataFrame:
    rows = []
    for ev in evaluations:
        flow2 = ev.flow2.results_by_iou[primary_iou]
        flow1_acc = ev.flow1.report.accuracy
        decomp = ev.flow2.error_decomposition
        bem_localizadas = (decomp.true_positive + decomp.classification_error) if decomp else 0
        flow2_cls_acc = decomp.true_positive / bem_localizadas if bem_localizadas else None
        falha = ev.flow1.detection_failure_rate
        row = {
            "modelo": ev.run_name,
            "fluxo1_acuracia": flow1_acc,
            "fluxo1_taxa_falha_deteccao": falha,
            "fluxo1_acuracia_dado_deteccao": flow1_acc / (1 - falha) if falha < 1 else None,
            "fluxo1_acuracia_sem_padding": ev.flow1_tight.report.accuracy if ev.flow1_tight else None,
            "fluxo1_f1_macro": ev.flow1.report.macro_f1(),
            f"fluxo2_mAP@{primary_iou}": flow2.mean_ap,
            "fluxo2_precisao": flow2.prf1.precision,
            "fluxo2_revocacao": flow2.prf1.recall,
            "fluxo2_f1": flow2.prf1.f1,
            "fluxo2_iou_medio_vp": flow2.prf1.mean_iou_tp,
            "fluxo2_acuracia_classif_dado_localizado": flow2_cls_acc,
        }
        if ev.flow2.error_decomposition is not None:
            frac = ev.flow2.error_decomposition.fractions()
            row["fluxo2_pct_erro_localizacao"] = frac["localization_error"]
            row["fluxo2_pct_erro_classificacao"] = frac["classification_error"]
            row["fluxo2_pct_erro_ambos"] = frac["both_error"]
            row["fluxo2_pct_deteccao_perdida"] = frac["missed"]
            row["fluxo2_pct_falso_positivo_fundo"] = frac["background_error"]
        rows.append(row)
    return pd.DataFrame(rows)


def plot_iou_vs_classification(
    evaluations: list[ModelEvaluation], output_path: str | Path, n_bins: int = 10, min_amostras: int = 20
) -> Path:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for ev in evaluations:
        pairs = ev.flow2.iou_classification_pairs
        if not pairs:
            continue
        ious = np.array([p[0] for p in pairs])
        correct = np.array([p[1] for p in pairs], dtype=float)
        rates = []
        contagens = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (ious >= lo) & (ious < hi)
            n = int(mask.sum())
            contagens.append(n)
            rates.append(correct[mask].mean() if n >= min_amostras else np.nan)
        (line,) = ax.plot(bin_centers, rates, marker="o", label=f"{ev.run_name} (Fluxo 2)")
        for centro, taxa, n in zip(bin_centers, rates, contagens):
            if not np.isnan(taxa):
                ax.annotate(
                    f"n={n}",
                    xy=(centro, taxa),
                    xytext=(0, -12),
                    textcoords="offset points",
                    fontsize=6,
                    ha="center",
                    color=line.get_color(),
                )
        ax.axhline(ev.flow1.report.accuracy, color=line.get_color(), linestyle="--", linewidth=1, alpha=0.7)
        ax.annotate(
            f"{ev.run_name} (teto Fluxo 1)",
            xy=(1.0, ev.flow1.report.accuracy),
            xytext=(-4, 4),
            textcoords="offset points",
            ha="right",
            fontsize=7,
            color=line.get_color(),
        )

    ax.set_xlabel("IoU da caixa predita (faixa)")
    ax.set_ylabel("Taxa de acerto de classificacao")
    ax.set_title(
        f"IoU vs. acerto de classificacao (Fluxo 2). Linha tracejada: teto do Fluxo 1\n"
        f"faixas com menos de {min_amostras} amostras omitidas",
        fontsize=10,
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
