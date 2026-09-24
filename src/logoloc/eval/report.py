from __future__ import annotations

import datetime
import json
from pathlib import Path

import pandas as pd

from ..config import ProjectConfig


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_sem dados_"
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def _dict_to_markdown(d: dict) -> str:
    rows = [f"| {k} | {_fmt(v)} |" for k, v in d.items()]
    return "\n".join(["| parametro | valor |", "| --- | --- |"] + rows)


def _load_summary(results_dir: Path, run_name: str) -> dict:
    with open(results_dir / run_name / "summary.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _load_run_config(runs_dir: Path, run_name: str) -> dict | None:
    path = runs_dir / run_name / "run_config.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _model_section(results_dir: Path, runs_dir: Path, run_name: str) -> str:
    summary = _load_summary(results_dir, run_name)
    lines = [f"## {run_name}", ""]

    lines.append("### Fluxo 1: classificacao com localizacao ideal")
    lines.append("")
    lines.append(_dict_to_markdown({
        "acuracia": summary["flow1"]["accuracy"],
        "f1_macro": summary["flow1"]["macro_f1"],
        "taxa_falha_deteccao": summary["flow1"]["detection_failure_rate"],
    }))
    if "flow1_tight" in summary:
        lines.append("")
        lines.append("Variante sem padding (sensibilidade):")
        lines.append("")
        lines.append(_dict_to_markdown({
            "acuracia": summary["flow1_tight"]["accuracy"],
            "f1_macro": summary["flow1_tight"]["macro_f1"],
        }))

    lines.append("")
    lines.append("### Fluxo 2: deteccao ponta a ponta")
    lines.append("")
    for thr, r in summary["flow2"].items():
        lines.append(f"IoU >= {thr}:")
        lines.append("")
        lines.append(_dict_to_markdown({
            "mAP": r["mAP"],
            "precisao": r["precision"],
            "revocacao": r["recall"],
            "f1": r["f1"],
            "vp": r["tp"],
            "fp": r["fp"],
            "fn": r["fn"],
            "iou_medio_vp": r["mean_iou_tp"],
        }))
        lines.append("")

    if "flow2_error_decomposition" in summary:
        lines.append("### Decomposicao de erro (Fluxo 2)")
        lines.append("")
        lines.append(_dict_to_markdown(summary["flow2_error_decomposition"]))
        lines.append("")

    sens_path = results_dir / run_name / "localization_sensitivity.json"
    if sens_path.exists():
        with open(sens_path, "r", encoding="utf-8") as f:
            sens = json.load(f)
        por_nivel = sens.get("por_nivel", [])
        if por_nivel:
            seeds = sens.get("seeds", [])
            lines.append("### Sensibilidade ao erro de localizacao")
            lines.append("")
            lines.append("Caixas anotadas perturbadas ate niveis controlados de IoU e injetadas na RoI head.")
            lines.append("As mesmas instancias aparecem em todos os niveis.")
            if len(seeds) > 1:
                lines.append(
                    f"Media de {len(seeds)} sementes de perturbacao; acuracia_std e o desvio padrao entre sementes."
                )
            lines.append("")
            tabela = pd.DataFrame(por_nivel)
            colunas = ["nivel_alvo", "n", "iou_medio", "acuracia"]
            if len(seeds) > 1 and "acuracia_std" in tabela.columns:
                colunas.append("acuracia_std")
            if "taxa_acima_limiar" in tabela.columns:
                colunas.append("taxa_acima_limiar")
            colunas.append("taxa_fundo_vence")
            lines.append(_df_to_markdown(tabela[[c for c in colunas if c in tabela.columns]]))
            lines.append("")
            if "taxa_acima_limiar" in tabela.columns:
                lines.append(
                    "taxa_acima_limiar: fracao de caixas cujo score de classe passa do limiar de "
                    "confianca do detector, ou seja, que virariam uma deteccao emitida."
                )
                lines.append("")
            topo = next((r for r in por_nivel if r["nivel_alvo"] == 1.0), None)
            base = min(por_nivel, key=lambda r: r["nivel_alvo"])
            if topo and base and topo is not base:
                queda = topo["acuracia"] - base["acuracia"]
                lines.append(
                    f"Queda de {queda:.3f} na acuracia entre a caixa anotada "
                    f"(IoU 1.0) e IoU {base['nivel_alvo']:.1f}."
                )
                lines.append("")
            if (results_dir / run_name / "localization_sensitivity.png").exists():
                lines.append(
                    f"![Sensibilidade a localizacao - {run_name}]({run_name}/localization_sensitivity.png)"
                )
                lines.append("")

    confusion_path = results_dir / run_name / "flow1_confusion_matrix.png"
    if confusion_path.exists():
        lines.append(f"![Matriz de confusao - {run_name}]({run_name}/flow1_confusion_matrix.png)")
        lines.append("")

    run_config = _load_run_config(runs_dir, run_name)
    if run_config is not None:
        lines.append("### Configuracao do treino")
        lines.append("")
        campos = {}
        total_epochs = run_config.get("total_epochs")
        if total_epochs:
            campos["epocas"] = total_epochs
        campos["tempo_treino"] = run_config.get("training_time_human")
        start_epoch = run_config.get("start_epoch") or 0
        if start_epoch:
            campos["observacao"] = (
                f"treino retomado da epoca {start_epoch}; o tempo acima cobre apenas "
                f"as {run_config.get('epochs_ran')} epocas desta sessao"
            )
        campos["gpu"] = run_config.get("hardware", {}).get("gpu")
        campos["cuda_disponivel"] = run_config.get("hardware", {}).get("cuda_available")
        campos["torch"] = run_config.get("hardware", {}).get("torch_version")
        lines.append(_dict_to_markdown(campos))
        lines.append("")

    return "\n".join(lines)


def generate_report(
    project_cfg: ProjectConfig,
    comparison: pd.DataFrame,
    run_names: list[str],
    results_dir: Path,
    runs_dir: Path,
    output_path: Path,
) -> Path:
    ds = project_cfg.dataset
    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# Relatorio de resultados: deteccao de logotipos",
        "",
        f"Gerado em {generated_at}.",
        "",
        "## Dataset",
        "",
        _dict_to_markdown({
            "nome": ds.name,
            "classes": ds.num_classes,
            "treino_por_classe": ds.images_per_class_train,
            "validacao_por_classe": ds.images_per_class_val,
            "teste_por_classe": ds.images_per_class_test,
            "sem_logo_teste": ds.no_logo_images_test,
        }),
        "",
        "## Comparacao entre modelos",
        "",
        _df_to_markdown(comparison),
        "",
    ]

    plot_path = results_dir / "iou_vs_classification.png"
    if plot_path.exists():
        lines.append("![IoU vs acerto de classificacao](iou_vs_classification.png)")
        lines.append("")

    for run_name in run_names:
        lines.append(_model_section(results_dir, runs_dir, run_name))
        lines.append("")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
