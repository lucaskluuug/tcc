from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

NOMES_MODO = {"reducao": "Redução", "deslocamento": "Deslocamento"}


def carregar(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    with open(run_dir / "experimentos" / "resumo.json", encoding="utf-8") as f:
        dados = json.load(f)
    dados["exp3"] = pd.read_csv(run_dir / "experimentos" / "exp3_resumo.csv")
    antigo = run_dir / "summary.json"
    if antigo.exists():
        with open(antigo, encoding="utf-8") as f:
            dados["recorte"] = json.load(f)
    return dados


def tabela_deteccao(runs: list[dict]) -> pd.DataFrame:
    linhas = []
    for r in runs:
        exp1 = r["experimento1"]
        d50, d75 = exp1["deteccao"]["0.5"], exp1["deteccao"].get("0.75", {})
        frac = exp1["decomposicao_fracoes"]
        linhas.append(
            {
                "modelo": r["run_name"],
                "mAP@0,5": d50["mAP"],
                "mAP@0,75": d75.get("mAP"),
                "precisao": d50["precisao"],
                "revocacao": d50["revocacao"],
                "f1": d50["f1"],
                "erro_classificacao": frac["classification_error"],
                "erro_localizacao": frac["localization_error"],
                "erro_ambos": frac["both_error"],
                "duplicata": frac["duplicate"],
                "fundo": frac["background_error"],
                "perdida": frac["missed"],
            }
        )
    return pd.DataFrame(linhas)


def tabela_exp1_vs_exp2(runs: list[dict]) -> pd.DataFrame:
    linhas = []
    for r in runs:
        modelo, ideal = r["experimento1"]["por_logotipo"], r["experimento2"]
        linhas.append(
            {
                "modelo": r["run_name"],
                "acuracia_exp1": modelo["acuracia"],
                "acuracia_exp2": ideal["acuracia"],
                "ganho_relativo": ideal["acuracia"] / modelo["acuracia"] - 1 if modelo["acuracia"] else None,
                "acima_limiar_exp1": modelo["taxa_acima_limiar"],
                "acima_limiar_exp2": ideal["taxa_acima_limiar"],
                "nao_localizado_exp1": modelo["taxa_nao_localizado"],
                "iou_mediano_exp1": modelo["iou_mediano"],
                "acuracia_exp2_acima_limiar": ideal["acuracia_acima_limiar"],
                "acuracia_exp2_abaixo_limiar": ideal["acuracia_abaixo_limiar"],
            }
        )
    return pd.DataFrame(linhas)


def tabela_protocolos(runs: list[dict]) -> pd.DataFrame | None:
    linhas = []
    for r in runs:
        antigo = r.get("recorte")
        if not antigo:
            continue
        linhas.append(
            {
                "modelo": r["run_name"],
                "recorte_margem_10": antigo["flow1"]["accuracy"],
                "recorte_sem_margem": antigo.get("flow1_tight", {}).get("accuracy"),
                "recorte_falha_deteccao": antigo["flow1"]["detection_failure_rate"],
                "injecao": r["experimento2"]["acuracia"],
            }
        )
    return pd.DataFrame(linhas) if linhas else None


def tabela_degradacao(runs: list[dict]) -> pd.DataFrame:
    partes = []
    for r in runs:
        df = r["exp3"].copy()
        df["modelo"] = r["run_name"]
        partes.append(df)
    df = pd.concat(partes)
    largo = df.pivot_table(index=["modelo", "nivel_alvo"], columns="modo", values=["acuracia", "taxa_acima_limiar"])
    largo.columns = [f"{metrica}_{modo}" for metrica, modo in largo.columns]
    return largo.reset_index().sort_values(["modelo", "nivel_alvo"], ascending=[True, False])


def grafico_degradacao(runs: list[dict], caminho: str | Path, limiar_treino: float = 0.5) -> Path:
    import matplotlib.pyplot as plt

    fig, eixos = plt.subplots(1, len(runs), figsize=(6 * len(runs), 4.5), sharey=True, squeeze=False)
    cores = {"reducao": "#1f77b4", "deslocamento": "#d62728"}
    for ax, r in zip(eixos[0], runs):
        for modo, grupo in r["exp3"].groupby("modo"):
            grupo = grupo.sort_values("nivel_alvo")
            nome, cor = NOMES_MODO.get(modo, modo), cores.get(modo)
            ax.plot(grupo["nivel_alvo"], grupo["acuracia"], marker="o", color=cor, label=f"{nome}: acurácia")
            ax.plot(
                grupo["nivel_alvo"], grupo["taxa_acima_limiar"], marker="s", linestyle="--", color=cor,
                alpha=0.7, label=f"{nome}: viraria detecção",
            )
        exp1 = r["experimento1"]["por_logotipo"]
        ax.plot(exp1["iou_mediano"], exp1["acuracia"], marker="*", markersize=14, color="black", linestyle="none",
                label="Detector (Exp. 1)")
        ax.axvline(limiar_treino, color="gray", linestyle=":", linewidth=1)
        ax.annotate("limiar de treino", xy=(limiar_treino, 0.95), xytext=(4, 0), textcoords="offset points",
                    fontsize=8, color="gray")
        ax.set_xlim(1.02, 0.28)
        ax.set_ylim(0, 1)
        ax.set_xlabel("IoU da caixa fornecida")
        ax.set_title(r["run_name"])
        ax.grid(alpha=0.3)
    eixos[0][0].set_ylabel("Fração das instâncias")
    eixos[0][-1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=200)
    plt.close(fig)
    return caminho


def _markdown(df: pd.DataFrame) -> str:
    def fmt(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else f"{v:.3f}"
        return str(v)

    cab = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    corpo = ["| " + " | ".join(fmt(v) for v in linha) + " |" for linha in df.itertuples(index=False)]
    return "\n".join([cab, sep, *corpo])


def gerar_relatorio(runs: list[dict], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tabelas = {
        "Experimento 1: detecção (mAP com confiança > 0,05, demais métricas com confiança > 0,25)": tabela_deteccao(runs),
        "Experimentos 1 e 2: marca atribuída a cada logotipo, sem limiar de confiança": tabela_exp1_vs_exp2(runs),
        "Experimento 2: recorte versus injeção": tabela_protocolos(runs),
        "Experimento 3: degradação por nível de IoU": tabela_degradacao(runs),
    }
    nomes_csv = ["tabela_exp1_deteccao", "tabela_exp1_vs_exp2", "tabela_exp2_protocolos", "tabela_exp3_degradacao"]

    linhas = ["# Resultados dos experimentos", ""]
    for (titulo, df), nome in zip(tabelas.items(), nomes_csv):
        if df is None:
            continue
        df.to_csv(output_dir / f"{nome}.csv", index=False)
        linhas += [f"## {titulo}", "", _markdown(df), ""]

    grafico_degradacao(runs, output_dir / "degradacao.png")
    linhas += ["## Curvas de degradação", "", "![Degradação](degradacao.png)", ""]

    caminho = output_dir / "relatorio_experimentos.md"
    caminho.write_text("\n".join(linhas), encoding="utf-8")
    return caminho
