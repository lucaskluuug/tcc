from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path

import pandas as pd

from ..data.records import ImageRecord
from .degradation import MODOS, NIVEIS, degradar, iou_xyxy
from .metrics import GTBox, PredBox, compute_detection_prf1, compute_map, error_decomposition
from .predictors import FullPredictor
from .roi_classifier import RoIClassifier

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Limiares:
    operacao: float = 0.25
    map: float = 0.05
    piso_fundo: float = 0.1
    ious_map: tuple[float, ...] = (0.5, 0.75)
    iou_principal: float = 0.5


def anotacoes(records: list[ImageRecord]) -> list[GTBox]:
    return [
        GTBox(image_id=r.image_id, class_name=inst.class_name, box=inst.bbox.as_xyxy())
        for r in records
        for inst in r.instances
    ]


def detectar(records: list[ImageRecord], predict_fn: FullPredictor, log_every: int = 500) -> list[PredBox]:
    preds: list[PredBox] = []
    for pos, record in enumerate(records, 1):
        for det in predict_fn(record.image_path):
            preds.append(PredBox(image_id=record.image_id, class_name=det.class_name, box=det.box_xyxy, score=det.score))
        if log_every and pos % log_every == 0:
            logger.info("  deteccao: %d/%d imagens", pos, len(records))
    return preds


def resposta_por_logotipo(records: list[ImageRecord], preds: list[PredBox], limiares: Limiares) -> pd.DataFrame:
    por_imagem: dict[str, list[PredBox]] = {}
    for p in preds:
        por_imagem.setdefault(p.image_id, []).append(p)

    linhas = []
    for record in records:
        candidatas = por_imagem.get(record.image_id, [])
        for idx, inst in enumerate(record.instances):
            gt = inst.bbox.as_xyxy()
            sobrepostas = [(p, iou_xyxy(p.box, gt)) for p in candidatas]
            sobrepostas = [(p, iou) for p, iou in sobrepostas if iou >= limiares.piso_fundo]
            if sobrepostas:
                escolhida, iou = max(sobrepostas, key=lambda par: par[0].score)
                classe, score = escolhida.class_name, escolhida.score
            else:
                classe, score, iou = None, 0.0, 0.0
            linhas.append(
                {
                    "image_id": record.image_id,
                    "instancia": idx,
                    "classe_verdadeira": inst.class_name,
                    "localizado": classe is not None,
                    "classe_predita": classe,
                    "score": score,
                    "iou": iou,
                    "acerto": classe == inst.class_name,
                    "acima_limiar": score > limiares.operacao,
                }
            )
    return pd.DataFrame(linhas)


def experimento1(
    records: list[ImageRecord], preds: list[PredBox], class_names: list[str], limiares: Limiares
) -> tuple[dict, pd.DataFrame]:
    gts = anotacoes(records)
    preds_map = [p for p in preds if p.score > limiares.map]
    preds_operacao = [p for p in preds if p.score > limiares.operacao]

    deteccao = {}
    for thr in limiares.ious_map:
        mean_ap, ap_por_classe = compute_map(gts, preds_map, class_names, thr)
        prf1 = compute_detection_prf1(gts, preds_operacao, thr)
        deteccao[str(thr)] = {
            "mAP": mean_ap,
            "precisao": prf1.precision,
            "revocacao": prf1.recall,
            "f1": prf1.f1,
            "vp": prf1.tp,
            "fp": prf1.fp,
            "fn": prf1.fn,
            "ap_por_classe": ap_por_classe,
        }

    decomposicao = error_decomposition(
        gts, preds_operacao, fg_iou_threshold=limiares.iou_principal, bg_iou_floor=limiares.piso_fundo
    )

    por_logotipo = resposta_por_logotipo(records, preds, limiares)
    localizados = por_logotipo[por_logotipo["localizado"]]
    resumo = {
        "deteccao": deteccao,
        "decomposicao_erros": dataclasses.asdict(decomposicao),
        "decomposicao_fracoes": decomposicao.fractions(),
        "por_logotipo": {
            "n": int(len(por_logotipo)),
            "acuracia": float(por_logotipo["acerto"].mean()),
            "taxa_acima_limiar": float(por_logotipo["acima_limiar"].mean()),
            "taxa_nao_localizado": float(1 - por_logotipo["localizado"].mean()),
            "iou_mediano": float(localizados["iou"].median()) if len(localizados) else 0.0,
            "iou_medio": float(localizados["iou"].mean()) if len(localizados) else 0.0,
        },
    }
    return resumo, por_logotipo


def experimento3(
    records: list[ImageRecord],
    classify: RoIClassifier,
    limiares: Limiares,
    modos: tuple[str, ...] = MODOS,
    niveis: tuple[float, ...] = NIVEIS,
    log_every: int = 100,
) -> pd.DataFrame:
    com_logo = [r for r in records if not r.is_no_logo]
    linhas: list[dict] = []
    for pos, record in enumerate(com_logo, 1):
        caixas, meta = [], []
        for idx, inst in enumerate(record.instances):
            gt = inst.bbox.as_xyxy()
            for modo in modos:
                for nivel in niveis:
                    caixa, obtido = degradar(gt, nivel, modo, record.width, record.height)
                    caixas.append(caixa)
                    meta.append((idx, modo, nivel, obtido))

        for (idx, modo, nivel, obtido), pred in zip(meta, classify(record.image_path, caixas)):
            verdadeira = record.instances[idx].class_name
            linhas.append(
                {
                    "image_id": record.image_id,
                    "instancia": idx,
                    "classe_verdadeira": verdadeira,
                    "modo": modo,
                    "nivel_alvo": nivel,
                    "iou_obtido": obtido,
                    "limitado_pela_imagem": obtido > nivel + 1e-6,
                    "classe_predita": pred.class_name,
                    "score": pred.score,
                    "prob_fundo": pred.background_score,
                    "acerto": pred.class_name == verdadeira,
                    "acima_limiar": pred.score > limiares.operacao,
                    "fundo_vence": pred.background_score > pred.score,
                }
            )
        if log_every and pos % log_every == 0:
            logger.info("  injecao: %d/%d imagens", pos, len(com_logo))
    return pd.DataFrame(linhas)


def resumo_experimento3(por_caixa: pd.DataFrame) -> pd.DataFrame:
    agrupado = por_caixa.groupby(["modo", "nivel_alvo"]).agg(
        n=("acerto", "size"),
        iou_medio=("iou_obtido", "mean"),
        acuracia=("acerto", "mean"),
        taxa_acima_limiar=("acima_limiar", "mean"),
        taxa_fundo_vence=("fundo_vence", "mean"),
        taxa_limitada=("limitado_pela_imagem", "mean"),
    )
    return agrupado.reset_index().sort_values(["modo", "nivel_alvo"], ascending=[True, False])


def resumo_experimento2(por_caixa: pd.DataFrame) -> dict:
    ideal = por_caixa[(por_caixa["nivel_alvo"] >= 1.0) & (por_caixa["modo"] == por_caixa["modo"].iloc[0])]
    acima, abaixo = ideal[ideal["acima_limiar"]], ideal[~ideal["acima_limiar"]]
    return {
        "n": int(len(ideal)),
        "acuracia": float(ideal["acerto"].mean()),
        "taxa_acima_limiar": float(ideal["acima_limiar"].mean()),
        "acuracia_acima_limiar": float(acima["acerto"].mean()) if len(acima) else None,
        "acuracia_abaixo_limiar": float(abaixo["acerto"].mean()) if len(abaixo) else None,
        "taxa_fundo_vence": float(ideal["fundo_vence"].mean()),
    }


def salvar(
    output_dir: str | Path,
    run_name: str,
    limiares: Limiares,
    resumo_exp1: dict,
    por_logotipo: pd.DataFrame,
    por_caixa: pd.DataFrame,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    por_logotipo.to_csv(output_dir / "exp1_por_logotipo.csv", index=False)
    por_caixa.to_csv(output_dir / "exp3_por_caixa.csv", index=False)
    resumo_exp3 = resumo_experimento3(por_caixa)
    resumo_exp3.to_csv(output_dir / "exp3_resumo.csv", index=False)

    resumo = {
        "run_name": run_name,
        "limiares": dataclasses.asdict(limiares),
        "experimento1": resumo_exp1,
        "experimento2": resumo_experimento2(por_caixa),
    }
    caminho = output_dir / "resumo.json"
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    return caminho
