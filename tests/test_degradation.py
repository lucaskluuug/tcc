import math
import random

import pytest

from logoloc.eval.degradation import DIRECOES, MODOS, NIVEIS, degradar, iou_xyxy

LARGURA, ALTURA = 1024, 768

CAIXAS = [
    (400.0, 300.0, 500.0, 360.0),
    (900.0, 100.0, 1020.0, 200.0),
    (2.0, 50.0, 150.0, 120.0),
    (0.0, 0.0, 1024.0, 768.0),
    (100.0, 100.0, 950.0, 400.0),
]


def _dentro_da_imagem(box):
    return -1e-9 <= box[0] < box[2] <= LARGURA + 1e-9 and -1e-9 <= box[1] < box[3] <= ALTURA + 1e-9


def _contem(externa, interna):
    return all(
        (
            externa[0] <= interna[0] + 1e-9,
            externa[1] <= interna[1] + 1e-9,
            externa[2] >= interna[2] - 1e-9,
            externa[3] >= interna[3] - 1e-9,
        )
    )


def _ampliacao_cabe(box, tau):
    s = 1 / math.sqrt(tau)
    return (box[2] - box[0]) * s <= LARGURA and (box[3] - box[1]) * s <= ALTURA


def _mesmo_tamanho(a, b):
    return math.isclose(a[2] - a[0], b[2] - b[0]) and math.isclose(a[3] - a[1], b[3] - b[1])


@pytest.mark.parametrize("modo", ["reducao", "deslocamento"])
@pytest.mark.parametrize("tau", NIVEIS)
@pytest.mark.parametrize("box", CAIXAS)
def test_atinge_iou_exata(box, tau, modo):
    caixas = degradar(box, tau, modo, LARGURA, ALTURA)
    assert caixas
    for _, nova, obtido in caixas:
        assert obtido == pytest.approx(tau, abs=1e-9)
        assert iou_xyxy(box, nova) == pytest.approx(tau, abs=1e-9)
        assert _dentro_da_imagem(nova)


def test_deslocamento_usa_as_oito_direcoes_quando_cabem():
    box = (400.0, 300.0, 500.0, 360.0)
    caixas = degradar(box, 0.5, "deslocamento", LARGURA, ALTURA)
    assert sorted(d for d, _, _ in caixas) == sorted(DIRECOES)
    for _, nova, _ in caixas:
        assert _mesmo_tamanho(nova, box)


def test_deslocamento_descarta_direcoes_que_saem_da_imagem():
    box = (900.0, 100.0, 1020.0, 200.0)
    direcoes = {d for d, _, _ in degradar(box, 0.5, "deslocamento", LARGURA, ALTURA)}
    assert "esquerda" in direcoes and "direita" not in direcoes


def test_deslocamento_sem_direcao_livre_corta_na_borda():
    box = (0.0, 0.0, 1024.0, 768.0)
    [(direcao, _, obtido)] = degradar(box, 0.5, "deslocamento", LARGURA, ALTURA)
    assert direcao == "borda"
    assert obtido == pytest.approx(0.5, abs=1e-9)


def test_diagonal_anda_a_mesma_fracao_nos_dois_eixos():
    box = (400.0, 300.0, 500.0, 360.0)
    caixas = {d: nova for d, nova, _ in degradar(box, 0.5, "deslocamento", LARGURA, ALTURA)}
    nova = caixas["baixo_direita"]
    assert (nova[0] - box[0]) / 100 == pytest.approx((nova[1] - box[1]) / 60)


@pytest.mark.parametrize("tau", NIVEIS)
@pytest.mark.parametrize("box", CAIXAS)
def test_ampliacao(box, tau):
    [(_, nova, obtido)] = degradar(box, tau, "ampliacao", LARGURA, ALTURA)
    assert _dentro_da_imagem(nova)
    assert _contem(nova, box)
    if _ampliacao_cabe(box, tau):
        assert obtido == pytest.approx(tau, abs=1e-9)
    else:
        assert obtido > tau


@pytest.mark.parametrize("modo", MODOS)
def test_caixas_aleatorias(modo):
    rng = random.Random(0)
    for _ in range(2000):
        w, h = rng.uniform(5, LARGURA), rng.uniform(5, ALTURA)
        x1, y1 = rng.uniform(0, LARGURA - w), rng.uniform(0, ALTURA - h)
        box = (x1, y1, x1 + w, y1 + h)
        tau = rng.uniform(0.05, 0.99)
        for direcao, nova, obtido in degradar(box, tau, modo, LARGURA, ALTURA):
            assert _dentro_da_imagem(nova)
            if modo == "ampliacao":
                assert _contem(nova, box)
                assert obtido >= tau - 1e-9
                if _ampliacao_cabe(box, tau):
                    assert obtido == pytest.approx(tau, abs=1e-9)
            else:
                assert obtido == pytest.approx(tau, abs=1e-9)
            if modo == "deslocamento" and direcao != "borda":
                assert _mesmo_tamanho(nova, box)


def test_nivel_um_nao_altera():
    box = (10.0, 20.0, 110.0, 80.0)
    for modo in MODOS:
        assert degradar(box, 1.0, modo, LARGURA, ALTURA) == [("", box, 1.0)]


def test_deterministico():
    box = (900.0, 100.0, 1020.0, 200.0)
    for modo in MODOS:
        assert degradar(box, 0.4, modo, LARGURA, ALTURA) == degradar(box, 0.4, modo, LARGURA, ALTURA)


def test_reducao_fica_dentro_da_original():
    box = (100.0, 100.0, 300.0, 200.0)
    [(_, nova, _)] = degradar(box, 0.3, "reducao", LARGURA, ALTURA)
    assert _contem(box, nova)


def test_ampliacao_encosta_na_borda_sem_sair():
    box = (900.0, 100.0, 1020.0, 200.0)
    [(_, nova, obtido)] = degradar(box, 0.5, "ampliacao", LARGURA, ALTURA)
    assert nova[2] == pytest.approx(LARGURA)
    assert obtido == pytest.approx(0.5, abs=1e-9)


def test_modo_invalido():
    with pytest.raises(ValueError):
        degradar((0.0, 0.0, 10.0, 10.0), 0.5, "distorcao", LARGURA, ALTURA)
