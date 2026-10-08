import math
import random

import pytest

from logoloc.eval.degradation import MODOS, NIVEIS, degradar, iou_xyxy

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
    return externa[0] <= interna[0] + 1e-9 and externa[1] <= interna[1] + 1e-9 and externa[2] >= interna[2] - 1e-9 and externa[3] >= interna[3] - 1e-9


def _ampliacao_cabe(box, tau):
    s = 1 / math.sqrt(tau)
    return (box[2] - box[0]) * s <= LARGURA and (box[3] - box[1]) * s <= ALTURA


@pytest.mark.parametrize("modo", ["reducao", "deslocamento"])
@pytest.mark.parametrize("tau", NIVEIS)
@pytest.mark.parametrize("box", CAIXAS)
def test_atinge_iou_exata(box, tau, modo):
    nova, obtido = degradar(box, tau, modo, LARGURA, ALTURA)
    assert obtido == pytest.approx(tau, abs=1e-9)
    assert iou_xyxy(box, nova) == pytest.approx(tau, abs=1e-9)
    assert _dentro_da_imagem(nova)


@pytest.mark.parametrize("tau", NIVEIS)
@pytest.mark.parametrize("box", CAIXAS)
def test_ampliacao(box, tau):
    nova, obtido = degradar(box, tau, "ampliacao", LARGURA, ALTURA)
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
        nova, obtido = degradar(box, tau, modo, LARGURA, ALTURA)
        assert _dentro_da_imagem(nova)
        if modo == "ampliacao":
            assert _contem(nova, box)
            assert obtido >= tau - 1e-9
            if _ampliacao_cabe(box, tau):
                assert obtido == pytest.approx(tau, abs=1e-9)
        else:
            assert obtido == pytest.approx(tau, abs=1e-9)


def test_nivel_um_nao_altera():
    box = (10.0, 20.0, 110.0, 80.0)
    for modo in MODOS:
        assert degradar(box, 1.0, modo, LARGURA, ALTURA) == (box, 1.0)


def test_deterministico():
    box = (900.0, 100.0, 1020.0, 200.0)
    for modo in MODOS:
        assert degradar(box, 0.4, modo, LARGURA, ALTURA) == degradar(box, 0.4, modo, LARGURA, ALTURA)


def test_reducao_fica_dentro_da_original():
    box = (100.0, 100.0, 300.0, 200.0)
    nova, _ = degradar(box, 0.3, "reducao", LARGURA, ALTURA)
    assert _contem(box, nova)


def test_deslocamento_mantem_tamanho_quando_cabe():
    box = (400.0, 300.0, 500.0, 360.0)
    nova, _ = degradar(box, 0.5, "deslocamento", LARGURA, ALTURA)
    assert nova[2] - nova[0] == pytest.approx(100.0)
    assert (nova[1], nova[3]) == (box[1], box[3])


def test_ampliacao_encosta_na_borda_sem_sair():
    box = (900.0, 100.0, 1020.0, 200.0)
    nova, obtido = degradar(box, 0.5, "ampliacao", LARGURA, ALTURA)
    assert nova[2] == pytest.approx(LARGURA)
    assert obtido == pytest.approx(0.5, abs=1e-9)


def test_modo_invalido():
    with pytest.raises(ValueError):
        degradar((0.0, 0.0, 10.0, 10.0), 0.5, "distorcao", LARGURA, ALTURA)
