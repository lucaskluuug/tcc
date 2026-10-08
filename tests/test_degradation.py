import random

import pytest

from logoloc.eval.degradation import MODOS, NIVEIS, degradar, iou_xyxy

LARGURA = 1024


def _dentro_da_imagem(box, largura):
    return 0.0 <= box[0] < box[2] <= largura + 1e-9 and box[1] < box[3]


@pytest.mark.parametrize("modo", MODOS)
@pytest.mark.parametrize("tau", NIVEIS)
@pytest.mark.parametrize(
    "box",
    [
        (400.0, 300.0, 500.0, 360.0),
        (900.0, 100.0, 1020.0, 200.0),
        (2.0, 50.0, 150.0, 120.0),
        (0.0, 0.0, 1024.0, 768.0),
        (100.0, 100.0, 950.0, 400.0),
    ],
)
def test_atinge_iou_exata(box, tau, modo):
    nova, obtido = degradar(box, tau, modo, LARGURA)
    assert obtido == pytest.approx(tau, abs=1e-9)
    assert iou_xyxy(box, nova) == pytest.approx(tau, abs=1e-9)
    assert _dentro_da_imagem(nova, LARGURA)


@pytest.mark.parametrize("modo", MODOS)
def test_caixas_aleatorias(modo):
    rng = random.Random(0)
    for _ in range(2000):
        w = rng.uniform(5, LARGURA)
        x1 = rng.uniform(0, LARGURA - w)
        y1 = rng.uniform(0, 500)
        box = (x1, y1, x1 + w, y1 + rng.uniform(5, 200))
        tau = rng.uniform(0.05, 0.99)
        nova, obtido = degradar(box, tau, modo, LARGURA)
        assert obtido == pytest.approx(tau, abs=1e-9)
        assert _dentro_da_imagem(nova, LARGURA)


def test_nivel_um_nao_altera():
    box = (10.0, 20.0, 110.0, 80.0)
    for modo in MODOS:
        assert degradar(box, 1.0, modo, LARGURA) == (box, 1.0)


def test_deterministico():
    box = (900.0, 100.0, 1020.0, 200.0)
    for modo in MODOS:
        assert degradar(box, 0.4, modo, LARGURA) == degradar(box, 0.4, modo, LARGURA)


def test_reducao_fica_dentro_da_original():
    box = (100.0, 100.0, 300.0, 200.0)
    nova, _ = degradar(box, 0.3, "reducao", LARGURA)
    assert box[0] <= nova[0] and nova[2] <= box[2] and box[1] <= nova[1] and nova[3] <= box[3]


def test_deslocamento_mantem_tamanho_quando_cabe():
    box = (400.0, 300.0, 500.0, 360.0)
    nova, _ = degradar(box, 0.5, "deslocamento", LARGURA)
    assert nova[2] - nova[0] == pytest.approx(100.0)
    assert (nova[1], nova[3]) == (box[1], box[3])


def test_modo_invalido():
    with pytest.raises(ValueError):
        degradar((0.0, 0.0, 10.0, 10.0), 0.5, "ampliacao", LARGURA)
