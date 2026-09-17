import random

import pytest

from logoloc.eval.jitter import iou_xyxy, jitter_to_iou


def test_iou_identico():
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_sem_sobreposicao():
    assert iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_meia_sobreposicao():
    assert iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


@pytest.mark.parametrize("alvo", [0.9, 0.8, 0.7, 0.6, 0.5, 0.3])
def test_jitter_atinge_iou_alvo(alvo):
    rng = random.Random(0)
    box = (100.0, 100.0, 300.0, 250.0)
    for _ in range(20):
        novo, obtido = jitter_to_iou(box, alvo, width=640, height=480, rng=rng)
        assert abs(obtido - alvo) <= 0.05
        assert iou_xyxy(box, novo) == pytest.approx(obtido, abs=1e-6)


def test_jitter_iou_1_devolve_caixa_original():
    rng = random.Random(0)
    box = (10.0, 10.0, 50.0, 50.0)
    novo, obtido = jitter_to_iou(box, 1.0, width=100, height=100, rng=rng)
    assert novo == box
    assert obtido == 1.0


def test_jitter_respeita_limites_da_imagem():
    rng = random.Random(3)
    box = (5.0, 5.0, 40.0, 40.0)
    for _ in range(50):
        (x1, y1, x2, y2), _ = jitter_to_iou(box, 0.5, width=64, height=64, rng=rng)
        assert 0 <= x1 < x2 <= 64
        assert 0 <= y1 < y2 <= 64


def test_jitter_determinístico_com_mesma_semente():
    a = jitter_to_iou((0.0, 0.0, 20.0, 20.0), 0.6, 100, 100, random.Random(7))
    b = jitter_to_iou((0.0, 0.0, 20.0, 20.0), 0.6, 100, 100, random.Random(7))
    assert a == b
