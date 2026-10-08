from __future__ import annotations

import math

Box = tuple[float, float, float, float]

MODOS = ("reducao", "deslocamento")
NIVEIS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)


def iou_xyxy(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    uniao = area_a + area_b - inter
    return inter / uniao if uniao > 0 else 0.0


def reduzir(box: Box, tau: float) -> Box:
    x1, y1, x2, y2 = box
    s = math.sqrt(tau)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    meia_w, meia_h = (x2 - x1) * s / 2, (y2 - y1) * s / 2
    return (cx - meia_w, cy - meia_h, cx + meia_w, cy + meia_h)


def deslocar(box: Box, tau: float, largura_imagem: float) -> Box:
    x1, y1, x2, y2 = box
    w = x2 - x1
    W = max(largura_imagem, x2)
    dx = w * (1 - tau) / (1 + tau)
    espaco_direita, espaco_esquerda = W - x2, x1

    if espaco_direita >= espaco_esquerda:
        if espaco_direita >= dx:
            return (x1 + dx, y1, x2 + dx, y2)
        d = w - tau * (W - x1)
        return (x1 + d, y1, W, y2)

    if espaco_esquerda >= dx:
        return (x1 - dx, y1, x2 - dx, y2)
    d = w - tau * x2
    return (0.0, y1, x2 - d, y2)


def degradar(box: Box, tau: float, modo: str, largura_imagem: float) -> tuple[Box, float]:
    if tau >= 1.0:
        return box, 1.0
    if modo == "reducao":
        nova = reduzir(box, tau)
    elif modo == "deslocamento":
        nova = deslocar(box, tau, largura_imagem)
    else:
        raise ValueError(f"modo desconhecido: {modo}")
    return nova, iou_xyxy(box, nova)
