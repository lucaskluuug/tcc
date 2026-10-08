from __future__ import annotations

import math

Box = tuple[float, float, float, float]

MODOS = ("reducao", "deslocamento", "ampliacao")
NIVEIS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)

DIRECOES = {
    "direita": (1, 0),
    "esquerda": (-1, 0),
    "baixo": (0, 1),
    "cima": (0, -1),
    "baixo_direita": (1, 1),
    "cima_direita": (1, -1),
    "baixo_esquerda": (-1, 1),
    "cima_esquerda": (-1, -1),
}
TIPO_DIRECAO = {
    "direita": "horizontal",
    "esquerda": "horizontal",
    "baixo": "vertical",
    "cima": "vertical",
    "baixo_direita": "diagonal",
    "cima_direita": "diagonal",
    "baixo_esquerda": "diagonal",
    "cima_esquerda": "diagonal",
    "borda": "borda",
    "": "",
}


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


def transladar(box: Box, tau: float, sentido_x: int, sentido_y: int) -> Box:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if sentido_x and sentido_y:
        fracao = 1 - math.sqrt(2 * tau / (1 + tau))
    else:
        fracao = (1 - tau) / (1 + tau)
    dx, dy = sentido_x * fracao * w, sentido_y * fracao * h
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def cabe(box: Box, largura_imagem: float, altura_imagem: float) -> bool:
    return box[0] >= 0 and box[1] >= 0 and box[2] <= largura_imagem and box[3] <= altura_imagem


def deslocar_cortando_na_borda(box: Box, tau: float, largura_imagem: float) -> Box:
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


def deslocamentos(box: Box, tau: float, largura_imagem: float, altura_imagem: float) -> list[tuple[str, Box]]:
    W, H = max(largura_imagem, box[2]), max(altura_imagem, box[3])
    validas = []
    for nome, (sx, sy) in DIRECOES.items():
        nova = transladar(box, tau, sx, sy)
        if cabe(nova, W, H):
            validas.append((nome, nova))
    if validas:
        return validas
    return [("borda", deslocar_cortando_na_borda(box, tau, largura_imagem))]


def ampliar(box: Box, tau: float, largura_imagem: float, altura_imagem: float) -> Box:
    x1, y1, x2, y2 = box
    W, H = max(largura_imagem, x2), max(altura_imagem, y2)
    s = 1 / math.sqrt(tau)
    nova_w, nova_h = min((x2 - x1) * s, W), min((y2 - y1) * s, H)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nx1 = min(max(cx - nova_w / 2, 0.0), W - nova_w)
    ny1 = min(max(cy - nova_h / 2, 0.0), H - nova_h)
    return (nx1, ny1, nx1 + nova_w, ny1 + nova_h)


def degradar(
    box: Box, tau: float, modo: str, largura_imagem: float, altura_imagem: float
) -> list[tuple[str, Box, float]]:
    if tau >= 1.0:
        return [("", box, 1.0)]
    if modo == "reducao":
        candidatas = [("", reduzir(box, tau))]
    elif modo == "deslocamento":
        candidatas = deslocamentos(box, tau, largura_imagem, altura_imagem)
    elif modo == "ampliacao":
        candidatas = [("", ampliar(box, tau, largura_imagem, altura_imagem))]
    else:
        raise ValueError(f"modo desconhecido: {modo}")
    return [(direcao, nova, iou_xyxy(box, nova)) for direcao, nova in candidatas]
