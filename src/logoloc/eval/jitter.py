from __future__ import annotations

import math
import random

Box = tuple[float, float, float, float]


def iou_xyxy(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _perturb(box: Box, direction: tuple[float, float, float, float], t: float, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    dx, dy, dlw, dlh = direction

    cx += t * dx * w
    cy += t * dy * h
    w *= math.exp(t * dlw)
    h *= math.exp(t * dlh)

    nx1 = max(0.0, min(width - 1.0, cx - w / 2))
    ny1 = max(0.0, min(height - 1.0, cy - h / 2))
    nx2 = max(nx1 + 1.0, min(float(width), cx + w / 2))
    ny2 = max(ny1 + 1.0, min(float(height), cy + h / 2))
    return (nx1, ny1, nx2, ny2)


def _random_direction(rng: random.Random) -> tuple[float, float, float, float]:
    while True:
        d = tuple(rng.gauss(0.0, 1.0) for _ in range(4))
        norm = math.sqrt(sum(v * v for v in d))
        if norm > 1e-6:
            return tuple(v / norm for v in d)


def jitter_to_iou(
    box: Box,
    target_iou: float,
    width: int,
    height: int,
    rng: random.Random,
    tolerance: float = 0.02,
    max_iter: int = 40,
) -> tuple[Box, float]:
    if target_iou >= 1.0:
        return box, 1.0

    direction = _random_direction(rng)

    lo, hi = 0.0, 0.25
    for _ in range(20):
        if iou_xyxy(box, _perturb(box, direction, hi, width, height)) <= target_iou:
            break
        lo = hi
        hi *= 2
        if hi > 64:
            break

    best_box, best_iou = _perturb(box, direction, hi, width, height), None
    best_iou = iou_xyxy(box, best_box)
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        candidate = _perturb(box, direction, mid, width, height)
        achieved = iou_xyxy(box, candidate)
        if abs(achieved - best_iou) > 0 and abs(achieved - target_iou) < abs(best_iou - target_iou):
            best_box, best_iou = candidate, achieved
        if abs(achieved - target_iou) <= tolerance:
            return candidate, achieved
        if achieved > target_iou:
            lo = mid
        else:
            hi = mid
    return best_box, best_iou
