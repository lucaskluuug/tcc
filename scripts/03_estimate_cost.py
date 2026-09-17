#!/usr/bin/env python
"""Estima o tempo de treino de cada config no dispositivo atual (CPU ou GPU).

Uso:
    python scripts/03_estimate_cost.py --faster-rcnn configs/faster_rcnn/resnet50.yaml configs/faster_rcnn/resnet101.yaml
    python scripts/03_estimate_cost.py --yolo configs/yolov8/n.yaml configs/yolov8/s.yaml configs/yolov8/m.yaml
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from logoloc.config import load_dataclass_yaml, load_project_config
from logoloc.models.faster_rcnn import FasterRCNNConfig, build_faster_rcnn
from logoloc.models.yolo import YoloConfig, build_yolo, train_kwargs


def estimate_faster_rcnn(model_cfg_path: str, project_cfg, n_measure: int = 5) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = load_dataclass_yaml(FasterRCNNConfig, model_cfg_path)
    num_classes_with_bg = project_cfg.dataset.num_classes + 1

    model = build_faster_rcnn(num_classes_with_bg, model_cfg).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=model_cfg.lr, momentum=model_cfg.momentum)

    images = [torch.rand(3, model_cfg.min_size, model_cfg.min_size, device=device) for _ in range(model_cfg.batch_size)]
    targets = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 100.0, 100.0]], device=device),
            "labels": torch.tensor([1], dtype=torch.int64, device=device),
        }
        for _ in range(model_cfg.batch_size)
    ]

    model.train()
    for _ in range(2):
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    start = time.time()
    for _ in range(n_measure):
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    elapsed = time.time() - start
    time_per_batch = elapsed / n_measure

    n_train_images = project_cfg.dataset.images_per_class_train * project_cfg.dataset.num_classes
    if project_cfg.augmentation.enabled:
        n_train_images *= 1 + project_cfg.augmentation.copies_per_image
    n_batches_per_epoch = math.ceil(n_train_images / model_cfg.batch_size)
    epoch_time = n_batches_per_epoch * time_per_batch
    total_time = epoch_time * model_cfg.epochs

    print(f"\n[Faster R-CNN | {model_cfg.backbone}] dispositivo: {device}")
    print(f"  tempo/batch: {time_per_batch:.2f}s | batches/epoca: {n_batches_per_epoch} | tempo/epoca: {epoch_time / 60:.1f} min")
    print(f"  {model_cfg.epochs} epocas -> tempo total estimado: {total_time / 60:.1f} min ({total_time / 3600:.2f} h)")


def estimate_yolo(model_cfg_path: str, project_cfg, data_yaml: str, fraction: float = 0.15) -> None:
    model_cfg = load_dataclass_yaml(YoloConfig, model_cfg_path)
    model = build_yolo(model_cfg)

    marks: dict[str, list[float]] = {"train_start": [], "train_end": [], "fit_end": []}
    model.add_callback("on_train_epoch_start", lambda trainer: marks["train_start"].append(time.time()))
    model.add_callback("on_train_epoch_end", lambda trainer: marks["train_end"].append(time.time()))
    model.add_callback("on_fit_epoch_end", lambda trainer: marks["fit_end"].append(time.time()))

    kwargs = train_kwargs(
        model_cfg, data_yaml, project="outputs/_cost_estimate", name=f"yolov8{model_cfg.scale}", seed=project_cfg.seed
    )
    kwargs.update(epochs=2, fraction=fraction, verbose=False, plots=False)
    model.train(**kwargs)

    if len(marks["train_end"]) < 2 or len(marks["fit_end"]) < 2:
        print(f"\n[YOLOv8{model_cfg.scale}] nao consegui medir 2 epocas limpas (poucos batches na fracao?), pulei a estimativa.")
        return

    train_time_on_fraction = marks["train_end"][1] - marks["train_start"][1]
    val_time = marks["fit_end"][1] - marks["train_end"][1]

    train_time_full = train_time_on_fraction / fraction
    epoch_time_full = train_time_full + val_time
    total_time = epoch_time_full * model_cfg.epochs

    print(f"\n[YOLOv8{model_cfg.scale}] dispositivo: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(
        f"  2a epoca (steady-state, {fraction * 100:.0f}% do treino): "
        f"{train_time_on_fraction:.1f}s de treino + {val_time:.1f}s de validacao (P2 inteiro)"
    )
    print(f"  extrapolado p/ 100% do treino: {train_time_full / 60:.1f} min + {val_time / 60:.1f} min de val = {epoch_time_full / 60:.1f} min/epoca")
    print(f"  {model_cfg.epochs} epocas -> tempo total estimado: {total_time / 60:.1f} min ({total_time / 3600:.2f} h)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--faster-rcnn", nargs="*", default=[], help="paths de configs/faster_rcnn/*.yaml a estimar")
    parser.add_argument("--yolo", nargs="*", default=[], help="paths de configs/yolov8/*.yaml a estimar")
    parser.add_argument("--yolo-data-yaml", default="data/processed/yolo/data.yaml")
    args = parser.parse_args()

    project_cfg = load_project_config(args.base_config)

    for path in args.faster_rcnn:
        estimate_faster_rcnn(path, project_cfg)

    for path in args.yolo:
        if not Path(args.yolo_data_yaml).exists():
            print(f"\n[YOLO] pulei {path}: rode scripts/01 e scripts/02 primeiro para gerar {args.yolo_data_yaml}")
            continue
        estimate_yolo(path, project_cfg, args.yolo_data_yaml)


if __name__ == "__main__":
    main()
