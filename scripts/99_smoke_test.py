#!/usr/bin/env python
"""Smoke test end-to-end com dados sinteticos (nao precisa do FlickrLogos-32
baixado): exercita dataset, crops, augmentation, conversao YOLO, os 2
modelos e a avaliacao Fluxo 1/2.

Uso:
    python scripts/99_smoke_test.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image

from logoloc.config import CropConfig, AugmentationConfig
from logoloc.data.augment import generate_augmented_p1
from logoloc.data.convert_torchvision import FlickrLogosDetectionDataset, collate_fn, num_classes_with_background
from logoloc.data.convert_yolo import convert_all_to_yolo
from logoloc.data.crops import generate_gt_crops, tight_variant
from logoloc.data.label_map import build_label_map
from logoloc.data.records import BBox, ImageRecord, LogoInstance
from logoloc.eval.consolidate import build_comparison_table, evaluate_model, plot_iou_vs_classification, save_evaluation
from logoloc.eval.predictors import faster_rcnn_predictor, yolo_predictor
from logoloc.models.faster_rcnn import FasterRCNNConfig, build_faster_rcnn
from logoloc.models.yolo import YoloConfig, build_yolo, predict_kwargs, train_kwargs


CLASS_NAMES = ["fakebrand_a", "fakebrand_b", "fakebrand_c"]


def make_synthetic_records(tmp_dir: Path, split: str, n_per_class: int = 4) -> list[ImageRecord]:
    img_dir = tmp_dir / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for class_name in CLASS_NAMES:
        for i in range(n_per_class):
            w, h = 320, 240
            arr = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
            x, y, bw, bh = 50, 40, 100, 80
            arr[y : y + bh, x : x + bw] = [200, 30, 30]
            path = img_dir / f"{class_name}_{i}.jpg"
            Image.fromarray(arr).save(path, quality=90)
            records.append(
                ImageRecord(
                    image_path=path,
                    split=split,
                    width=w,
                    height=h,
                    instances=[LogoInstance(class_name=class_name, bbox=BBox(x, y, bw, bh))],
                    image_id=f"{split}/{class_name}_{i}",
                )
            )
    return records


def main() -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="logoloc_smoke_"))
    print(f"Diretorio temporario: {tmp_dir}")
    try:
        _run(tmp_dir)
        print("\n=== SMOKE TEST OK: pipeline roda de ponta a ponta ===")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _run(tmp_dir: Path) -> None:
    label_map = build_label_map(CLASS_NAMES)
    id_to_name = {v: k for k, v in label_map.items()}

    p1 = make_synthetic_records(tmp_dir, "P1", n_per_class=3)
    p3 = make_synthetic_records(tmp_dir, "P3", n_per_class=2)
    print(f"[ok] records sinteticos: P1={len(p1)} P3={len(p3)}")

    aug_cfg = AugmentationConfig(copies_per_image=2)
    p1_aug = generate_augmented_p1(p1, aug_cfg, output_dir=tmp_dir / "processed" / "p1_augmented", seed=0)
    assert len(p1_aug) > len(p1)
    print(f"[ok] augmentation: P1 {len(p1)} -> {len(p1_aug)}")

    crop_cfg = CropConfig()
    crops_p3 = generate_gt_crops(p3, crop_cfg, output_dir=tmp_dir / "processed" / "p3_gt_crops")
    crops_p3_tight = generate_gt_crops(p3, tight_variant(crop_cfg), output_dir=tmp_dir / "processed" / "p3_gt_crops_tight")
    assert len(crops_p3) == len(p3)
    print(f"[ok] crops Fluxo 1: {len(crops_p3)} (+ {len(crops_p3_tight)} variante sem padding)")

    import torch

    ds = FlickrLogosDetectionDataset(p1_aug, label_map)
    image_tensor, target = ds[0]
    assert image_tensor.ndim == 3
    assert target["boxes"].shape[1] == 4
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    images, targets = next(iter(loader))
    print(f"[ok] dataset torchvision: batch de {len(images)} imagens, target keys={list(targets[0].keys())}")

    frcnn_cfg = FasterRCNNConfig(min_size=128, max_size=128, epochs=1, batch_size=2)
    model = build_faster_rcnn(num_classes_with_background(label_map), frcnn_cfg)
    model.train()
    clean_targets = [{k: v for k, v in t.items() if k != "image_id_str"} for t in targets]
    loss_dict = model(list(images), clean_targets)
    total_loss = sum(loss_dict.values())
    total_loss.backward()
    print(f"[ok] Faster R-CNN forward+backward: loss={float(total_loss):.4f} keys={list(loss_dict.keys())}")

    model.eval()
    device = torch.device("cpu")
    predict_fn = faster_rcnn_predictor(model, device, id_to_name)
    evaluation = evaluate_model(
        "smoke_faster_rcnn", predict_fn, crops_p3, p3, CLASS_NAMES, (0.5,),
        primary_iou_threshold=0.5, background_iou_floor=0.1, crops_p3_tight=crops_p3_tight,
    )
    save_evaluation(evaluation, tmp_dir / "results" / "smoke_faster_rcnn")
    print(
        f"[ok] Faster R-CNN eval: flow1 acc={evaluation.flow1.report.accuracy:.3f} "
        f"(sem padding: {evaluation.flow1_tight.report.accuracy:.3f}) "
        f"flow2 mAP@0.5={evaluation.flow2.results_by_iou[0.5].mean_ap:.3f} "
        f"erro_decomp={evaluation.flow2.error_decomposition.fractions()}"
    )

    yolo_dir = tmp_dir / "processed" / "yolo"
    data_yaml = convert_all_to_yolo({"P1": p1_aug, "P2": p3, "P3": p3}, label_map, yolo_dir)
    assert data_yaml.exists()
    n_label_files = len(list((yolo_dir / "labels" / "train").glob("*.txt")))
    assert n_label_files == len(p1_aug)
    print(f"[ok] conversao YOLO: {n_label_files} arquivos de label em {yolo_dir}")

    yolo_cfg = YoloConfig(scale="n", imgsz=64, epochs=1, batch=2)
    yolo_model = build_yolo(yolo_cfg)
    kwargs = train_kwargs(yolo_cfg, str(data_yaml), project=str(tmp_dir / "runs"), name="smoke_yolo", seed=0)
    kwargs["verbose"] = False
    yolo_model.train(**kwargs)
    print("[ok] YOLOv8n treinou 1 epoca no dataset sintetico sem erro")

    predict_fn_yolo = yolo_predictor(yolo_model, predict_kwargs(yolo_cfg))
    evaluation_yolo = evaluate_model(
        "smoke_yolo", predict_fn_yolo, crops_p3, p3, CLASS_NAMES, (0.5,),
        primary_iou_threshold=0.5, background_iou_floor=0.1, crops_p3_tight=crops_p3_tight,
    )
    save_evaluation(evaluation_yolo, tmp_dir / "results" / "smoke_yolo")
    print(
        f"[ok] YOLO eval: flow1 acc={evaluation_yolo.flow1.report.accuracy:.3f} "
        f"flow2 mAP@0.5={evaluation_yolo.flow2.results_by_iou[0.5].mean_ap:.3f} "
        f"erro_decomp={evaluation_yolo.flow2.error_decomposition.fractions()}"
    )

    comparison = build_comparison_table([evaluation, evaluation_yolo], primary_iou=0.5)
    print("[ok] tabela comparativa:\n" + comparison.to_string(index=False))
    plot_iou_vs_classification([evaluation, evaluation_yolo], tmp_dir / "results" / "iou_vs_classification.png")
    print("[ok] grafico IoU x classificacao (com linha de teto do Fluxo 1) gerado sem erro")


if __name__ == "__main__":
    main()
