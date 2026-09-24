from __future__ import annotations

import dataclasses

VALID_SCALES = ("n", "s", "m", "l", "x")


@dataclasses.dataclass
class YoloConfig:
    scale: str = "s"
    pretrained: bool = True

    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.5
    max_det: int = 300

    epochs: int = 100
    batch: int = 16
    optimizer: str = "auto"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    freeze: int | None = None

    def __post_init__(self) -> None:
        if self.scale not in VALID_SCALES:
            raise ValueError(f"scale deve ser um de {VALID_SCALES}, recebido '{self.scale}'")


def weights_name(cfg: YoloConfig) -> str:
    return f"yolov8{cfg.scale}.pt" if cfg.pretrained else f"yolov8{cfg.scale}.yaml"


def build_yolo(cfg: YoloConfig):
    from ultralytics import YOLO

    return YOLO(weights_name(cfg))


def train_kwargs(cfg: YoloConfig, data_yaml: str, project: str, name: str, seed: int) -> dict:
    return dict(
        data=data_yaml,
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        optimizer=cfg.optimizer,
        lr0=cfg.lr0,
        lrf=cfg.lrf,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs,
        box=cfg.box,
        cls=cfg.cls,
        dfl=cfg.dfl,
        freeze=cfg.freeze,
        seed=seed,
        project=project,
        name=name,
        deterministic=True,
    )


def predict_kwargs(cfg: YoloConfig) -> dict:
    return dict(imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou, max_det=cfg.max_det)
