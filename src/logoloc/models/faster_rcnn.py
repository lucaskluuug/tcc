from __future__ import annotations

import dataclasses

import torch.nn as nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.rpn import AnchorGenerator


@dataclasses.dataclass
class FasterRCNNConfig:
    backbone: str = "resnet50"
    trainable_backbone_layers: int = 3
    anchor_sizes: tuple = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios: tuple = ((0.5, 1.0, 2.0),) * 5
    min_size: int = 800
    max_size: int = 1333

    rpn_fg_iou_thresh: float = 0.7
    rpn_bg_iou_thresh: float = 0.3
    rpn_pre_nms_top_n_train: int = 2000
    rpn_pre_nms_top_n_test: int = 1000
    rpn_post_nms_top_n_train: int = 2000
    rpn_post_nms_top_n_test: int = 1000
    rpn_nms_thresh: float = 0.7

    box_score_thresh: float = 0.05
    box_nms_thresh: float = 0.5
    box_detections_per_img: int = 100
    box_fg_iou_thresh: float = 0.5
    box_bg_iou_thresh: float = 0.5

    epochs: int = 20
    batch_size: int = 4
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    lr_step_size: int = 8
    lr_gamma: float = 0.1
    pretrained_backbone: bool = True


def _build_backbone(cfg: FasterRCNNConfig) -> nn.Module:
    try:
        return resnet_fpn_backbone(
            backbone_name=cfg.backbone,
            weights="DEFAULT" if cfg.pretrained_backbone else None,
            trainable_layers=cfg.trainable_backbone_layers,
        )
    except TypeError:
        return resnet_fpn_backbone(
            backbone_name=cfg.backbone,
            pretrained=cfg.pretrained_backbone,
            trainable_layers=cfg.trainable_backbone_layers,
        )


def build_faster_rcnn(num_classes_with_background: int, cfg: FasterRCNNConfig) -> FasterRCNN:
    backbone = _build_backbone(cfg)
    anchor_generator = AnchorGenerator(sizes=cfg.anchor_sizes, aspect_ratios=cfg.aspect_ratios)
    model = FasterRCNN(
        backbone,
        num_classes=num_classes_with_background,
        min_size=cfg.min_size,
        max_size=cfg.max_size,
        rpn_anchor_generator=anchor_generator,
        rpn_fg_iou_thresh=cfg.rpn_fg_iou_thresh,
        rpn_bg_iou_thresh=cfg.rpn_bg_iou_thresh,
        rpn_pre_nms_top_n_train=cfg.rpn_pre_nms_top_n_train,
        rpn_pre_nms_top_n_test=cfg.rpn_pre_nms_top_n_test,
        rpn_post_nms_top_n_train=cfg.rpn_post_nms_top_n_train,
        rpn_post_nms_top_n_test=cfg.rpn_post_nms_top_n_test,
        rpn_nms_thresh=cfg.rpn_nms_thresh,
        box_score_thresh=cfg.box_score_thresh,
        box_nms_thresh=cfg.box_nms_thresh,
        box_detections_per_img=cfg.box_detections_per_img,
        box_fg_iou_thresh=cfg.box_fg_iou_thresh,
        box_bg_iou_thresh=cfg.box_bg_iou_thresh,
    )
    return model
