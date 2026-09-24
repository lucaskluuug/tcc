from __future__ import annotations

import argparse
import csv
import logging
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import ProjectConfig, load_dataclass_yaml, load_project_config
from ..data.augment import generate_augmented_p1
from ..data.convert_torchvision import FlickrLogosDetectionDataset, collate_fn, num_classes_with_background
from ..data.label_map import load_label_map
from ..data.splits import prepare_dataset
from ..models.faster_rcnn import FasterRCNNConfig, build_faster_rcnn
from ..seed import set_global_seed
from ..utils.run_log import RunLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_datasets(project_cfg: ProjectConfig, force_prepare: bool = False):
    records_by_split = prepare_dataset(project_cfg, force=force_prepare)
    label_map = load_label_map(project_cfg.resolved_path(project_cfg.paths.processed_dir) / "label_map.json")

    p1 = records_by_split["P1"]
    if project_cfg.augmentation.enabled:
        p1 = generate_augmented_p1(
            p1,
            project_cfg.augmentation,
            output_dir=project_cfg.resolved_path(project_cfg.paths.processed_dir) / "p1_augmented",
            seed=project_cfg.seed,
        )

    train_ds = FlickrLogosDetectionDataset(p1, label_map)
    val_ds = FlickrLogosDetectionDataset(records_by_split["P2"], label_map)
    return train_ds, val_ds, label_map


def train_faster_rcnn(
    project_cfg: ProjectConfig,
    model_cfg: FasterRCNNConfig,
    run_name: str,
    max_batches_per_epoch: int | None = None,
    checkpoint_backup_dir: str | None = None,
    resume_from: str | None = None,
) -> Path:
    set_global_seed(project_cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Dispositivo de treino: %s", device)

    train_ds, val_ds, label_map = build_datasets(project_cfg)
    num_workers = 2 if device.type == "cuda" else 0
    train_loader = DataLoader(
        train_ds,
        batch_size=model_cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    model = build_faster_rcnn(num_classes_with_background(label_map), model_cfg).to(device)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=model_cfg.lr,
        momentum=model_cfg.momentum,
        weight_decay=model_cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=model_cfg.lr_step_size, gamma=model_cfg.lr_gamma)

    run_dir = project_cfg.resolved_path(project_cfg.paths.outputs_dir) / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_logger = RunLogger(run_dir, run_name, model_cfg)
    loss_log_path = run_dir / "train_loss_by_epoch.csv"

    start_epoch = 0
    if resume_from is not None and Path(resume_from).exists():
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_epoch = checkpoint["epoch"]
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            for _ in range(start_epoch):
                scheduler.step()
        logger.info("Retomando '%s' a partir da epoca %d/%d (checkpoint: %s)", run_name, start_epoch, model_cfg.epochs, resume_from)

        resume_loss_csv = Path(resume_from).parent / "train_loss_by_epoch.csv"
        if resume_loss_csv.exists() and resume_loss_csv.resolve() != loss_log_path.resolve():
            shutil.copy2(resume_loss_csv, loss_log_path)

    if not loss_log_path.exists():
        with open(loss_log_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch", "mean_loss", "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"])

    for epoch in range(start_epoch, model_cfg.epochs):
        model.train()
        epoch_losses = {"total": 0.0, "loss_classifier": 0.0, "loss_box_reg": 0.0, "loss_objectness": 0.0, "loss_rpn_box_reg": 0.0}
        n_batches = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                break
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items() if k != "image_id_str"} for t in targets]

            loss_dict = model(images, targets)
            total_loss = sum(loss_dict.values())

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            n_batches += 1
            epoch_losses["total"] += float(total_loss.item())
            for k in ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"):
                if k in loss_dict:
                    epoch_losses[k] += float(loss_dict[k].item())

        scheduler.step()
        mean_loss = epoch_losses["total"] / max(n_batches, 1)
        logger.info("Epoca %d/%d, loss medio: %.4f", epoch + 1, model_cfg.epochs, mean_loss)
        with open(loss_log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    epoch + 1,
                    mean_loss,
                    epoch_losses["loss_classifier"] / max(n_batches, 1),
                    epoch_losses["loss_box_reg"] / max(n_batches, 1),
                    epoch_losses["loss_objectness"] / max(n_batches, 1),
                    epoch_losses["loss_rpn_box_reg"] / max(n_batches, 1),
                ]
            )

        checkpoint_path = run_dir / "model_last.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch + 1,
                "label_map": label_map,
            },
            checkpoint_path,
        )

        if checkpoint_backup_dir is not None:
            backup_run_dir = Path(checkpoint_backup_dir) / run_name
            backup_run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint_path, backup_run_dir / "model_last.pt")
            shutil.copy2(loss_log_path, backup_run_dir / "train_loss_by_epoch.csv")
            logger.info("Checkpoint da epoca %d copiado para %s", epoch + 1, backup_run_dir)

    run_logger.save(
        extra={
            "epochs_ran": model_cfg.epochs - start_epoch,
            "start_epoch": start_epoch,
            "total_epochs": model_cfg.epochs,
            "label_map_size": len(label_map),
        }
    )
    if checkpoint_backup_dir is not None:
        shutil.copy2(run_dir / "run_config.json", Path(checkpoint_backup_dir) / run_name / "run_config.json")
    logger.info("Treino concluido. Checkpoint final em %s", run_dir / "model_last.pt")
    return run_dir / "model_last.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--model-config", default="configs/faster_rcnn/resnet50.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--max-batches-per-epoch", type=int, default=None, help="so' para smoke test em CPU")
    parser.add_argument(
        "--checkpoint-backup-dir",
        default=None,
        help="copia o checkpoint pra essa pasta a cada epoca (ex.: uma pasta na Google Drive montada no Colab)",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="caminho de um model_last.pt de uma epoca anterior deste run (ex.: o backup na Drive) pra continuar de onde parou",
    )
    args = parser.parse_args()

    project_cfg = load_project_config(args.base_config)
    model_cfg = load_dataclass_yaml(FasterRCNNConfig, args.model_config)
    train_faster_rcnn(
        project_cfg,
        model_cfg,
        args.run_name,
        max_batches_per_epoch=args.max_batches_per_epoch,
        checkpoint_backup_dir=args.checkpoint_backup_dir,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
