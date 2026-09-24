from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from ..config import ProjectConfig, load_dataclass_yaml, load_project_config
from ..data.augment import generate_augmented_p1
from ..data.convert_yolo import convert_all_to_yolo
from ..data.label_map import load_label_map
from ..data.splits import prepare_dataset
from ..models.yolo import YoloConfig, build_yolo, train_kwargs
from ..seed import set_global_seed
from ..utils.run_log import RunLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def prepare_yolo_data(project_cfg: ProjectConfig, force_prepare: bool = False, force_convert: bool = False):
    records_by_split = prepare_dataset(project_cfg, force=force_prepare)
    label_map = load_label_map(project_cfg.resolved_path(project_cfg.paths.processed_dir) / "label_map.json")

    if project_cfg.augmentation.enabled:
        records_by_split["P1"] = generate_augmented_p1(
            records_by_split["P1"],
            project_cfg.augmentation,
            output_dir=project_cfg.resolved_path(project_cfg.paths.processed_dir) / "p1_augmented",
            seed=project_cfg.seed,
        )

    yolo_dir = project_cfg.resolved_path(project_cfg.paths.processed_dir) / "yolo"
    data_yaml_path = convert_all_to_yolo(records_by_split, label_map, yolo_dir, force=force_convert)
    return data_yaml_path, label_map


def _backup_checkpoint_callback(run_name: str, backup_dir: str):
    def _on_model_save(trainer) -> None:
        weights_dir = Path(trainer.save_dir) / "weights"
        backup_weights_dir = Path(backup_dir) / run_name / "weights"
        backup_weights_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("last.pt", "best.pt"):
            src = weights_dir / fname
            if src.exists():
                shutil.copy2(src, backup_weights_dir / fname)

    return _on_model_save


def train_yolo(
    project_cfg: ProjectConfig,
    model_cfg: YoloConfig,
    run_name: str,
    checkpoint_backup_dir: str | None = None,
    resume_from: str | None = None,
) -> str:
    set_global_seed(project_cfg.seed)
    data_yaml_path, _label_map = prepare_yolo_data(project_cfg)

    outputs_dir = project_cfg.resolved_path(project_cfg.paths.outputs_dir) / "runs"
    run_logger = RunLogger(outputs_dir / run_name, run_name, model_cfg)

    resuming = resume_from is not None and Path(resume_from).exists()
    if resuming:
        from ultralytics import YOLO

        model = YOLO(resume_from)
        logger.info("Retomando '%s' a partir do checkpoint: %s", run_name, resume_from)
    else:
        model = build_yolo(model_cfg)

    if checkpoint_backup_dir is not None:
        model.add_callback("on_model_save", _backup_checkpoint_callback(run_name, checkpoint_backup_dir))

    kwargs = train_kwargs(model_cfg, str(data_yaml_path), project=str(outputs_dir), name=run_name, seed=project_cfg.seed)
    if resuming:
        kwargs["resume"] = True
    logger.info("Iniciando treino YOLOv8%s com kwargs: %s", model_cfg.scale, kwargs)
    results = model.train(**kwargs)

    run_logger.save(extra={"save_dir": str(results.save_dir) if hasattr(results, "save_dir") else None})
    if checkpoint_backup_dir is not None:
        shutil.copy2(
            outputs_dir / run_name / "run_config.json", Path(checkpoint_backup_dir) / run_name / "run_config.json"
        )
    logger.info("Treino concluido. Resultados em %s", outputs_dir / run_name)
    return str(outputs_dir / run_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--model-config", default="configs/yolov8/s.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--checkpoint-backup-dir",
        default=None,
        help="copia last.pt/best.pt pra essa pasta a cada epoca (ex.: uma pasta na Google Drive montada no Colab)",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="caminho de um last.pt de uma epoca anterior deste run (ex.: o backup na Drive) pra continuar de onde parou",
    )
    args = parser.parse_args()

    project_cfg = load_project_config(args.base_config)
    model_cfg = load_dataclass_yaml(YoloConfig, args.model_config)
    train_yolo(
        project_cfg,
        model_cfg,
        args.run_name,
        checkpoint_backup_dir=args.checkpoint_backup_dir,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
