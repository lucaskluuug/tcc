from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from .records import BBox, ImageRecord, LogoInstance

logger = logging.getLogger(__name__)

NO_LOGO_DIRNAME = "no-logo"
SPLIT_FILE_HINTS = {
    "P1": ("trainset", "train-set", "train_set"),
    "P2": ("valset", "val-set", "val_set", "validationset"),
    "P3": ("testset", "test-set", "test_set"),
}


def _is_junk(path: Path) -> bool:
    return "__MACOSX" in path.parts or path.name.startswith("._")


def _find_split_file(raw_dir: Path, split: str) -> Path:
    hints = SPLIT_FILE_HINTS[split]
    candidates = sorted(p for p in raw_dir.rglob("*.txt") if not _is_junk(p))
    matches = [c for c in candidates if c.stem.split(".")[0].lower() in hints]
    if not matches:
        raise FileNotFoundError(
            f"Nao encontrei o arquivo de split para {split} dentro de {raw_dir}. "
            f"Procurei por *.txt cujo nome comece com algum de {hints}. "
            "Rode scripts/00_inspect_raw.py para ver a arvore real e ajuste "
            "SPLIT_FILE_HINTS ou _find_split_file em raw_adapter.py."
        )
    preferred = [c for c in matches if "relpath" in c.stem.lower()]
    return preferred[0] if preferred else matches[0]


def _read_relpaths(split_file: Path) -> list[str]:
    with open(split_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines()]
    return [ln for ln in lines if ln]


def _find_dir_case_insensitive(parent: Path, target_name: str) -> Path | None:
    if not parent.is_dir():
        return None
    target_lower = target_name.lower()
    for entry in parent.iterdir():
        if entry.is_dir() and entry.name.lower() == target_lower:
            return entry
    return None


def _bboxes_file_for_image(image_path: Path) -> Path | None:
    parts = list(image_path.parts)
    try:
        jpg_idx = parts.index("jpg")
    except ValueError:
        return None
    mask_parts = parts.copy()
    mask_parts[jpg_idx] = "masks"
    candidate = Path(*mask_parts).with_name(image_path.name + ".bboxes.txt")
    if candidate.exists():
        return candidate

    alt_class_dir = _find_dir_case_insensitive(candidate.parent.parent, candidate.parent.name)
    if alt_class_dir is not None:
        alt_candidate = alt_class_dir / candidate.name
        if alt_candidate.exists():
            return alt_candidate
    return None


def _parse_bboxes_file(bboxes_path: Path, class_name: str) -> list[LogoInstance]:
    with open(bboxes_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    if not lines:
        return []
    body = lines[1:] if not lines[0][0].isdigit() and not lines[0][0] == "-" else lines
    instances = []
    for line in body:
        parts = line.replace(",", " ").split()
        if len(parts) < 4:
            continue
        x, y, w, h = (float(v) for v in parts[:4])
        instances.append(LogoInstance(class_name=class_name, bbox=BBox(x, y, w, h)))
    return instances


def _class_name_from_image_path(raw_dir: Path, image_path: Path) -> str:
    rel = image_path.relative_to(raw_dir)
    parts = rel.parts
    try:
        jpg_idx = parts.index("jpg")
    except ValueError:
        return NO_LOGO_DIRNAME
    if jpg_idx + 1 >= len(parts):
        return NO_LOGO_DIRNAME
    class_dir = parts[jpg_idx + 1]
    return NO_LOGO_DIRNAME if class_dir.lower() == NO_LOGO_DIRNAME else class_dir


def _image_id(raw_dir: Path, image_path: Path) -> str:
    rel = image_path.relative_to(raw_dir).with_suffix("")
    return str(rel).replace("\\", "/")


def load_split(raw_dir: str | Path, split: str) -> list[ImageRecord]:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"raw_dir nao existe: {raw_dir}. Baixe o FlickrLogos-32 (ver README) "
            "e extraia dentro dessa pasta antes de rodar o pipeline."
        )
    raw_dir = raw_dir.resolve()
    split_file = _find_split_file(raw_dir, split)
    relpaths = _read_relpaths(split_file)

    records: list[ImageRecord] = []
    for relpath in relpaths:
        image_path = (raw_dir / relpath).resolve()
        if not image_path.exists():
            logger.warning("Imagem listada em %s nao encontrada: %s", split_file.name, image_path)
            continue
        class_name = _class_name_from_image_path(raw_dir, image_path)
        instances: list[LogoInstance] = []
        if class_name != NO_LOGO_DIRNAME:
            bboxes_path = _bboxes_file_for_image(image_path)
            if bboxes_path is not None:
                instances = _parse_bboxes_file(bboxes_path, class_name)
            else:
                logger.warning("Sem arquivo de bbox para imagem com classe: %s", image_path)
        with Image.open(image_path) as im:
            width, height = im.size
        records.append(
            ImageRecord(
                image_path=image_path,
                split=split,
                width=width,
                height=height,
                instances=instances,
                image_id=_image_id(raw_dir, image_path),
            )
        )
    return records


def load_all_splits(raw_dir: str | Path) -> dict[str, list[ImageRecord]]:
    return {split: load_split(raw_dir, split) for split in ("P1", "P2", "P3")}


def discover_class_names(raw_dir: str | Path) -> list[str]:
    raw_dir = Path(raw_dir)
    jpg_roots = [p for p in raw_dir.rglob("jpg") if p.is_dir() and not _is_junk(p)]
    if not jpg_roots:
        raise FileNotFoundError(f"Nao encontrei pasta 'jpg' dentro de {raw_dir}")
    jpg_root = max(jpg_roots, key=lambda p: sum(1 for _ in p.iterdir()))
    class_dirs = sorted(
        p.name for p in jpg_root.iterdir() if p.is_dir() and p.name.lower() != NO_LOGO_DIRNAME
    )
    return class_dirs
