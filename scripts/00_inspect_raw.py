#!/usr/bin/env python
"""Imprime a arvore de diretorios do FlickrLogos-32 baixado.

Uso:
    python scripts/00_inspect_raw.py data/raw/FlickrLogos-32
"""
from __future__ import annotations

import sys
from pathlib import Path


def print_tree(root: Path, max_depth: int = 3, max_entries_per_dir: int = 8) -> None:
    def _walk(path: Path, prefix: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return
        shown = entries[:max_entries_per_dir]
        for i, entry in enumerate(shown):
            connector = "|-- " if i < len(shown) - 1 or len(entries) > max_entries_per_dir else "`-- "
            print(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir():
                _walk(entry, prefix + "    ", depth + 1)
        if len(entries) > max_entries_per_dir:
            print(f"{prefix}... (+{len(entries) - max_entries_per_dir} itens)")

    print(f"{root}/")
    _walk(root, "", 1)


def summarize(root: Path) -> None:
    txt_files = list(root.rglob("*.txt"))
    jpg_files = list(root.rglob("*.jpg"))
    bbox_files = [p for p in txt_files if "bbox" in p.name.lower()]
    split_like = [p for p in txt_files if any(k in p.stem.lower() for k in ("train", "val", "test"))]

    print("\n=== Resumo ===")
    print(f"Total de .jpg encontrados: {len(jpg_files)}")
    print(f"Total de .txt encontrados: {len(txt_files)}")
    print(f"Arquivos que parecem ser de bbox (contem 'bbox' no nome): {len(bbox_files)}")
    if bbox_files:
        print(f"  exemplo: {bbox_files[0].relative_to(root)}")
        with open(bbox_files[0], "r", encoding="utf-8", errors="ignore") as f:
            print(f"  primeiras linhas: {[ln.strip() for ln in f.readlines()[:3]]}")
    print(f"Arquivos que parecem listas de split (train/val/test no nome): {len(split_like)}")
    for p in split_like[:10]:
        print(f"  {p.relative_to(root)}")

    print(
        "\nCompare isso com o docstring de src/logoloc/data/raw_adapter.py. "
        "Se os nomes/estrutura baterem, o pipeline deve funcionar direto. "
        "Se nao baterem, ajuste SPLIT_FILE_HINTS, _bboxes_file_for_image e "
        "_class_name_from_image_path nesse arquivo."
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python scripts/00_inspect_raw.py <pasta_do_dataset_extraido>")
        sys.exit(1)
    root = Path(sys.argv[1])
    if not root.exists():
        print(f"Pasta nao existe: {root}")
        sys.exit(1)
    print_tree(root)
    summarize(root)
