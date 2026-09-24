from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)


def cached(cache_path: str | Path, compute_fn: Callable[[], T], force: bool = False) -> T:
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        logger.info("Lendo do cache: %s", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    logger.info("Cache ausente (ou force=True), computando: %s", cache_path)
    value = compute_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(value, f)
    return value


def clear_cache(cache_path: str | Path) -> None:
    cache_path = Path(cache_path)
    if cache_path.exists():
        cache_path.unlink()
        logger.info("Cache removido: %s", cache_path)
