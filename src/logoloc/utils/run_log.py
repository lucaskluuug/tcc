from __future__ import annotations

import dataclasses
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


def _gpu_info() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode().strip()
    except Exception:
        return "sem GPU NVIDIA detectada (nvidia-smi indisponivel)"


def hardware_snapshot() -> dict[str, Any]:
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        torch_version = torch.__version__
    except ImportError:
        cuda_available = False
        torch_version = "torch nao instalado"

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "gpu": _gpu_info(),
    }


class RunLogger:
    def __init__(self, run_dir: str | Path, run_name: str, config: Any):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.config = config
        self._start_time = time.time()

    def _config_as_dict(self) -> Any:
        if dataclasses.is_dataclass(self.config):
            return dataclasses.asdict(self.config)
        return self.config

    def save(self, extra: dict[str, Any] | None = None) -> Path:
        elapsed_s = time.time() - self._start_time
        payload = {
            "run_name": self.run_name,
            "config": self._config_as_dict(),
            "hardware": hardware_snapshot(),
            "training_time_seconds": elapsed_s,
            "training_time_human": f"{elapsed_s / 60:.1f} min",
        }
        if extra:
            payload.update(extra)

        out_path = self.run_dir / "run_config.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        return out_path
