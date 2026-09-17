#!/usr/bin/env python
"""Treina o YOLOv8. Wrapper fino sobre logoloc.train.train_yolo.

Uso:
    python scripts/05_train_yolo.py --model-config configs/yolov8/s.yaml --run-name yolov8s_run1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from logoloc.train.train_yolo import main

if __name__ == "__main__":
    main()
