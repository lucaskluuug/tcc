#!/usr/bin/env python
"""Treina o Faster R-CNN. Wrapper fino sobre logoloc.train.train_faster_rcnn.

Uso:
    python scripts/04_train_faster_rcnn.py --model-config configs/faster_rcnn/resnet50.yaml --run-name faster_rcnn_resnet50_run1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from logoloc.train.train_faster_rcnn import main

if __name__ == "__main__":
    main()
