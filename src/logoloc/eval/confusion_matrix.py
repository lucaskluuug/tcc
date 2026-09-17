from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .classification_metrics import ClassificationReport


def plot_confusion_matrix(report: ClassificationReport, output_path: str | Path, normalize: bool = True) -> Path:
    cm = report.confusion.astype(float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(max(8, len(report.class_names) * 0.4), max(6, len(report.class_names) * 0.4)))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(report.class_names)))
    ax.set_yticks(range(len(report.class_names)))
    ax.set_xticklabels(report.class_names, rotation=90, fontsize=6)
    ax.set_yticklabels(report.class_names, fontsize=6)
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe verdadeira")
    ax.set_title("Matriz de confusao" + (" (normalizada por linha)" if normalize else ""))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
