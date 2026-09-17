from __future__ import annotations

import dataclasses

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


@dataclasses.dataclass
class ClassificationReport:
    class_names: list[str]
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    support: np.ndarray
    accuracy: float
    confusion: np.ndarray

    def macro_f1(self) -> float:
        return float(np.mean(self.f1))

    def as_table(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "classe": self.class_names,
                "precisao": self.precision,
                "revocacao": self.recall,
                "f1": self.f1,
                "suporte": self.support,
            }
        )


def classification_report(
    y_true: list[str], y_pred: list[str], class_names: list[str]
) -> ClassificationReport:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=class_names, zero_division=0
    )
    accuracy = float(np.mean([t == p for t, p in zip(y_true, y_pred)])) if y_true else 0.0
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    return ClassificationReport(
        class_names=class_names,
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
        accuracy=accuracy,
        confusion=cm,
    )
