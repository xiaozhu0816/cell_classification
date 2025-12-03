from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn import metrics


@dataclass
class AverageMeter:
    name: str
    value: float = 0.0
    sum: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.count += n
        self.value = val

    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


def binary_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)
    results = {
        "accuracy": metrics.accuracy_score(labels, preds),
        "precision": metrics.precision_score(labels, preds, zero_division=0),
        "recall": metrics.recall_score(labels, preds, zero_division=0),
        "f1": metrics.f1_score(labels, preds, zero_division=0),
    }
    try:
        results["auc"] = metrics.roc_auc_score(labels, probs)
    except ValueError:
        results["auc"] = float("nan")
    return results
