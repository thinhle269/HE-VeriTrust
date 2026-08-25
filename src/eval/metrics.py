
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


def compute_metrics(y_true, y_pred, num_classes: int,
                    loss: Optional[float] = None,
                    label_names=None, with_confusion: bool = False) -> Dict:
    labels = list(range(int(num_classes)))
    if len(y_true) == 0:
        out = {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0}
        if loss is not None:
            out["loss"] = float(loss)
        return out
    per = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels,
                                   average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels,
                                      average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels,
                                                 average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels,
                                           average="macro", zero_division=0)),
        "min_class_f1": float(np.min(per)),
    }
    names = label_names or [str(i) for i in labels]
    out["per_class_f1"] = {n: float(v) for n, v in zip(names, per)}
    if loss is not None:
        out["loss"] = float(loss)
    if with_confusion:
        out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels)
    return out


def class_weights(counts: np.ndarray, mode: str = "effective_number",
                  beta: float = 0.9999) -> Optional[np.ndarray]:

    counts = np.asarray(counts, dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    mode = (mode or "none").lower()
    if mode in ("none", "off", ""):
        return None
    if mode == "inverse":
        w = counts.sum() / counts
    else:
        eff = (1.0 - np.power(float(beta), counts)) / (1.0 - float(beta))
        w = counts.sum() / eff
    return (w / w.sum() * len(counts)).astype(np.float32)
