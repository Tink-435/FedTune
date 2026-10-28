"""Evaluator: global test metrics (FR 24). Accuracy and macro-F1 for classification."""
from __future__ import annotations

import numpy as np


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    acc = float((y_true == y_pred).mean()) if len(y_true) else float("nan")
    f1s = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        if tp + fp + fn == 0:
            continue                       # class absent from both: excluded from the macro average
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return {"accuracy": round(acc, 6), "macro_f1": round(float(np.mean(f1s)) if f1s else float("nan"), 6)}


class Evaluator:
    """Evaluates a global adapter on a coordinator-held set (P6): the global validation set V_g
    every round, the test set T once at the end (or every round if evaluation.test_every_round)."""

    def __init__(self, host, test, num_classes: int):
        self.host, self.test, self.num_classes = host, test, num_classes

    def __call__(self, adapter, head) -> dict:
        if self.test is None or len(self.test) == 0:
            return {"accuracy": float("nan"), "macro_f1": float("nan")}
        self.host.load_adapter(adapter, head)
        return classification_metrics(self.test.labels, self.host.predict(self.test.texts), self.num_classes)
