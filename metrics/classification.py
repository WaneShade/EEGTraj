# -*- coding: utf-8 -*-
"""Small classification metric wrappers used by experiment summaries."""

from typing import Dict, Iterable

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def classification_summary(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    """Return the common scalar metrics reported by the training engines."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
