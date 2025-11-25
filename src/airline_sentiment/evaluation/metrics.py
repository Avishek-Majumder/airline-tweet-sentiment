"""
Evaluation metrics for airline tweet sentiment experiments.

This module provides helper functions to compute:
- Overall accuracy
- Per-class precision, recall, F1-score
- Macro-averaged F1-score

It is intended to be used across:
- Classical ML models
- CNN–BiLSTM model
- Transformer-based models
"""

from __future__ import annotations

from typing import Dict, Any, Iterable, Optional

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_classification_metrics(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    label_names: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """
    Compute accuracy, per-class precision/recall/F1, and macro F1.

    Parameters
    ----------
    y_true : Iterable[int]
        Ground-truth numeric label IDs.
    y_pred : Iterable[int]
        Predicted numeric label IDs.
    label_names : dict, optional
        Mapping from label_id -> label_str (e.g. {0: "negative", 1: "neutral", 2: "positive"}).
        If None, generic class_<id> names will be used.

    Returns
    -------
    Dict[str, Any]
        Dictionary with:
        - "accuracy"
        - "macro_f1"
        - precision_<label_name>, recall_<label_name>, f1_<label_name>, support_<label_name>
    """
    y_true_arr = np.asarray(list(y_true))
    y_pred_arr = np.asarray(list(y_pred))

    acc = accuracy_score(y_true_arr, y_pred_arr)

    # Use sorted unique labels present in y_true for deterministic ordering
    unique_labels = sorted(np.unique(y_true_arr).tolist())

    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        labels=unique_labels,
        zero_division=0,
    )

    macro_f1 = float(np.mean(f1s)) if len(f1s) > 0 else 0.0

    metrics: Dict[str, Any] = {
        "accuracy": float(acc),
        "macro_f1": macro_f1,
    }

    for idx, label_id in enumerate(unique_labels):
        p = float(precisions[idx])
        r = float(recalls[idx])
        f1 = float(f1s[idx])
        sup = int(supports[idx])

        if label_names is not None and label_id in label_names:
            label_str = label_names[label_id]
        else:
            label_str = f"class_{label_id}"

        metrics[f"precision_{label_str}"] = p
        metrics[f"recall_{label_str}"] = r
        metrics[f"f1_{label_str}"] = f1
        metrics[f"support_{label_str}"] = sup

    return metrics


__all__ = ["compute_classification_metrics"]
