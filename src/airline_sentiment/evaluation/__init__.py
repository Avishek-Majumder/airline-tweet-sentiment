"""
Evaluation utilities for airline tweet sentiment experiments.

This subpackage provides two main capabilities:

1. Core classification metrics
   - Shared accuracy and F1-style metrics used by:
     * Classical ML models
     * CNN–BiLSTM
     * Transformer-based models

2. Result aggregation and analysis
   - Helpers to load experiment result CSVs from `outputs/`
     (ML, CNN–BiLSTM, transformers).
   - Functions to summarize and compare models across datasets.

Modules
-------
- `metrics.py`:
    `compute_classification_metrics(y_true, y_pred, label_names=None)`
    returns a dictionary with:
      * accuracy
      * macro_f1
      * per-class precision, recall, F1, and support.

- `analysis.py`:
    - `load_results_df(pattern="ml_results_*.csv")`:
        Load and concatenate result CSVs from `outputs/`.
    - `summarize_best_per_dataset(results_df, metric="macro_f1", group_cols=None)`:
        Return the best configuration per dataset, based on a chosen metric.
"""

from __future__ import annotations

from .metrics import compute_classification_metrics
from .analysis import (
    load_results_df,
    summarize_best_per_dataset,
)

__all__ = [
    "compute_classification_metrics",
    "load_results_df",
    "summarize_best_per_dataset",
]
