"""
Classical machine learning models for airline tweet sentiment.

This subpackage implements the baseline models used in our experiments
on airline service review sentiment, operating on TF–IDF or Bag-of-Words
features:

Models
------
- Logistic Regression
- k-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- AdaBoost

The typical workflow is orchestrated via `experiment_runner.py`, which:

1. Loads processed train/val/test splits via
   `airline_sentiment.data.dataset_manager`.
2. Builds TF–IDF and/or BoW feature representations.
3. Trains the enabled models (as configured in `config/ml.yaml`).
4. Evaluates them on the test set using shared classification metrics.
5. Logs all results to `outputs/ml_results_<dataset>.csv`.
"""

from __future__ import annotations

from .logistic_regression import build_logistic_regression
from .knn import build_knn
from .svm import build_svm
from .decision_tree import build_decision_tree
from .random_forest import build_random_forest
from .adaboost import build_adaboost
from .experiment_runner import run_ml_experiments_for_dataset

__all__ = [
    "build_logistic_regression",
    "build_knn",
    "build_svm",
    "build_decision_tree",
    "build_random_forest",
    "build_adaboost",
    "run_ml_experiments_for_dataset",
]
