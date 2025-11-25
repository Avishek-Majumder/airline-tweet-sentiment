"""
Support Vector Machine (SVM) classifier for airline tweet sentiment experiments.

This module provides a helper function that constructs a scikit-learn
`SVC` model using hyperparameters from `config/ml.yaml`.

Typical usage
-------------
from airline_sentiment.models.ml.svm import build_svm

clf = build_svm()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
"""

from __future__ import annotations

from typing import Dict, Any, Optional

from sklearn.svm import SVC

from airline_sentiment.utils.config import load_ml_config


def build_svm(ml_config: Optional[Dict[str, Any]] = None) -> SVC:
    """
    Build an SVM classifier (SVC) configured via ml.yaml.

    Parameters
    ----------
    ml_config : dict, optional
        Configuration dictionary loaded from `ml.yaml`. If None, it is
        loaded automatically.

    Returns
    -------
    SVC
        Configured SVM classifier.
    """
    cfg = ml_config or load_ml_config()
    params = cfg.get("models", {}).get("svm", {})

    model = SVC(
        C=params.get("C", 1.0),
        kernel=params.get("kernel", "linear"),
        gamma=params.get("gamma", "scale"),
        class_weight=params.get("class_weight", "balanced"),
        probability=params.get("probability", True),
    )
    return model


__all__ = ["build_svm"]
