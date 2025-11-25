"""
Logistic Regression classifier for airline tweet sentiment experiments.

This module provides a helper function that constructs a scikit-learn
`LogisticRegression` model using hyperparameters from `config/ml.yaml`.

Typical usage
-------------
from airline_sentiment.models.ml.logistic_regression import build_logistic_regression

clf = build_logistic_regression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
"""

from __future__ import annotations

from typing import Dict, Any, Optional

from sklearn.linear_model import LogisticRegression

from airline_sentiment.utils.config import load_ml_config


def build_logistic_regression(ml_config: Optional[Dict[str, Any]] = None) -> LogisticRegression:
    """
    Build a LogisticRegression model configured via ml.yaml.

    Parameters
    ----------
    ml_config : dict, optional
        Configuration dictionary loaded from `ml.yaml`. If None, it is
        loaded automatically.

    Returns
    -------
    LogisticRegression
        Configured logistic regression classifier.
    """
    cfg = ml_config or load_ml_config()
    params = cfg.get("models", {}).get("logistic_regression", {})

    model = LogisticRegression(
        C=params.get("C", 1.0),
        penalty=params.get("penalty", "l2"),
        solver=params.get("solver", "lbfgs"),
        max_iter=params.get("max_iter", 1000),
        class_weight=params.get("class_weight", "balanced"),
        n_jobs=-1,
    )
    return model


__all__ = ["build_logistic_regression"]
