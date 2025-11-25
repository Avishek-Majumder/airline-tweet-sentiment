"""
AdaBoost classifier for airline tweet sentiment experiments.

This module provides a helper function that constructs a scikit-learn
`AdaBoostClassifier` model using hyperparameters from `config/ml.yaml`.

Typical usage
-------------
from airline_sentiment.models.ml.adaboost import build_adaboost

clf = build_adaboost()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
"""

from __future__ import annotations

from typing import Dict, Any, Optional

from sklearn.ensemble import AdaBoostClassifier

from airline_sentiment.utils.config import load_ml_config


def build_adaboost(
    ml_config: Optional[Dict[str, Any]] = None
) -> AdaBoostClassifier:
    """
    Build an AdaBoostClassifier model configured via ml.yaml.

    Parameters
    ----------
    ml_config : dict, optional
        Configuration dictionary loaded from `ml.yaml`. If None, it is
        loaded automatically.

    Returns
    -------
    AdaBoostClassifier
        Configured AdaBoost classifier.
    """
    cfg = ml_config or load_ml_config()
    params = cfg.get("models", {}).get("adaboost", {})

    model = AdaBoostClassifier(
        n_estimators=params.get("n_estimators", 200),
        learning_rate=params.get("learning_rate", 0.5),
        algorithm=params.get("algorithm", "SAMME.R"),
        random_state=params.get("random_state", 42),
    )
    return model


__all__ = ["build_adaboost"]
