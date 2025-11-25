"""
Random Forest classifier for airline tweet sentiment experiments.

This module provides a helper function that constructs a scikit-learn
`RandomForestClassifier` model using hyperparameters from `config/ml.yaml`.

Typical usage
-------------
from airline_sentiment.models.ml.random_forest import build_random_forest

clf = build_random_forest()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
"""

from __future__ import annotations

from typing import Dict, Any, Optional

from sklearn.ensemble import RandomForestClassifier

from airline_sentiment.utils.config import load_ml_config


def build_random_forest(
    ml_config: Optional[Dict[str, Any]] = None
) -> RandomForestClassifier:
    """
    Build a RandomForestClassifier model configured via ml.yaml.

    Parameters
    ----------
    ml_config : dict, optional
        Configuration dictionary loaded from `ml.yaml`. If None, it is
        loaded automatically.

    Returns
    -------
    RandomForestClassifier
        Configured random forest classifier.
    """
    cfg = ml_config or load_ml_config()
    params = cfg.get("models", {}).get("random_forest", {})

    model = RandomForestClassifier(
        n_estimators=params.get("n_estimators", 300),
        criterion=params.get("criterion", "gini"),
        max_depth=params.get("max_depth", None),
        min_samples_split=params.get("min_samples_split", 2),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        bootstrap=params.get("bootstrap", True),
        n_jobs=params.get("n_jobs", -1),
        class_weight=params.get("class_weight", "balanced"),
        random_state=params.get("random_state", 42),
    )
    return model


__all__ = ["build_random_forest"]
