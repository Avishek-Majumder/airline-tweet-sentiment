"""
k-Nearest Neighbors classifier for airline tweet sentiment experiments.

This module provides a helper function that constructs a scikit-learn
`KNeighborsClassifier` model using hyperparameters from `config/ml.yaml`.

Typical usage
-------------
from airline_sentiment.models.ml.knn import build_knn

clf = build_knn()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
"""

from __future__ import annotations

from typing import Dict, Any, Optional

from sklearn.neighbors import KNeighborsClassifier

from airline_sentiment.utils.config import load_ml_config


def build_knn(ml_config: Optional[Dict[str, Any]] = None) -> KNeighborsClassifier:
    """
    Build a KNeighborsClassifier model configured via ml.yaml.

    Parameters
    ----------
    ml_config : dict, optional
        Configuration dictionary loaded from `ml.yaml`. If None, it is
        loaded automatically.

    Returns
    -------
    KNeighborsClassifier
        Configured KNN classifier.
    """
    cfg = ml_config or load_ml_config()
    params = cfg.get("models", {}).get("knn", {})

    model = KNeighborsClassifier(
        n_neighbors=params.get("n_neighbors", 5),
        weights=params.get("weights", "distance"),
        metric=params.get("metric", "minkowski"),
        p=params.get("p", 2),
        n_jobs=-1,
    )
    return model


__all__ = ["build_knn"]
