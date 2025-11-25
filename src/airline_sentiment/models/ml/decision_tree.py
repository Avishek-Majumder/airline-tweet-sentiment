"""
Decision Tree classifier for airline tweet sentiment experiments.

This module provides a helper function that constructs a scikit-learn
`DecisionTreeClassifier` model using hyperparameters from `config/ml.yaml`.

Typical usage
-------------
from airline_sentiment.models.ml.decision_tree import build_decision_tree

clf = build_decision_tree()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
"""

from __future__ import annotations

from typing import Dict, Any, Optional

from sklearn.tree import DecisionTreeClassifier

from airline_sentiment.utils.config import load_ml_config


def build_decision_tree(
    ml_config: Optional[Dict[str, Any]] = None
) -> DecisionTreeClassifier:
    """
    Build a DecisionTreeClassifier model configured via ml.yaml.

    Parameters
    ----------
    ml_config : dict, optional
        Configuration dictionary loaded from `ml.yaml`. If None, it is
        loaded automatically.

    Returns
    -------
    DecisionTreeClassifier
        Configured decision tree classifier.
    """
    cfg = ml_config or load_ml_config()
    params = cfg.get("models", {}).get("decision_tree", {})

    model = DecisionTreeClassifier(
        criterion=params.get("criterion", "gini"),
        max_depth=params.get("max_depth", None),
        min_samples_split=params.get("min_samples_split", 2),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        class_weight=params.get("class_weight", "balanced"),
        random_state=cfg.get("ml_experiments", {}).get("random_seed", 42),
    )
    return model


__all__ = ["build_decision_tree"]
