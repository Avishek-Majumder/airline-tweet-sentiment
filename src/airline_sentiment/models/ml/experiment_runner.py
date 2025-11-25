"""
Classical ML experiment runner for airline tweet sentiment.

This module orchestrates:
- Loading processed train/val/test splits.
- Vectorizing text using TF–IDF or Bag-of-Words.
- Training/evaluating classical ML models (LR, KNN, SVM, DT, RF, AdaBoost).
- Computing accuracy, precision, recall, and F1-score.
- Saving aggregated results to a CSV file for further analysis.

It uses:
- config/config.yaml      (global settings)
- config/ml.yaml          (ML & feature configs)
- airline_sentiment.data.dataset_manager
- airline_sentiment.features.tfidf_vectorizer
- airline_sentiment.features.bow_vectorizer
- airline_sentiment.models.ml.*   (model builders)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

from airline_sentiment.data.dataset_manager import load_all_splits
from airline_sentiment.features.tfidf_vectorizer import TfidfFeatureExtractor
from airline_sentiment.features.bow_vectorizer import BowFeatureExtractor
from airline_sentiment.models.ml.logistic_regression import build_logistic_regression
from airline_sentiment.models.ml.knn import build_knn
from airline_sentiment.models.ml.svm import build_svm
from airline_sentiment.models.ml.decision_tree import build_decision_tree
from airline_sentiment.models.ml.random_forest import build_random_forest
from airline_sentiment.models.ml.adaboost import build_adaboost
from airline_sentiment.utils.config import (
    load_global_config,
    load_ml_config,
    PROJECT_ROOT,
)


MODEL_BUILDERS = {
    "logistic_regression": build_logistic_regression,
    "knn": build_knn,
    "svm": build_svm,
    "decision_tree": build_decision_tree,
    "random_forest": build_random_forest,
    "adaboost": build_adaboost,
}


def _compute_metrics(
    y_true,
    y_pred,
    label_names: Dict[int, str] | None = None,
) -> Dict[str, Any]:
    """
    Compute accuracy, per-class precision/recall/F1, and macro F1.

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground-truth and predicted label IDs.
    label_names : dict, optional
        Mapping from numeric label_id -> label_str (e.g. 0->"negative").

    Returns
    -------
    Dict[str, Any]
        Dictionary of metrics suitable for tabular logging.
    """
    acc = accuracy_score(y_true, y_pred)

    # Per-class precision, recall, F1, support (ordered by label index)
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=sorted(set(y_true)),
        zero_division=0,
    )

    macro_f1 = f1s.mean() if len(f1s) > 0 else 0.0

    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "macro_f1": macro_f1,
    }

    for idx, (p, r, f1, sup) in enumerate(zip(precisions, recalls, f1s, supports)):
        label_str = label_names.get(idx, f"class_{idx}") if label_names else f"class_{idx}"
        metrics[f"precision_{label_str}"] = p
        metrics[f"recall_{label_str}"] = r
        metrics[f"f1_{label_str}"] = f1
        metrics[f"support_{label_str}"] = sup

    return metrics


def run_ml_experiments_for_dataset(
    dataset_name: str,
    apply_augmentation_to_train: bool = True,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Run classical ML experiments on a single dataset.

    For the given dataset (e.g. "airline_us"), this function:
    - Loads train/val/test splits (optionally with augmentation on train).
    - Builds TF–IDF and/or BoW feature extractors.
    - For each enabled model in ml.yaml:
        - Trains on the training set.
        - Evaluates on the test set.
        - Logs evaluation metrics.

    Parameters
    ----------
    dataset_name : str
        One of:
        - "airline_us"
        - "airline_global"
        - "airline_merged"
    apply_augmentation_to_train : bool, default True
        Whether to apply augmentation-based class balancing on the
        training split.
    save_results : bool, default True
        If True, save a CSV of results under outputs/ml_results_<dataset>.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame containing one row per (model, feature_type) with
        metrics.
    """
    global_cfg = load_global_config()
    ml_cfg = load_ml_config()

    # Load data splits (train/val/test)
    splits = load_all_splits(
        dataset_name,
        apply_augmentation_to_train=apply_augmentation_to_train,
        config=global_cfg,
    )
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    # Map numeric labels to strings for metric naming
    label_mapping = global_cfg.get("labels", {}).get("mapping", {})
    id_to_label = {int(v): k for k, v in label_mapping.items()}

    # Feature configuration
    ml_exp_cfg = ml_cfg.get("ml_experiments", {})
    use_tfidf = bool(ml_exp_cfg.get("features", {}).get("use_tfidf", True))
    use_bow = bool(ml_exp_cfg.get("features", {}).get("use_bow", True))
    enabled_models: List[str] = list(ml_exp_cfg.get("enabled_models", []))

    if not enabled_models:
        raise ValueError("No ML models enabled in ml.yaml (ml_experiments.enabled_models).")

    results: List[Dict[str, Any]] = []

    # Prepare training labels
    y_train = train_df["label_id"].values
    y_val = val_df["label_id"].values
    y_test = test_df["label_id"].values

    # Text columns (already cleaned)
    X_train_text = train_df["text_clean"].tolist()
    X_val_text = val_df["text_clean"].tolist()
    X_test_text = test_df["text_clean"].tolist()

    # ------------------------------------------------------------------
    # Helper to run a single model + feature type combination
    # ------------------------------------------------------------------
    def _run_single(
        model_name: str,
        feature_type: str,
        X_train_text,
        X_val_text,
        X_test_text,
    ):
        if model_name not in MODEL_BUILDERS:
            raise KeyError(f"Model '{model_name}' not recognized in MODEL_BUILDERS.")

        # Build features
        if feature_type == "tfidf":
            extractor = TfidfFeatureExtractor(ml_config=ml_cfg)
        elif feature_type == "bow":
            extractor = BowFeatureExtractor(ml_config=ml_cfg)
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

        X_train = extractor.fit_transform(X_train_text)
        # We transform val/test so that shapes match for evaluation if needed
        X_val = extractor.transform(X_val_text)
        X_test = extractor.transform(X_test_text)

        # Build model
        builder = MODEL_BUILDERS[model_name]
        model = builder(ml_config=ml_cfg)

        # Fit on training data
        model.fit(X_train, y_train)

        # Evaluate on test data
        y_pred = model.predict(X_test)

        metrics = _compute_metrics(y_test, y_pred, label_names=id_to_label)

        # Attach metadata
        record: Dict[str, Any] = {
            "dataset": dataset_name,
            "model": model_name,
            "feature_type": feature_type,
        }
        record.update(metrics)
        return record

    # ------------------------------------------------------------------
    # Run experiments for all enabled models and features
    # ------------------------------------------------------------------
    if use_tfidf:
        for model_name in enabled_models:
            results.append(
                _run_single(model_name, "tfidf", X_train_text, X_val_text, X_test_text)
            )

    if use_bow:
        for model_name in enabled_models:
            results.append(
                _run_single(model_name, "bow", X_train_text, X_val_text, X_test_text)
            )

    results_df = pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if save_results:
        outputs_root = global_cfg.get("paths", {}).get("outputs", "outputs")
        out_dir = PROJECT_ROOT / outputs_root
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"ml_results_{dataset_name}.csv"
        results_df.to_csv(out_path, index=False)

    return results_df


__all__ = ["run_ml_experiments_for_dataset"]
