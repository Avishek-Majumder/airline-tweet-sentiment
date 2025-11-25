#!/usr/bin/env python
"""
Run classical ML experiments for airline tweet sentiment datasets.

This script:
- Loads global config from config/config.yaml.
- Uses `run_ml_experiments_for_dataset` to:
    * load processed train/val/test splits
    * extract TF–IDF / BoW features
    * train and evaluate classical ML models (LR, KNN, SVM, DT, RF, AdaBoost)
- Saves results to outputs/ml_results_<dataset>.csv

Usage
-----
# Run experiments for all datasets
python scripts/run_ml_experiments.py

# Run experiments for a specific dataset
python scripts/run_ml_experiments.py --dataset airline_us
python scripts/run_ml_experiments.py --dataset airline_global
python scripts/run_ml_experiments.py --dataset airline_merged
"""

from __future__ import annotations

import argparse
from typing import List

from airline_sentiment.models.ml.experiment_runner import run_ml_experiments_for_dataset
from airline_sentiment.utils.config import load_global_config
from airline_sentiment.utils.logging_utils import get_logger
from airline_sentiment.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run classical ML experiments for airline tweet sentiment."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["airline_us", "airline_global", "airline_merged"],
        help=(
            "Optional dataset name. If omitted, experiments are run for "
            "all datasets defined in config/config.yaml."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger(__name__)
    set_global_seed()

    cfg = load_global_config()
    datasets_cfg = cfg.get("datasets", {})

    if args.dataset is not None:
        dataset_names: List[str] = [args.dataset]
    else:
        dataset_names = list(datasets_cfg.keys())
        if not dataset_names:
            raise ValueError("No datasets configured in config/config.yaml.")

    for ds in dataset_names:
        logger.info("Running classical ML experiments for dataset: %s", ds)
        results_df = run_ml_experiments_for_dataset(ds, apply_augmentation_to_train=True)
        logger.info(
            "Finished ML experiments for dataset: %s (rows: %d)",
            ds,
            len(results_df),
        )


if __name__ == "__main__":
    main()
