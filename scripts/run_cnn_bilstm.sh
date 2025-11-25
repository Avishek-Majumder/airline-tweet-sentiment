#!/usr/bin/env python
"""
Run CNN–BiLSTM experiments for airline tweet sentiment datasets.

This script:
- Loads global config from config/config.yaml.
- Uses `train_and_evaluate_cnn_bilstm` to:
    * load processed train/val/test splits
    * train a CNN–BiLSTM model with Word2Vec embeddings
    * evaluate on the test set (accuracy, per-class precision/recall/F1, macro F1)
- Saves results to outputs/cnn_bilstm_results_<dataset>.csv

Usage
-----
# Run CNN–BiLSTM for all datasets
python scripts/run_cnn_bilstm.py

# Run CNN–BiLSTM for a specific dataset
python scripts/run_cnn_bilstm.py --dataset airline_us
python scripts/run_cnn_bilstm.py --dataset airline_global
python scripts/run_cnn_bilstm.py --dataset airline_merged
"""

from __future__ import annotations

import argparse
from typing import List

from airline_sentiment.models.dl.cnn_bilstm import train_and_evaluate_cnn_bilstm
from airline_sentiment.utils.config import load_global_config
from airline_sentiment.utils.logging_utils import get_logger
from airline_sentiment.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CNN–BiLSTM experiments for airline tweet sentiment."
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
        logger.info("Running CNN–BiLSTM experiment for dataset: %s", ds)
        metrics = train_and_evaluate_cnn_bilstm(dataset_name=ds)
        logger.info(
            "Finished CNN–BiLSTM experiment for dataset: %s (macro_f1: %.4f, accuracy: %.4f)",
            ds,
            float(metrics.get("macro_f1", 0.0)),
            float(metrics.get("accuracy", 0.0)),
        )


if __name__ == "__main__":
    main()
