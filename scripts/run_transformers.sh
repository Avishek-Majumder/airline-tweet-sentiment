#!/usr/bin/env python
"""
Run transformer experiments for airline tweet sentiment datasets.

This script:
- Loads global config from config/config.yaml.
- Uses `run_transformer_experiments_for_dataset` to:
    * load processed train/val/test splits
    * build transformer datasets with HuggingFace tokenizers
    * fine-tune pre-trained transformer models (BERT, ELECTRA, etc.)
    * evaluate on the test set (accuracy, per-class precision/recall/F1, macro F1)
- Saves results to outputs/transformers_results_<dataset>.csv

Usage
-----
# Run transformers for all datasets and all enabled models
python scripts/run_transformers.py

# Run transformers for a specific dataset (all enabled models)
python scripts/run_transformers.py --dataset airline_us
python scripts/run_transformers.py --dataset airline_global
python scripts/run_transformers.py --dataset airline_merged

# Run a single transformer model (by key from transformers.yaml) on a dataset
python scripts/run_transformers.py --dataset airline_us --model_key bert_base
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from airline_sentiment.models.transformers.trainer import (
    run_transformer_experiments_for_dataset,
)
from airline_sentiment.utils.config import load_global_config
from airline_sentiment.utils.logging_utils import get_logger
from airline_sentiment.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run transformer experiments for airline tweet sentiment."
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
    parser.add_argument(
        "--model_key",
        type=str,
        default=None,
        help=(
            "Optional model key from transformers.yaml (e.g. 'bert_base', "
            "'electra_base', 'roberta_base'). If omitted, all enabled "
            "models are run for each dataset."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger(__name__)
    set_global_seed()

    cfg = load_global_config()
    datasets_cfg = cfg.get("datasets", {})

    # Determine which datasets to run
    if args.dataset is not None:
        dataset_names: List[str] = [args.dataset]
    else:
        dataset_names = list(datasets_cfg.keys())
        if not dataset_names:
            raise ValueError("No datasets configured in config/config.yaml.")

    for ds in dataset_names:
        logger.info(
            "Running transformer experiments for dataset: %s (model_key=%s)",
            ds,
            args.model_key,
        )
        results_df = run_transformer_experiments_for_dataset(
            dataset_name=ds,
            single_model_key=args.model_key,
            save_results=True,
        )
        logger.info(
            "Finished transformer experiments for dataset: %s (rows: %d)",
            ds,
            len(results_df),
        )


if __name__ == "__main__":
    main()
