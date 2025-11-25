#!/usr/bin/env python
"""
Prepare processed train/val/test splits for airline tweet sentiment datasets.

This script:
- Loads global config from config/config.yaml.
- Uses the preprocessing/splitter pipeline to:
    * load raw CSVs from data/raw/
    * clean the text
    * perform stratified train/val/test splits
- Saves splits under data/processed/<dataset_name>/train.csv, val.csv, test.csv

Usage
-----
# Prepare all configured datasets
python scripts/prepare_datasets.py

# Prepare a specific dataset only
python scripts/prepare_datasets.py --dataset airline_us
python scripts/prepare_datasets.py --dataset airline_global
python scripts/prepare_datasets.py --dataset airline_merged
"""

from __future__ import annotations

import argparse
from typing import Dict, Any

from airline_sentiment.preprocessing.splitter import (
    prepare_and_split_dataset,
    prepare_all_datasets,
)
from airline_sentiment.utils.config import load_global_config
from airline_sentiment.utils.logging_utils import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare processed train/val/test splits for airline tweet datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=["airline_us", "airline_global", "airline_merged"],
        help=(
            "Optional dataset name. If omitted, all datasets defined in "
            "config/config.yaml are prepared."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger(__name__)

    cfg: Dict[str, Any] = load_global_config()

    if args.dataset is None:
        logger.info("Preparing processed splits for ALL configured datasets...")
        prepare_all_datasets(config=cfg, save=True)
        logger.info("All datasets have been prepared.")
    else:
        logger.info("Preparing processed splits for dataset: %s", args.dataset)
        prepare_and_split_dataset(args.dataset, config=cfg, save=True)
        logger.info("Finished preparing dataset: %s", args.dataset)


if __name__ == "__main__":
    main()
