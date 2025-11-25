"""
Train/validation/test splitting for airline tweet sentiment datasets.

This module:
- Loads a raw dataset via `airline_sentiment.data.datasets.load_raw_dataset`.
- Cleans the text using `TextCleaner` from `preprocessing.text_cleaning`.
- Performs stratified train/val/test splits using config/config.yaml.
- Optionally saves the processed splits to data/processed/<dataset_name>/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split

from airline_sentiment.data.datasets import load_raw_dataset
from airline_sentiment.preprocessing.text_cleaning import TextCleaner
from airline_sentiment.utils.config import load_global_config, PROJECT_ROOT


def _get_split_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract split-related settings from the global config."""
    split_cfg = cfg.get("split", {})
    return {
        "test_size": float(split_cfg.get("test_size", 0.2)),
        "val_size": float(split_cfg.get("val_size", 0.1)),
        "stratify": bool(split_cfg.get("stratify", True)),
    }


def _get_processed_dir(cfg: Dict[str, Any], dataset_name: str) -> Path:
    """
    Compute the directory where processed splits for a dataset will be stored.

    Default: data/processed/<dataset_name>/
    """
    processed_root = cfg.get("paths", {}).get("processed", "data/processed")
    base_dir = PROJECT_ROOT / processed_root
    ds_dir = base_dir / dataset_name
    ds_dir.mkdir(parents=True, exist_ok=True)
    return ds_dir


def prepare_and_split_dataset(
    dataset_name: str,
    config: Dict[str, Any] | None = None,
    save: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load, clean, and split an airline sentiment dataset into train/val/test.

    Steps:
    1. Load raw dataset (text + label_id + label_str).
    2. Clean text using TextCleaner according to config/config.yaml.
    3. Perform stratified train/val/test split using config.split settings.
    4. Optionally save splits to data/processed/<dataset_name>/.

    Parameters
    ----------
    dataset_name : str
        One of: "airline_us", "airline_global", "airline_merged".
    config : dict, optional
        Pre-loaded global config. If None, config is loaded on demand.
    save : bool, default True
        If True, save train/val/test CSVs to the processed directory.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with keys "train", "val", "test".
    """
    cfg = config or load_global_config()
    split_cfg = _get_split_config(cfg)

    # 1. Load raw dataset
    df = load_raw_dataset(dataset_name, config=cfg)

    # 2. Clean text
    cleaner = TextCleaner(config=cfg)
    df = df.copy()
    df["text_raw"] = df["text"]
    df["text_clean"] = cleaner.clean_many(df["text_raw"])

    # 3. Perform stratified train/test split
    test_size = split_cfg["test_size"]
    val_size = split_cfg["val_size"]
    stratify = df["label_id"] if split_cfg["stratify"] else None

    df_train_val, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=cfg.get("project", {}).get("random_seed", 42),
        stratify=stratify,
    )

    # Now split train_val into train and val
    # val_size is a fraction of (train_val + train), so we convert it to a
    # fraction of df_train_val.
    relative_val_size = val_size / (1.0 - test_size) if (1.0 - test_size) > 0 else val_size

    stratify_train_val = (
        df_train_val["label_id"] if split_cfg["stratify"] else None
    )

    df_train, df_val = train_test_split(
        df_train_val,
        test_size=relative_val_size,
        random_state=cfg.get("project", {}).get("random_seed", 42),
        stratify=stratify_train_val,
    )

    # Add split column for clarity
    df_train = df_train.copy()
    df_train["split"] = "train"
    df_val = df_val.copy()
    df_val["split"] = "val"
    df_test = df_test.copy()
    df_test["split"] = "test"

    splits = {
        "train": df_train,
        "val": df_val,
        "test": df_test,
    }

    # 4. Save CSVs if requested
    if save:
        out_dir = _get_processed_dir(cfg, dataset_name)
        for split_name, df_split in splits.items():
            out_path = out_dir / f"{split_name}.csv"
            df_split.to_csv(out_path, index=False)

    return splits


def prepare_all_datasets(
    config: Dict[str, Any] | None = None,
    save: bool = True,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Convenience function to prepare and split all configured datasets.

    Returns a nested dictionary:
        {
            "airline_us": { "train": df, "val": df, "test": df },
            "airline_global": { ... },
            "airline_merged": { ... },
        }
    """
    cfg = config or load_global_config()
    datasets_cfg = cfg.get("datasets", {})
    results: Dict[str, Dict[str, pd.DataFrame]] = {}

    for dataset_name in datasets_cfg.keys():
        splits = prepare_and_split_dataset(dataset_name, config=cfg, save=save)
        results[dataset_name] = splits

    return results


__all__ = ["prepare_and_split_dataset", "prepare_all_datasets"]
