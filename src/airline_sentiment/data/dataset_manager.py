"""
Unified access to processed airline tweet sentiment datasets.

This module sits on top of:
- `airline_sentiment.preprocessing.splitter.prepare_and_split_dataset`
- `airline_sentiment.preprocessing.augmentation.balance_classes_with_augmentation`

and provides convenient helpers to:

1. Ensure that processed train/val/test splits exist for a dataset.
2. Load train/val/test splits from data/processed/<dataset_name>/.
3. Optionally apply augmentation-based class balancing to the train split.

Typical usage:
--------------
from airline_sentiment.data.dataset_manager import (
    ensure_processed_splits,
    load_processed_split,
    load_all_splits,
)

ensure_processed_splits("airline_us")
df_train = load_processed_split("airline_us", "train", apply_augmentation=True)
df_val   = load_processed_split("airline_us", "val")
df_test  = load_processed_split("airline_us", "test")
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Literal

import pandas as pd

from airline_sentiment.preprocessing.splitter import prepare_and_split_dataset
from airline_sentiment.preprocessing.augmentation import (
    Augmenter,
    balance_classes_with_augmentation,
)
from airline_sentiment.utils.config import load_global_config, PROJECT_ROOT


SplitName = Literal["train", "val", "test"]


def _get_processed_dir(cfg: Dict[str, Any], dataset_name: str) -> Path:
    """Return the directory where processed splits for a dataset are stored."""
    processed_root = cfg.get("paths", {}).get("processed", "data/processed")
    base_dir = PROJECT_ROOT / processed_root
    return base_dir / dataset_name


def ensure_processed_splits(
    dataset_name: str,
    config: Dict[str, Any] | None = None,
    force: bool = False,
) -> None:
    """
    Ensure processed train/val/test splits exist for a dataset.

    If they are missing, or if `force=True`, this function will call
    `prepare_and_split_dataset` to generate them.

    Parameters
    ----------
    dataset_name : str
        One of: "airline_us", "airline_global", "airline_merged".
    config : dict, optional
        Pre-loaded global config; if None, it is loaded automatically.
    force : bool, default False
        If True, always re-generate the splits, overwriting existing CSVs.
    """
    cfg = config or load_global_config()
    processed_dir = _get_processed_dir(cfg, dataset_name)

    expected_files = {
        "train": processed_dir / "train.csv",
        "val": processed_dir / "val.csv",
        "test": processed_dir / "test.csv",
    }

    if not force and all(p.is_file() for p in expected_files.values()):
        # All splits already exist; nothing to do.
        return

    # (Re-)generate splits
    prepare_and_split_dataset(dataset_name, config=cfg, save=True)


def _load_split_csv(
    dataset_name: str,
    split: SplitName,
    config: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Load a specific split CSV for a dataset."""
    cfg = config or load_global_config()
    processed_dir = _get_processed_dir(cfg, dataset_name)
    csv_path = processed_dir / f"{split}.csv"

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Processed split '{split}' for dataset '{dataset_name}' not found at: {csv_path}. "
            f"Run ensure_processed_splits('{dataset_name}') first."
        )

    return pd.read_csv(csv_path)


def load_processed_split(
    dataset_name: str,
    split: SplitName,
    apply_augmentation: bool = False,
    config: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Load a processed split (train/val/test) with optional augmentation.

    For the "train" split, you may enable augmentation-based class
    balancing via `apply_augmentation=True`. For "val" and "test",
    augmentation is always disabled (data must stay untouched).

    Parameters
    ----------
    dataset_name : str
        Dataset name ("airline_us", "airline_global", "airline_merged").
    split : {"train", "val", "test"}
        Split to load.
    apply_augmentation : bool, default False
        If True and split == "train", balance classes using augmentation.
    config : dict, optional
        Global config; if None, loaded automatically.

    Returns
    -------
    pd.DataFrame
        DataFrame containing at least:
        - text_raw
        - text_clean
        - label_id
        - label_str
        - split
        and optionally `is_augmented` when augmentation is applied.
    """
    cfg = config or load_global_config()

    # Ensure splits exist
    ensure_processed_splits(dataset_name, config=cfg, force=False)

    df = _load_split_csv(dataset_name, split, config=cfg)

    # Only augment the training split
    if split == "train" and apply_augmentation:
        augmenter = Augmenter(config=cfg)
        df = balance_classes_with_augmentation(
            df,
            augmenter=augmenter,
            label_col="label_str",
            text_col="text_clean",
            config=cfg,
        )

    return df


def load_all_splits(
    dataset_name: str,
    apply_augmentation_to_train: bool = False,
    config: Dict[str, Any] | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load all three splits (train/val/test) for a dataset.

    Parameters
    ----------
    dataset_name : str
        Dataset name ("airline_us", "airline_global", "airline_merged").
    apply_augmentation_to_train : bool, default False
        Whether to apply augmentation-based balancing to the training split.
    config : dict, optional
        Global config; if None, loaded automatically.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with keys "train", "val", "test".
    """
    cfg = config or load_global_config()

    ensure_processed_splits(dataset_name, config=cfg, force=False)

    train_df = load_processed_split(
        dataset_name,
        "train",
        apply_augmentation=apply_augmentation_to_train,
        config=cfg,
    )
    val_df = load_processed_split(dataset_name, "val", apply_augmentation=False, config=cfg)
    test_df = load_processed_split(dataset_name, "test", apply_augmentation=False, config=cfg)

    return {"train": train_df, "val": val_df, "test": test_df}


__all__ = ["ensure_processed_splits", "load_processed_split", "load_all_splits"]
