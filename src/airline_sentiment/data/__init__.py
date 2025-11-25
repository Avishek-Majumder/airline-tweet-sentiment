"""
Data loading and management for airline tweet sentiment experiments.

This subpackage provides:

- High-level access to raw airline tweet datasets from `data/raw/`.
- Normalization of column names and label mappings.
- Management of processed train/val/test splits with optional
  augmentation-based class balancing.

Main entrypoints
----------------
- `load_raw_dataset(...)`:
    Load a raw CSV for a configured dataset (e.g., airline_us) and
    normalize columns to a canonical schema.

- `ensure_processed_splits(...)`:
    Make sure processed train/val/test splits exist for a dataset,
    creating them if necessary via the preprocessing pipeline.

- `load_all_splits(...)`:
    Load train/val/test processed DataFrames for a dataset, optionally
    applying augmentation-based balancing to the training split.
"""

from __future__ import annotations

from .datasets import load_raw_dataset
from .dataset_manager import (
    ensure_processed_splits,
    load_processed_split,
    load_all_splits,
)

__all__ = [
    "load_raw_dataset",
    "ensure_processed_splits",
    "load_processed_split",
    "load_all_splits",
]
