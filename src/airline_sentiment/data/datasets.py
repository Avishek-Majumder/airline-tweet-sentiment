"""
Dataset loading utilities for airline tweet sentiment experiments.

This module is responsible for:
- Loading raw CSV files for each dataset (US airline, global airline, merged).
- Normalizing column names to a standard schema: "text" and "label".
- Mapping dataset-specific label formats to a unified label space:
    negative -> 0
    neutral  -> 1
    positive -> 2

We rely on paths and label mapping specified in config/config.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd

from airline_sentiment.utils.config import (
    load_global_config,
    PROJECT_ROOT,
)


def _get_global_config() -> Dict[str, Any]:
    """Lazy-load the global configuration."""
    return load_global_config()


def _resolve_raw_path(dataset_name: str, cfg: Dict[str, Any]) -> Path:
    """
    Resolve the path to the raw CSV file for a given dataset name.

    Parameters
    ----------
    dataset_name : str
        One of "airline_us", "airline_global", "airline_merged".
    cfg : dict
        Global configuration dictionary (from config.yaml).

    Returns
    -------
    Path
        Path to the raw CSV file.
    """
    datasets_cfg = cfg.get("datasets", {})
    if dataset_name not in datasets_cfg:
        raise KeyError(f"Unknown dataset name in config: {dataset_name}")

    ds_cfg = datasets_cfg[dataset_name]
    filename = ds_cfg["filename"]

    data_root = cfg.get("paths", {}).get("raw", "data/raw")
    raw_dir = PROJECT_ROOT / data_root

    return raw_dir / filename


def _map_label_to_str(label: Any) -> str:
    """
    Map a dataset-specific label (string or integer) to one of:
    "negative", "neutral", "positive".

    This function encodes common patterns for known airline sentiment datasets:
    - US airline Twitter dataset: labels often as strings
      ("negative", "neutral", "positive" or similar).
    - Sentiment140-like datasets: labels as integers (0, 2, 4).

    Parameters
    ----------
    label : Any
        Original label value.

    Returns
    -------
    str
        One of "negative", "neutral", "positive".

    Raises
    ------
    ValueError
        If the label cannot be mapped.
    """
    if label is None:
        raise ValueError("Encountered None label while mapping to sentiment string.")

    # String-like labels
    if isinstance(label, str):
        l = label.strip().lower()
        if l in {"neg", "negative", "bad"}:
            return "negative"
        if l in {"neu", "neutral"}:
            return "neutral"
        if l in {"pos", "positive", "good"}:
            return "positive"
        # Some datasets might already use these exact strings
        if l in {"0", "1", "2"}:
            # We assume these are already mapped as strings of indices
            mapping_idx = {"0": "negative", "1": "neutral", "2": "positive"}
            return mapping_idx[l]

    # Numeric-like labels (e.g., Sentiment140: 0=neg, 2=neutral, 4=pos)
    if isinstance(label, (int, float)):
        # Use int conversion to handle e.g. numpy types
        v = int(label)
        if v == 0:
            return "negative"
        if v == 1:
            # Some datasets might use 0/1/2
            return "neutral"
        if v == 2:
            # In a 0/1/2 scheme, 2 is positive
            return "positive"
        if v == 4:
            # Sentiment140: 4 is positive
            return "positive"

    raise ValueError(f"Could not map label value '{label}' to sentiment class.")


def _map_label_str_to_id(label_str: str, cfg: Dict[str, Any]) -> int:
    """
    Map a sentiment label string to a numeric ID using config.yaml.

    Parameters
    ----------
    label_str : str
        One of "negative", "neutral", "positive".
    cfg : dict
        Global configuration (provides the mapping).

    Returns
    -------
    int
        Numeric label ID (e.g., 0, 1, 2).
    """
    label_str = label_str.lower()
    mapping = cfg.get("labels", {}).get("mapping", {})
    if label_str not in mapping:
        raise KeyError(f"Label string '{label_str}' not found in config labels.mapping.")
    return int(mapping[label_str])


def _normalize_columns(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    Normalize column names for known airline tweet datasets.

    We standardize to:
    - "text"  column for tweet text
    - "label" column for the original dataset label value

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded from CSV.
    dataset_name : str
        Name of the dataset, used to apply dataset-specific heuristics.

    Returns
    -------
    pd.DataFrame
        Dataframe with at least "text" and "label" columns.
    """
    cols = {c.lower(): c for c in df.columns}

    text_col = None
    label_col = None

    # Common text column names
    for candidate in ["text", "tweet_text", "content"]:
        if candidate in cols:
            text_col = cols[candidate]
            break

    # Common label column names
    for candidate in ["label", "airline_sentiment", "sentiment", "polarity"]:
        if candidate in cols:
            label_col = cols[candidate]
            break

    if text_col is None:
        raise KeyError(
            f"Could not find a suitable text column in dataset '{dataset_name}'. "
            f"Available columns: {list(df.columns)}"
        )

    if label_col is None:
        raise KeyError(
            f"Could not find a suitable label column in dataset '{dataset_name}'. "
            f"Available columns: {list(df.columns)}"
        )

    # Create a copy with standardized column names
    df_norm = df.copy()
    df_norm.rename(columns={text_col: "text", label_col: "label"}, inplace=True)

    # Drop rows with missing text or label
    df_norm = df_norm.dropna(subset=["text", "label"])

    return df_norm[["text", "label"]]


def load_raw_dataset(dataset_name: str, config: Dict[str, Any] | None = None) -> pd.DataFrame:
    """
    Load a raw airline tweet dataset, normalize columns, and map labels.

    This function:
    1. Reads the CSV from data/raw/ as configured in config.yaml.
    2. Normalizes column names to "text" and "label".
    3. Maps dataset-specific label values to label IDs using the global mapping.

    Parameters
    ----------
    dataset_name : str
        One of:
        - "airline_us"
        - "airline_global"
        - "airline_merged"
    config : dict, optional
        Pre-loaded global config. If None, the config is loaded on demand.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - "text" : str, tweet text
        - "label_id" : int, numeric sentiment label (0/1/2)
        - "label_str" : str, sentiment string ("negative"/"neutral"/"positive")
    """
    cfg = config or _get_global_config()
    csv_path = _resolve_raw_path(dataset_name, cfg)

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Raw dataset file for '{dataset_name}' not found at: {csv_path}. "
            f"Please place the CSV there or update config.yaml."
        )

    df_raw = pd.read_csv(csv_path)
    df_norm = _normalize_columns(df_raw, dataset_name)

    # Map original labels to sentiment strings, then to numeric IDs
    label_strs = df_norm["label"].apply(_map_label_to_str)
    label_ids = label_strs.apply(lambda s: _map_label_str_to_id(s, cfg))

    df_out = df_norm.copy()
    df_out["label_str"] = label_strs
    df_out["label_id"] = label_ids

    return df_out[["text", "label_id", "label_str"]]


__all__ = [
    "load_raw_dataset",
]
