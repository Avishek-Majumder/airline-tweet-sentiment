"""
Result aggregation and analysis utilities.

This module provides helpers to:
- Load experiment result CSVs (e.g., ML, CNN–BiLSTM, transformers) from
  the outputs directory.
- Concatenate them into a single DataFrame.
- Produce simple summaries, such as the best model per dataset based
  on a chosen metric (e.g., accuracy or macro_f1).

It is meant for lightweight, scriptable analysis. For richer visual
exploration, we can additionally use notebooks under `notebooks/`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from airline_sentiment.utils.config import load_global_config, PROJECT_ROOT


def get_outputs_dir(global_config: Optional[Dict[str, Any]] = None) -> Path:
    """
    Return the root outputs directory as defined in config/config.yaml.
    """
    cfg = global_config or load_global_config()
    outputs_root = cfg.get("paths", {}).get("outputs", "outputs")
    out_dir = PROJECT_ROOT / outputs_root
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_result_files(
    pattern: str = "ml_results_*.csv",
    global_config: Optional[Dict[str, Any]] = None,
) -> List[Path]:
    """
    Find result CSV files matching a glob pattern in the outputs directory.

    Parameters
    ----------
    pattern : str, default "ml_results_*.csv"
        Glob pattern to match CSV result files.
        Examples:
            "ml_results_*.csv"
            "cnn_bilstm_results_*.csv"
            "transformers_results_*.csv"
    global_config : dict, optional
        Pre-loaded global config; if None, loaded automatically.

    Returns
    -------
    List[Path]
        List of matching CSV file paths.
    """
    out_dir = get_outputs_dir(global_config=global_config)
    return sorted(out_dir.glob(pattern))


def load_results_df(
    pattern: str = "ml_results_*.csv",
    global_config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Load and concatenate all result CSVs matching the given pattern.

    Parameters
    ----------
    pattern : str, default "ml_results_*.csv"
        Glob pattern to match CSV result files.
    global_config : dict, optional
        Pre-loaded global config; if None, loaded automatically.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame of results. If no files are found,
        an empty DataFrame is returned.
    """
    files = load_result_files(pattern=pattern, global_config=global_config)
    if not files:
        return pd.DataFrame()

    dfs = []
    for path in files:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def summarize_best_per_dataset(
    results_df: pd.DataFrame,
    metric: str = "accuracy",
    group_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Summarize the best configuration per dataset based on a chosen metric.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame containing experiment results (e.g. from load_results_df).
        Must contain:
            - column "dataset"
            - the specified metric column (e.g. "accuracy" or "macro_f1")
    metric : str, default "accuracy"
        Metric column to use when selecting the best configuration.
    group_cols : list of str, optional
        Additional columns to include in the grouping (e.g. ["feature_type"]
        for ML models, or ["model"] for transformers). If None, only
        "dataset" is used.

    Returns
    -------
    pd.DataFrame
        DataFrame where each row corresponds to a dataset and contains
        the configuration with the highest value of the chosen metric.
    """
    if results_df.empty:
        return results_df

    if "dataset" not in results_df.columns:
        raise KeyError("results_df must contain a 'dataset' column.")

    if metric not in results_df.columns:
        raise KeyError(f"results_df must contain the metric column '{metric}'.")

    if group_cols is None:
        group_cols = []

    # We group by dataset and any additional grouping columns, and
    # within each group we pick the row with the maximum metric.
    grouping = ["dataset"] + group_cols

    # Sort by metric descending so that first row in each group is the best
    df_sorted = results_df.sort_values(by=metric, ascending=False)

    # Drop duplicates per group, keeping the first (best) row
    best_df = df_sorted.drop_duplicates(subset=grouping, keep="first")

    # For clarity, sort by dataset name
    best_df = best_df.sort_values(by=["dataset"] + group_cols).reset_index(drop=True)

    return best_df


def main():
    """
    Example CLI-like entrypoint.

    This function:
    - Loads all ML results from 'ml_results_*.csv'.
    - Prints a summary of the best (model, feature_type) per dataset
      based on macro F1-score.
    """
    results_df = load_results_df(pattern="ml_results_*.csv")
    if results_df.empty:
        print("No ML result files found in outputs/.")
        return

    # For ML, we generally want to consider model + feature_type
    best_df = summarize_best_per_dataset(
        results_df,
        metric="macro_f1",
        group_cols=["model", "feature_type"],
    )

    pd.set_option("display.max_columns", None)
    print("Best ML configurations per dataset (sorted by macro F1):")
    print(best_df)


if __name__ == "__main__":
    main()


__all__ = [
    "get_outputs_dir",
    "load_result_files",
    "load_results_df",
    "summarize_best_per_dataset",
]
