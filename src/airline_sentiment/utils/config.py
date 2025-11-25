"""
Config loading utilities for the airline_tweet_sentiment project.

This module provides simple helpers to load YAML configuration files
for:
- Global project & dataset settings (config.yaml)
- Classical ML experiments (ml.yaml)
- CNN–BiLSTM model (cnn_bilstm.yaml)
- Transformer models (transformers.yaml)

All paths are resolved relative to the project root, assuming the
following structure:

airline-tweet-sentiment/
├── config/
│   ├── config.yaml
│   ├── ml.yaml
│   ├── cnn_bilstm.yaml
│   └── transformers.yaml
└── src/
    └── airline_sentiment/
        └── utils/
            └── config.py  <-- this file
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml

# Resolve project root:
# config.py -> utils -> airline_sentiment -> src -> PROJECT_ROOT
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]

CONFIG_DIR: Path = PROJECT_ROOT / "config"

GLOBAL_CONFIG_PATH: Path = CONFIG_DIR / "config.yaml"
ML_CONFIG_PATH: Path = CONFIG_DIR / "ml.yaml"
CNN_BILSTM_CONFIG_PATH: Path = CONFIG_DIR / "cnn_bilstm.yaml"
TRANSFORMERS_CONFIG_PATH: Path = CONFIG_DIR / "transformers.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.

    Parameters
    ----------
    path : Path
        Path to the YAML file.

    Returns
    -------
    Dict[str, Any]
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    yaml.YAMLError
        If the YAML content is invalid.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a YAML mapping at the top level.")

    return data


def load_global_config(override_path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """
    Load the global project configuration (config.yaml).

    Parameters
    ----------
    override_path : str or Path, optional
        Optional custom path to a config file. If provided, this path
        will be used instead of the default GLOBAL_CONFIG_PATH.

    Returns
    -------
    Dict[str, Any]
        Global configuration dictionary.
    """
    path = Path(override_path) if override_path is not None else GLOBAL_CONFIG_PATH
    return _load_yaml(path)


def load_ml_config(override_path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """
    Load the classical ML experiments configuration (ml.yaml).
    """
    path = Path(override_path) if override_path is not None else ML_CONFIG_PATH
    return _load_yaml(path)


def load_cnn_bilstm_config(override_path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """
    Load the CNN–BiLSTM configuration (cnn_bilstm.yaml).
    """
    path = Path(override_path) if override_path is not None else CNN_BILSTM_CONFIG_PATH
    return _load_yaml(path)


def load_transformers_config(override_path: str | os.PathLike | None = None) -> Dict[str, Any]:
    """
    Load the transformers configuration (transformers.yaml).
    """
    path = Path(override_path) if override_path is not None else TRANSFORMERS_CONFIG_PATH
    return _load_yaml(path)


__all__ = [
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "GLOBAL_CONFIG_PATH",
    "ML_CONFIG_PATH",
    "CNN_BILSTM_CONFIG_PATH",
    "TRANSFORMERS_CONFIG_PATH",
    "load_global_config",
    "load_ml_config",
    "load_cnn_bilstm_config",
    "load_transformers_config",
]
