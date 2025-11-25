"""
Utility helpers for configuration, seeding, and logging.

This subpackage centralizes common infrastructure used throughout the
airline tweet sentiment codebase:

- Configuration loading:
    Read YAML files from the `config/` directory and expose a simple
    API for accessing global settings, model hyperparameters, and paths.

- Random seeding:
    Provide a single function to set seeds for Python, NumPy, and
    PyTorch so that experiments remain as reproducible as possible.

- Logging:
    Create consistent loggers for scripts and modules, with optional
    file logging controlled via `config/config.yaml`.

Main entrypoints
----------------
- `load_global_config`, `load_ml_config`, `load_cnn_bilstm_config`,
  `load_transformers_config`, `PROJECT_ROOT` (from `config.py`)

- `set_global_seed` (from `seed.py`)

- `get_logger` (from `logging_utils.py`)
"""

from __future__ import annotations

from .config import (
    PROJECT_ROOT,
    load_global_config,
    load_ml_config,
    load_cnn_bilstm_config,
    load_transformers_config,
)
from .seed import set_global_seed
from .logging_utils import get_logger

__all__ = [
    "PROJECT_ROOT",
    "load_global_config",
    "load_ml_config",
    "load_cnn_bilstm_config",
    "load_transformers_config",
    "set_global_seed",
    "get_logger",
]
