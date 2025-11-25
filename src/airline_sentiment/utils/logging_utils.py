"""
Logging utilities for the airline_tweet_sentiment project.

We use a simple wrapper around Python's built-in logging module to:

- Create a named logger with a consistent format.
- Optionally log to both console (stdout) and a file.
- Control the base log directory via config/config.yaml (`paths.logs`).

Typical usage
-------------
from airline_sentiment.utils.logging_utils import get_logger

logger = get_logger(__name__)
logger.info("Starting experiment...")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Dict, Any

from airline_sentiment.utils.config import load_global_config, PROJECT_ROOT


_LOGGERS: Dict[str, logging.Logger] = {}


def _get_logs_dir(global_config: Optional[Dict[str, Any]] = None) -> Path:
    """
    Resolve the logs directory from global config.

    Defaults to 'runs/' at the project root.
    """
    cfg = global_config or load_global_config()
    logs_root = cfg.get("paths", {}).get("logs", "runs")
    logs_dir = PROJECT_ROOT / logs_root
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def get_logger(
    name: str = "airline_sentiment",
    log_to_file: bool = False,
    filename: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Get a configured logger instance.

    Parameters
    ----------
    name : str, default "airline_sentiment"
        Logger name (typically __name__ of the caller).
    log_to_file : bool, default False
        If True, also log to a file in the logs directory.
    filename : str, optional
        Log filename (without path). If None and log_to_file=True,
        a default name 'airline_sentiment.log' is used.
    level : int, default logging.INFO
        Logging level.

    Returns
    -------
    logging.Logger
        Configured logger.
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # avoid duplicate logs if root logger is configured

    # Formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_to_file:
        logs_dir = _get_logs_dir()
        log_filename = filename or "airline_sentiment.log"
        file_path = logs_dir / log_filename

        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger


__all__ = ["get_logger"]
