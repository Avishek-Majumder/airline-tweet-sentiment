"""
Random seed utilities for reproducible experiments.

We centralize all seeding logic here so that:
- Classical ML experiments
- CNN–BiLSTM training
- Transformer fine-tuning

can all call the same function and behave consistently, using the
`project.random_seed` value from `config/config.yaml` by default.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for some environments
    torch = None  # type: ignore[assignment]

from airline_sentiment.utils.config import load_global_config


def set_global_seed(seed: Optional[int] = None) -> int:
    """
    Set global random seeds for Python, NumPy, and (if available) PyTorch.

    Parameters
    ----------
    seed : int, optional
        Desired seed. If None, the value is read from
        `project.random_seed` in `config/config.yaml`. If that is also
        missing, a default of 42 is used.

    Returns
    -------
    int
        The seed value that was ultimately used.
    """
    if seed is None:
        cfg = load_global_config()
        seed = int(cfg.get("project", {}).get("random_seed", 42))

    # Python's built-in RNG
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Environment variable commonly used for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch (if installed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
        # For extra determinism (may affect performance)
        if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]

    return seed


__all__ = ["set_global_seed"]
