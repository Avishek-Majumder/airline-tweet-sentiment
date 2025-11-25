"""
Text preprocessing, splitting, and augmentation utilities.

This subpackage implements the full preprocessing pipeline used in our
airline tweet sentiment experiments:

- Text normalization and cleaning (URLs, mentions, hashtags, emojis,
  punctuation, stopwords, light stemming).
- Stratified train/val/test splitting and saving of processed CSVs.
- EDA-style augmentation and class balancing for the training split.

Main entrypoints
----------------
- `TextCleaner` (from `text_cleaning.py`):
    Configurable text cleaning class that transforms raw tweet text into
    `text_clean`, used by all downstream models.

- `prepare_and_split_dataset` / `prepare_all_datasets` (from `splitter.py`):
    High-level helpers that:
      * load raw CSVs via `airline_sentiment.data.datasets`
      * clean text
      * create stratified train/val/test splits
      * save them under `data/processed/<dataset_name>/`.

- `augment_text` / `balance_classes_with_augmentation` (from `augmentation.py`):
    Perform EDA-style text augmentation and use it to over-sample
    minority classes for more balanced training data.
"""

from __future__ import annotations

from .text_cleaning import TextCleaner
from .splitter import (
    prepare_and_split_dataset,
    prepare_all_datasets,
)
from .augmentation import (
    augment_text,
    balance_classes_with_augmentation,
)

__all__ = [
    "TextCleaner",
    "prepare_and_split_dataset",
    "prepare_all_datasets",
    "augment_text",
    "balance_classes_with_augmentation",
]
