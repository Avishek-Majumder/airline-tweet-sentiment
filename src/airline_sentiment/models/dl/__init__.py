"""
Deep learning models for airline tweet sentiment.

This subpackage currently provides the CNN–BiLSTM architecture that we
use as a deep learning baseline alongside transformer models.

Model
-----
- CNN–BiLSTM (`cnn_bilstm.py`):
    - Embedding layer initialized with Word2Vec vectors.
    - Multiple 1D convolution filters to capture local n-gram patterns.
    - BiLSTM layer to model longer-range sequential dependencies.
    - Dropout + fully connected layer for 3-way sentiment classification.

Pipeline
--------
The main entrypoint is:

    train_and_evaluate_cnn_bilstm(dataset_name: str) -> dict

It performs the following steps:

1. Load processed train/val/test splits using
   `airline_sentiment.data.dataset_manager`.
2. Train (or retrain) a Word2Vec model on cleaned tweets and build an
   embedding matrix compatible with the vocabulary.
3. Construct the CNN–BiLSTM model using hyperparameters from
   `config/cnn_bilstm.yaml`.
4. Train the model with early stopping based on validation macro F1.
5. Evaluate the best model on the test set.
6. Save a single-row CSV with metrics to
   `outputs/cnn_bilstm_results_<dataset>.csv`.

This function is wired into the `scripts/run_cnn_bilstm.py` CLI script.
"""

from __future__ import annotations

from .cnn_bilstm import (
    CNNBiLSTM,
    train_and_evaluate_cnn_bilstm,
)

__all__ = [
    "CNNBiLSTM",
    "train_and_evaluate_cnn_bilstm",
]
