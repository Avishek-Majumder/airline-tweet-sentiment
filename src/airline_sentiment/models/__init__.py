"""
Model implementations for airline tweet sentiment experiments.

This subpackage groups all model families used in our work:

1. Classical machine learning models (under `ml/`)
   - Logistic Regression
   - k-Nearest Neighbors
   - Support Vector Machine (SVM)
   - Decision Tree
   - Random Forest
   - AdaBoost

   These models operate on TF–IDF or Bag-of-Words features and are
   orchestrated by the `experiment_runner` module.

2. Deep learning model (under `dl/`)
   - CNN–BiLSTM with Word2Vec embeddings

   This model consumes integer-encoded sequences backed by a Word2Vec
   embedding matrix and is trained end-to-end on tweet sentiment labels.

3. Transformer-based models (under `transformers/`)
   - Pre-trained language models from HuggingFace (e.g., BERT, ELECTRA,
     ALBERT, RoBERTa, DistilBERT) fine-tuned for tweet-level sentiment
     classification.

Each subpackage provides its own training and evaluation entrypoints
that are wired into the top-level scripts in `scripts/`.
"""

from __future__ import annotations

# Classical ML
from .ml.experiment_runner import run_ml_experiments_for_dataset

# Deep learning (CNN–BiLSTM)
from .dl.cnn_bilstm import train_and_evaluate_cnn_bilstm

# Transformers
from .transformers.trainer import run_transformer_experiments_for_dataset

__all__ = [
    "run_ml_experiments_for_dataset",
    "train_and_evaluate_cnn_bilstm",
    "run_transformer_experiments_for_dataset",
]
