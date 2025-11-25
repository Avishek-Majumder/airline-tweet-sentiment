"""
Transformer-based models for airline tweet sentiment.

This subpackage integrates pre-trained transformer models from the
HuggingFace ecosystem (e.g., BERT, ELECTRA, ALBERT, RoBERTa, DistilBERT)
and fine-tunes them for tweet-level sentiment classification.

Components
----------

1. Dataset utilities (`dataset.py`)
   - `TweetTransformersDataset`:
       Lightweight PyTorch Dataset that wraps tokenized inputs and
       integer sentiment labels.
   - `create_transformers_datasets(dataset_name, tokenizer, ...)`:
       High-level factory that:
         * loads processed train/val/test splits via
           `airline_sentiment.data.dataset_manager`
         * tokenizes the `text_clean` column with a HuggingFace tokenizer
         * returns a dict with `"train"`, `"val"`, `"test"` datasets.

2. Training & evaluation (`trainer.py`)
   - `run_transformer_experiments_for_dataset(dataset_name, single_model_key=None, save_results=True)`:
       Orchestrates full experiments for one dataset and:
         * reads model configs from `config/transformers.yaml`
         * builds `AutoTokenizer` and `AutoModelForSequenceClassification`
         * uses HuggingFace `Trainer` for fine-tuning
         * evaluates on the test set
         * writes results to `outputs/transformers_results_<dataset>.csv`.

   - Internally, `_run_single_transformer_experiment(...)` handles the
     full pipeline for one specific `model_key` defined in the YAML
     configuration.

These functions are exposed via the CLI script
`scripts/run_transformers.py`, which is the main entrypoint used to
reproduce the transformer experiments in our paper.
"""

from __future__ import annotations

from .dataset import TweetTransformersDataset, create_transformers_datasets
from .trainer import run_transformer_experiments_for_dataset

__all__ = [
    "TweetTransformersDataset",
    "create_transformers_datasets",
    "run_transformer_experiments_for_dataset",
]
