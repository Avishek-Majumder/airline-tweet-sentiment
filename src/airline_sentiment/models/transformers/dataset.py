"""
Datasets for transformer-based airline tweet sentiment experiments.

This module provides utilities to:
- Load processed train/val/test splits via `dataset_manager`.
- Tokenize texts using a HuggingFace tokenizer.
- Wrap tokenized inputs and labels into PyTorch Datasets suitable for
  training transformer models (BERT, RoBERTa, ELECTRA, ALBERT, etc.).

Configuration sources:
- Global config:        config/config.yaml
- Transformers config:  config/transformers.yaml
"""

from __future__ import annotations

from typing import Dict, Any, List

import torch
from torch.utils.data import Dataset

from airline_sentiment.data.dataset_manager import load_all_splits
from airline_sentiment.utils.config import load_global_config, load_transformers_config
from airline_sentiment.utils.seed import set_global_seed


class TweetTransformersDataset(Dataset):
    """
    PyTorch Dataset for transformer-based tweet sentiment classification.

    Holds:
    - tokenized inputs (input_ids, attention_mask, etc.)
    - integer sentiment labels (label_id)
    """

    def __init__(
        self,
        encodings: Dict[str, torch.Tensor],
        labels: List[int],
    ) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _tokenize_texts(
    texts: List[str],
    tokenizer,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """
    Tokenize a list of texts using a HuggingFace tokenizer.

    Parameters
    ----------
    texts : list of str
        Cleaned tweet texts.
    tokenizer :
        Any HuggingFace tokenizer with a `__call__` method.
    max_length : int
        Maximum sequence length.

    Returns
    -------
    Dict[str, torch.Tensor]
        Dictionary of tokenized tensors (input_ids, attention_mask, etc.).
    """
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {k: v for k, v in enc.items()}


def create_transformers_datasets(
    dataset_name: str,
    tokenizer,
    transformers_config: Dict[str, Any] | None = None,
    global_config: Dict[str, Any] | None = None,
    apply_augmentation_to_train: bool = True,
) -> Dict[str, TweetTransformersDataset]:
    """
    Create train/val/test TweetTransformersDataset objects for a dataset.

    Steps:
    1. Load processed train/val/test splits via dataset_manager.
    2. Tokenize text_clean using the provided tokenizer.
    3. Wrap tokenized inputs and label_id into TweetTransformersDataset.

    Parameters
    ----------
    dataset_name : str
        One of "airline_us", "airline_global", "airline_merged".
    tokenizer :
        HuggingFace tokenizer instance (e.g. AutoTokenizer.from_pretrained(...)).
    transformers_config : dict, optional
        Configuration loaded from `transformers.yaml`. If None, it will
        be loaded automatically.
    global_config : dict, optional
        Global configuration loaded from `config.yaml`. If None, it will
        be loaded automatically.
    apply_augmentation_to_train : bool, default True
        Whether to apply augmentation-based balancing to the training
        split when loading from dataset_manager.

    Returns
    -------
    Dict[str, TweetTransformersDataset]
        Dictionary with keys "train", "val", "test".
    """
    set_global_seed()

    gcfg = global_config or load_global_config()
    tcfg_all = transformers_config or load_transformers_config()
    tcfg = tcfg_all.get("transformers", {})

    max_length = int(tcfg.get("common", {}).get("max_length", 64))

    # Load splits (with optional augmentation on train)
    splits = load_all_splits(
        dataset_name,
        apply_augmentation_to_train=apply_augmentation_to_train,
        config=gcfg,
    )
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    # Convert to lists
    train_texts = train_df["text_clean"].astype(str).tolist()
    val_texts = val_df["text_clean"].astype(str).tolist()
    test_texts = test_df["text_clean"].astype(str).tolist()

    train_labels = train_df["label_id"].astype(int).tolist()
    val_labels = val_df["label_id"].astype(int).tolist()
    test_labels = test_df["label_id"].astype(int).tolist()

    # Tokenize
    train_enc = _tokenize_texts(train_texts, tokenizer, max_length=max_length)
    val_enc = _tokenize_texts(val_texts, tokenizer, max_length=max_length)
    test_enc = _tokenize_texts(test_texts, tokenizer, max_length=max_length)

    # Wrap into datasets
    train_dataset = TweetTransformersDataset(train_enc, train_labels)
    val_dataset = TweetTransformersDataset(val_enc, val_labels)
    test_dataset = TweetTransformersDataset(test_enc, test_labels)

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }


__all__ = ["TweetTransformersDataset", "create_transformers_datasets"]
