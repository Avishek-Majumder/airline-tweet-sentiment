"""
CNN–BiLSTM model and training pipeline for airline tweet sentiment.

This module implements:
- A simple whitespace-based tokenizer and vocabulary builder.
- A TweetSequenceDataset for (text_clean, label_id) pairs.
- A CNN–BiLSTM architecture initialized with Word2Vec embeddings.
- Training loop with early stopping on validation macro F1.
- Evaluation on the test set and logging results to CSV.

Configuration sources:
- Global config:    config/config.yaml
- CNN–BiLSTM config: config/cnn_bilstm.yaml

Typical CLI usage
-----------------
python -m airline_sentiment.models.dl.cnn_bilstm --dataset airline_us
python -m airline_sentiment.models.dl.cnn_bilstm --dataset airline_global
python -m airline_sentiment.models.dl.cnn_bilstm --dataset airline_merged
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from airline_sentiment.data.dataset_manager import load_all_splits
from airline_sentiment.evaluation.metrics import compute_classification_metrics
from airline_sentiment.features.word2vec_embeddings import (
    Word2VecTrainer,
    build_embedding_matrix,
)
from airline_sentiment.utils.config import (
    load_global_config,
    load_cnn_bilstm_config,
    PROJECT_ROOT,
)
from airline_sentiment.utils.seed import set_global_seed


# ---------------------------------------------------------------------------
# Tokenization / vocabulary
# ---------------------------------------------------------------------------


def build_vocab(
    texts: List[str],
    max_vocab_size: int | None = None,
    min_freq: int = 1,
) -> Dict[str, int]:
    """
    Build a word -> index vocabulary from cleaned texts.

    Parameters
    ----------
    texts : list of str
        Cleaned, whitespace-separated texts.
    max_vocab_size : int, optional
        Maximum vocabulary size (most frequent words kept). If None,
        keep all words.
    min_freq : int, default 1
        Minimum frequency for a word to be included.

    Returns
    -------
    Dict[str, int]
        Mapping from token to integer index, starting at 1.
        Index 0 is reserved for padding.
    """
    counter = Counter()
    for t in texts:
        if not isinstance(t, str):
            t = str(t)
        tokens = t.split()
        counter.update(tokens)

    # Filter by min_freq
    items = [(w, c) for w, c in counter.items() if c >= min_freq]
    # Sort by frequency (desc) then alphabetically
    items.sort(key=lambda x: (-x[1], x[0]))

    if max_vocab_size is not None:
        items = items[:max_vocab_size]

    word_index: Dict[str, int] = {}
    idx = 1  # 0 is reserved for padding
    for w, _ in items:
        word_index[w] = idx
        idx += 1

    return word_index


def encode_text(
    text: str,
    word_index: Dict[str, int],
    max_length: int,
) -> List[int]:
    """
    Convert a cleaned text into a list of token IDs with fixed max_length.

    Parameters
    ----------
    text : str
        Cleaned, whitespace-separated text.
    word_index : Dict[str, int]
        Mapping from word -> index.
    max_length : int
        Maximum sequence length (truncate longer, pad shorter).

    Returns
    -------
    List[int]
        List of token IDs of length `max_length`.
    """
    if not isinstance(text, str):
        text = str(text)

    tokens = text.split()
    ids: List[int] = [word_index.get(tok, 0) for tok in tokens]  # 0 for OOV/padding

    # Truncate
    if len(ids) > max_length:
        ids = ids[:max_length]
    # Pad
    if len(ids) < max_length:
        ids = ids + [0] * (max_length - len(ids))

    return ids


# ---------------------------------------------------------------------------
# Dataset / DataLoader
# ---------------------------------------------------------------------------


class TweetSequenceDataset(Dataset):
    """
    Dataset for sequence-based models (CNN–BiLSTM).

    Uses:
    - text_clean column for input sequences.
    - label_id column for target classes.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        word_index: Dict[str, int],
        max_length: int,
        text_col: str = "text_clean",
        label_col: str = "label_id",
    ) -> None:
        self.texts = df[text_col].astype(str).tolist()
        self.labels = df[label_col].astype(int).tolist()
        self.word_index = word_index
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        ids = encode_text(text, self.word_index, self.max_length)
        x = torch.tensor(ids, dtype=torch.long)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------------
# CNN–BiLSTM model
# ---------------------------------------------------------------------------


class CNNBiLSTM(nn.Module):
    """
    CNN–BiLSTM architecture for sentence-level classification.

    Structure:
    - Embedding layer initialized from Word2Vec embeddings.
    - Multiple 1D convolutional layers (with different kernel sizes)
      + ReLU + max-pooling over time.
    - Concatenated conv features fed into a BiLSTM.
    - Dropout + fully-connected layer to 3-class logits.
    """

    def __init__(
        self,
        embedding_matrix: np.ndarray,
        conv_out_channels: int,
        conv_kernel_sizes: List[int],
        conv_stride: int,
        conv_padding_cfg: str | int,
        bilstm_hidden_size: int,
        bilstm_num_layers: int,
        bilstm_bidirectional: bool,
        dropout: float,
        num_classes: int,
        freeze_embeddings: bool = False,
    ) -> None:
        super().__init__()

        vocab_size, embedding_dim = embedding_matrix.shape
        # embedding_matrix includes padding index 0; we keep that as is
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.from_numpy(embedding_matrix))

        self.embedding.weight.requires_grad = not freeze_embeddings

        self.conv_kernel_sizes = conv_kernel_sizes
        self.convs = nn.ModuleList()
        for k in conv_kernel_sizes:
            padding = self._resolve_padding(k, conv_padding_cfg)
            conv = nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=conv_out_channels,
                kernel_size=k,
                stride=conv_stride,
                padding=padding,
            )
            self.convs.append(conv)

        self.activation = nn.ReLU()

        lstm_input_dim = conv_out_channels * len(conv_kernel_sizes)
        self.bilstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=bilstm_hidden_size,
            num_layers=bilstm_num_layers,
            batch_first=True,
            bidirectional=bilstm_bidirectional,
        )

        lstm_output_dim = bilstm_hidden_size * (2 if bilstm_bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)

    @staticmethod
    def _resolve_padding(kernel_size: int, padding_cfg: str | int) -> int:
        """
        Convert padding config to an integer padding value.

        If padding_cfg == "same", we approximate same-padding for 1D
        convolution as kernel_size // 2. Otherwise treat as an integer.
        """
        if isinstance(padding_cfg, str):
            if padding_cfg.lower() == "same":
                return kernel_size // 2
            raise ValueError(f"Unsupported padding config: {padding_cfg}")
        return int(padding_cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len) with token IDs.

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes).
        """
        # x: (B, L)
        emb = self.embedding(x)  # (B, L, D)
        emb = emb.transpose(1, 2)  # (B, D, L) for Conv1d

        conv_outputs = []
        for conv in self.convs:
            c = conv(emb)  # (B, C, L')
            c = self.activation(c)
            # Max pooling over time
            c = torch.max(c, dim=2).values  # (B, C)
            conv_outputs.append(c)

        # Concatenate conv outputs
        conv_cat = torch.cat(conv_outputs, dim=1)  # (B, C * num_kernels)

        # Expand to sequence of length 1 for LSTM
        lstm_input = conv_cat.unsqueeze(1)  # (B, 1, C * num_kernels)

        lstm_out, _ = self.bilstm(lstm_input)  # (B, 1, H) or (B, 1, 2H)
        lstm_out = lstm_out.squeeze(1)  # (B, H*)

        out = self.dropout(lstm_out)
        logits = self.fc(out)  # (B, num_classes)
        return logits


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train model for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    total_batches = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(total_batches, 1)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate model on a dataloader and return metrics dictionary."""
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(x_batch)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y_batch.cpu().numpy().tolist())

    # We'll map label IDs to strings outside (using global config)
    # and only pass numeric IDs here
    metrics = compute_classification_metrics(all_labels, all_preds)
    return metrics


def train_and_evaluate_cnn_bilstm(dataset_name: str) -> Dict[str, Any]:
    """
    Full pipeline to train and evaluate CNN–BiLSTM on a given dataset.

    Steps:
    1. Load configs, set global seed, choose device.
    2. Load train/val/test splits (with optional augmentation for train).
    3. Train or load Word2Vec model using configured datasets.
    4. Build vocabulary from current dataset's training texts.
    5. Build embedding matrix from Word2Vec + vocab.
    6. Construct CNN–BiLSTM model, dataloaders, optimizer, etc.
    7. Train with early stopping on val macro F1.
    8. Evaluate on test set, save metrics to CSV, and return them.
    """
    global_cfg = load_global_config()
    cnn_cfg = load_cnn_bilstm_config().get("cnn_bilstm", {})

    # Seed and device
    seed = set_global_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 1. Load dataset splits
    # ------------------------------------------------------------------
    training_cfg = cnn_cfg.get("training", {})
    augment_train = bool(training_cfg.get("augment_train", True))

    splits = load_all_splits(
        dataset_name,
        apply_augmentation_to_train=augment_train,
        config=global_cfg,
    )
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    max_length = int(cnn_cfg.get("max_length", global_cfg.get("preprocessing", {}).get("max_length", 60)))
    batch_size = int(cnn_cfg.get("batch_size", 64))
    num_workers = int(cnn_cfg.get("num_workers", 4))

    # ------------------------------------------------------------------
    # 2. Train or load Word2Vec model over configured datasets
    # ------------------------------------------------------------------
    w2v_trainer = Word2VecTrainer(cnn_config=cnn_cfg, global_config=global_cfg)
    w2v_cfg = cnn_cfg.get("word2vec", {})
    train_on = w2v_cfg.get("train_on", [dataset_name])

    # Build combined corpus from specified datasets' training splits
    corpus_texts: List[str] = []
    from airline_sentiment.data.dataset_manager import load_processed_split

    for ds_name in train_on:
        ds_splits = load_processed_split(
            ds_name,
            "train",
            apply_augmentation=True,
            config=global_cfg,
        )
        corpus_texts.extend(ds_splits["text_clean"].astype(str).tolist())

    corpus = w2v_trainer.build_corpus(corpus_texts)

    # We always (re)train the Word2Vec model in this pipeline
    w2v_model = w2v_trainer.train(corpus)
    w2v_trainer.save_model(w2v_model)

    # ------------------------------------------------------------------
    # 3. Build vocabulary from this dataset's training texts
    # ------------------------------------------------------------------
    train_texts = train_df["text_clean"].astype(str).tolist()
    word_index = build_vocab(train_texts, max_vocab_size=None, min_freq=1)

    # ------------------------------------------------------------------
    # 4. Build embedding matrix
    # ------------------------------------------------------------------
    embedding_dim = int(cnn_cfg.get("model", {}).get("embedding_dim", 300))
    embedding_matrix = build_embedding_matrix(
        w2v_model,
        word_index=word_index,
        embedding_dim=embedding_dim,
    )

    # ------------------------------------------------------------------
    # 5. Create datasets and dataloaders
    # ------------------------------------------------------------------
    train_dataset = TweetSequenceDataset(
        train_df,
        word_index=word_index,
        max_length=max_length,
    )
    val_dataset = TweetSequenceDataset(
        val_df,
        word_index=word_index,
        max_length=max_length,
    )
    test_dataset = TweetSequenceDataset(
        test_df,
        word_index=word_index,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # ------------------------------------------------------------------
    # 6. Build CNN–BiLSTM model
    # ------------------------------------------------------------------
    model_cfg = cnn_cfg.get("model", {})
    model = CNNBiLSTM(
        embedding_matrix=embedding_matrix,
        conv_out_channels=int(model_cfg.get("conv_out_channels", 100)),
        conv_kernel_sizes=list(model_cfg.get("conv_kernel_sizes", [3, 4, 5])),
        conv_stride=int(model_cfg.get("conv_stride", 1)),
        conv_padding_cfg=model_cfg.get("conv_padding", "same"),
        bilstm_hidden_size=int(model_cfg.get("bilstm_hidden_size", 128)),
        bilstm_num_layers=int(model_cfg.get("bilstm_num_layers", 1)),
        bilstm_bidirectional=bool(model_cfg.get("bilstm_bidirectional", True)),
        dropout=float(model_cfg.get("dropout", 0.5)),
        num_classes=int(model_cfg.get("num_classes", 3)),
        freeze_embeddings=bool(model_cfg.get("freeze_embeddings", False)),
    )
    model.to(device)

    # ------------------------------------------------------------------
    # 7. Training setup
    # ------------------------------------------------------------------
    training_cfg = cnn_cfg.get("training", {})
    epochs = int(training_cfg.get("epochs", 20))
    lr = float(training_cfg.get("learning_rate", 0.001))
    weight_decay = float(training_cfg.get("weight_decay", 0.0001))
    grad_clip = float(training_cfg.get("grad_clip", 5.0))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    early_cfg = training_cfg.get("early_stopping", {})
    es_enabled = bool(early_cfg.get("enabled", True))
    es_patience = int(early_cfg.get("patience", 3))

    # For logging / checkpoints
    logging_cfg = cnn_cfg.get("logging", {})
    log_interval = int(logging_cfg.get("log_interval", 50))
    save_best_only = bool(logging_cfg.get("save_best_only", True))
    metric_for_best = logging_cfg.get("metric_for_best", "val_macro_f1")

    checkpoints_dir = PROJECT_ROOT / logging_cfg.get("save_dir", "checkpoints/cnn_bilstm")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / f"{dataset_name}_best.pt"

    best_val_metric = -np.inf
    best_epoch = -1
    epochs_without_improvement = 0
    best_test_metrics: Dict[str, Any] = {}

    # Map numeric labels to strings for final logging
    label_mapping = global_cfg.get("labels", {}).get("mapping", {})
    id_to_label = {int(v): k for k, v in label_mapping.items()}

    # ------------------------------------------------------------------
    # 8. Training loop
    # ------------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Evaluate on validation and test after each epoch
        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)

        # Attach readable label-based keys
        val_metrics_named = val_metrics.copy()
        test_metrics_named = test_metrics.copy()

        # Determine metric for early stopping
        val_metric_value = val_metrics_named.get("macro_f1", 0.0)
        print(
            f"[CNN-BiLSTM][{dataset_name}] Epoch {epoch}/{epochs} "
            f"- loss: {avg_loss:.4f}, val_macro_f1: {val_metric_value:.4f}"
        )

        is_best = False
        if val_metric_value > best_val_metric:
            best_val_metric = val_metric_value
            best_epoch = epoch
            best_test_metrics = test_metrics_named
            epochs_without_improvement = 0
            is_best = True

            if save_best_only:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_metrics": val_metrics_named,
                        "test_metrics": test_metrics_named,
                        "word_index": word_index,
                        "seed": seed,
                    },
                    checkpoint_path,
                )
        else:
            epochs_without_improvement += 1

        # Manual checkpointing if not best-only
        if not save_best_only:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics_named,
                    "test_metrics": test_metrics_named,
                    "word_index": word_index,
                    "seed": seed,
                },
                checkpoints_dir / f"{dataset_name}_epoch{epoch}.pt",
            )

        # Early stopping
        if es_enabled and epochs_without_improvement >= es_patience:
            print(
                f"Early stopping triggered after {epochs_without_improvement} "
                f"epochs without improvement (best epoch: {best_epoch})."
            )
            break

    # ------------------------------------------------------------------
    # 9. Save final test metrics to CSV
    # ------------------------------------------------------------------
    outputs_root = global_cfg.get("paths", {}).get("outputs", "outputs")
    out_dir = PROJECT_ROOT / outputs_root
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cnn_bilstm_results_{dataset_name}.csv"

    # Add metadata fields
    record = {
        "dataset": dataset_name,
        "model": "cnn_bilstm",
        "feature_type": "word2vec",
        "best_epoch": best_epoch,
    }
    record.update(best_test_metrics)

    results_df = pd.DataFrame([record])
    results_df.to_csv(out_path, index=False)

    print(f"Saved CNN–BiLSTM test results for {dataset_name} to: {out_path}")
    print(f"Best validation macro F1: {best_val_metric:.4f} (epoch {best_epoch})")

    return record


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate CNN–BiLSTM for airline tweet sentiment."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="airline_us",
        choices=["airline_us", "airline_global", "airline_merged"],
        help="Dataset name to train on.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_evaluate_cnn_bilstm(dataset_name=args.dataset)


if __name__ == "__main__":
    main()
