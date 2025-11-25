"""
Word2Vec embedding utilities for the CNN–BiLSTM model.

This module provides:

- A `Word2VecTrainer` class to:
  - Build a tokenized corpus from cleaned texts.
  - Train a Word2Vec model using gensim, with hyperparameters taken
    from `config/cnn_bilstm.yaml`.
  - Save/load the trained Word2Vec model to/from disk.

- A `build_embedding_matrix` helper to construct a numpy embedding
  matrix given:
  - a gensim Word2Vec model, and
  - a word_index mapping (word -> integer ID) used by the CNN–BiLSTM
    tokenizer.

Typical usage
-------------
from airline_sentiment.features.word2vec_embeddings import (
    Word2VecTrainer,
    build_embedding_matrix,
)

trainer = Word2VecTrainer()
sentences = trainer.build_corpus(df_train["text_clean"])
w2v_model = trainer.train(sentences)
trainer.save_model(w2v_model)

embedding_matrix = build_embedding_matrix(
    w2v_model,
    word_index=tokenizer.word_index,
    embedding_dim=trainer.embedding_dim,
)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Dict, Any

import numpy as np
from gensim.models import Word2Vec

from airline_sentiment.utils.config import (
    load_cnn_bilstm_config,
    load_global_config,
    PROJECT_ROOT,
)


class Word2VecTrainer:
    """
    Train and manage a Word2Vec model for tweet sentiment embeddings.

    Hyperparameters are taken from `config/cnn_bilstm.yaml` under
    `cnn_bilstm.word2vec`.

    Attributes
    ----------
    embedding_dim : int
        Dimensionality of the embedding vectors.
    window : int
        Window size for Word2Vec.
    min_count : int
        Minimum frequency for a word to be included in the vocabulary.
    workers : int
        Number of worker threads.
    sg : int
        Training algorithm: 1 for skip-gram, 0 for CBOW.
    epochs : int
        Number of training epochs.
    """

    def __init__(
        self,
        cnn_config: Dict[str, Any] | None = None,
        global_config: Dict[str, Any] | None = None,
    ) -> None:
        self.cnn_cfg = cnn_config or load_cnn_bilstm_config()
        self.global_cfg = global_config or load_global_config()

        w2v_cfg = self.cnn_cfg.get("cnn_bilstm", {}).get("word2vec", {})

        self.embedding_dim: int = int(w2v_cfg.get("embedding_dim", 300))
        self.window: int = int(w2v_cfg.get("window", 5))
        self.min_count: int = int(w2v_cfg.get("min_count", 2))
        self.workers: int = int(w2v_cfg.get("workers", 4))
        self.sg: int = int(w2v_cfg.get("sg", 1))
        self.epochs: int = int(w2v_cfg.get("epochs", 10))

        project_cfg = self.global_cfg.get("project", {})
        self.random_seed: int = int(project_cfg.get("random_seed", 42))

        # Where to store the Word2Vec model by default
        models_root = self.global_cfg.get("paths", {}).get("models", "models")
        self.model_dir: Path = PROJECT_ROOT / models_root / "word2vec"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.default_model_path: Path = self.model_dir / "word2vec.model"

    # ------------------------------------------------------------------
    # Corpus building
    # ------------------------------------------------------------------

    @staticmethod
    def build_corpus(texts: Iterable[str]) -> List[List[str]]:
        """
        Convert cleaned texts into a list of token lists for Word2Vec.

        Parameters
        ----------
        texts : Iterable[str]
            Sequence of cleaned, whitespace-separated texts.

        Returns
        -------
        List[List[str]]
            List of tokenized sentences, each sentence a list of tokens.
        """
        corpus: List[List[str]] = []
        for t in texts:
            if not isinstance(t, str):
                t = str(t)
            tokens = t.split()
            if tokens:
                corpus.append(tokens)
        return corpus

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, corpus: List[List[str]]) -> Word2Vec:
        """
        Train a Word2Vec model on the given corpus.

        Parameters
        ----------
        corpus : List[List[str]]
            Tokenized sentences (list of list of tokens).

        Returns
        -------
        Word2Vec
            Trained gensim Word2Vec model.
        """
        model = Word2Vec(
            sentences=corpus,
            vector_size=self.embedding_dim,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            seed=self.random_seed,
        )
        model.train(corpus, total_examples=len(corpus), epochs=self.epochs)
        return model

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_model(self, model: Word2Vec, path: str | Path | None = None) -> Path:
        """
        Save a Word2Vec model to disk.

        Parameters
        ----------
        model : Word2Vec
            Trained Word2Vec model.
        path : str or Path, optional
            Destination path. If None, use the default model path.

        Returns
        -------
        Path
            Path where the model was saved.
        """
        save_path = Path(path) if path is not None else self.default_model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        return save_path

    def load_model(self, path: str | Path | None = None) -> Word2Vec:
        """
        Load a Word2Vec model from disk.

        Parameters
        ----------
        path : str or Path, optional
            Source path. If None, load from the default model path.

        Returns
        -------
        Word2Vec
            Loaded Word2Vec model.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        """
        load_path = Path(path) if path is not None else self.default_model_path
        if not load_path.is_file():
            raise FileNotFoundError(
                f"Word2Vec model file not found at: {load_path}. "
                "Train and save a model first."
            )
        return Word2Vec.load(str(load_path))


# ----------------------------------------------------------------------
# Embedding matrix builder
# ----------------------------------------------------------------------


def build_embedding_matrix(
    model: Word2Vec,
    word_index: Dict[str, int],
    embedding_dim: int,
) -> np.ndarray:
    """
    Build an embedding matrix from a Word2Vec model and a word_index.

    Parameters
    ----------
    model : Word2Vec
        Trained gensim Word2Vec model.
    word_index : Dict[str, int]
        Mapping from token (string) to integer index used by the
        tokenizer feeding the CNN–BiLSTM model.
    embedding_dim : int
        Dimensionality of the embeddings (should match model.vector_size).

    Returns
    -------
    np.ndarray
        Embedding matrix of shape (vocab_size + 1, embedding_dim).
        Row `i` contains the embedding for the word whose index is `i`.
        Row 0 is reserved for the padding token and initialized with zeros.
        Words not found in the Word2Vec vocabulary are initialized with
        a small random vector.
    """
    vocab_size = len(word_index)
    embedding_matrix = np.zeros((vocab_size + 1, embedding_dim), dtype=np.float32)

    # Precompute random range for OOV words
    oov_limit = 0.25
    for word, idx in word_index.items():
        if idx == 0:
            # Reserve index 0 for padding
            continue

        if word in model.wv:
            embedding_matrix[idx] = model.wv[word]
        else:
            # Initialize out-of-vocabulary words with small random vectors
            embedding_matrix[idx] = np.random.uniform(
                low=-oov_limit,
                high=oov_limit,
                size=(embedding_dim,),
            ).astype(np.float32)

    return embedding_matrix


__all__ = ["Word2VecTrainer", "build_embedding_matrix"]
