"""
Feature extraction utilities for airline tweet sentiment experiments.

This subpackage provides three main kinds of features:

1. Classical sparse text features
   - TF–IDF (term frequency–inverse document frequency)
   - Bag-of-Words (raw counts)

   These features are used by our classical ML models:
   Logistic Regression, KNN, SVM, Decision Tree, Random Forest, AdaBoost.

2. Dense Word2Vec embeddings
   - Trains a Word2Vec model over cleaned tweets.
   - Builds an embedding matrix compatible with the CNN–BiLSTM model.

Modules
-------
- `tfidf_vectorizer.py`:
    `TfidfFeatureExtractor` – wraps sklearn's TfidfVectorizer with
    configuration taken from `config/ml.yaml`.

- `bow_vectorizer.py`:
    `BowFeatureExtractor` – wraps sklearn's CountVectorizer with
    configuration from `config/ml.yaml`.

- `word2vec_embeddings.py`:
    `Word2VecTrainer` – trains and persists a gensim Word2Vec model.
    `build_embedding_matrix` – converts a tokenizer vocabulary into a
    dense embedding matrix using the trained Word2Vec vectors.
"""

from __future__ import annotations

from .tfidf_vectorizer import TfidfFeatureExtractor
from .bow_vectorizer import BowFeatureExtractor
from .word2vec_embeddings import Word2VecTrainer, build_embedding_matrix

__all__ = [
    "TfidfFeatureExtractor",
    "BowFeatureExtractor",
    "Word2VecTrainer",
    "build_embedding_matrix",
]
