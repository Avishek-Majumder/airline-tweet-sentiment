"""
TF–IDF feature extraction for airline tweet sentiment experiments.

This module provides a thin wrapper around scikit-learn's
`TfidfVectorizer`, configured via `config/ml.yaml`, so that all
classical ML models share the same TF–IDF settings.

Typical usage:
--------------
from airline_sentiment.features.tfidf_vectorizer import TfidfFeatureExtractor

extractor = TfidfFeatureExtractor()
X_train = extractor.fit_transform(df_train["text_clean"])
X_val   = extractor.transform(df_val["text_clean"])
X_test  = extractor.transform(df_test["text_clean"])
"""

from __future__ import annotations

from typing import Iterable, Optional, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer

from airline_sentiment.utils.config import load_ml_config


class TfidfFeatureExtractor:
    """
    Wrapper around scikit-learn's TfidfVectorizer, driven by ml.yaml.

    Parameters
    ----------
    ml_config : dict, optional
        Configuration dictionary loaded from `ml.yaml`. If None, it
        will be loaded automatically.

    Attributes
    ----------
    vectorizer : TfidfVectorizer
        The underlying scikit-learn vectorizer instance.
    """

    def __init__(self, ml_config: Optional[Dict[str, Any]] = None) -> None:
        cfg = ml_config or load_ml_config()
        tfidf_cfg = cfg.get("ml_experiments", {}).get("tfidf", {})

        self.vectorizer = TfidfVectorizer(
            max_features=tfidf_cfg.get("max_features", 5000),
            ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
            min_df=tfidf_cfg.get("min_df", 2),
            max_df=tfidf_cfg.get("max_df", 0.95),
            sublinear_tf=tfidf_cfg.get("sublinear_tf", True),
            use_idf=tfidf_cfg.get("use_idf", True),
        )

    def fit(self, texts: Iterable[str]):
        """
        Fit the TF–IDF vectorizer on a collection of texts.

        Parameters
        ----------
        texts : Iterable[str]
            Training texts (already cleaned).
        """
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: Iterable[str]):
        """
        Transform a collection of texts into TF–IDF features.

        Parameters
        ----------
        texts : Iterable[str]
            Input texts.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse matrix of TF–IDF features.
        """
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: Iterable[str]):
        """
        Fit the TF–IDF vectorizer and transform the texts in one step.

        Parameters
        ----------
        texts : Iterable[str]
            Training texts.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse matrix of TF–IDF features for the training set.
        """
        return self.vectorizer.fit_transform(texts)


__all__ = ["TfidfFeatureExtractor"]
