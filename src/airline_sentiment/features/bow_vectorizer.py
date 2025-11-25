"""
Bag-of-Words (BoW) feature extraction for airline tweet sentiment experiments.

This module provides a thin wrapper around scikit-learn's
`CountVectorizer`, configured via `config/ml.yaml`, so that all
classical ML models share the same BoW settings.

Typical usage:
--------------
from airline_sentiment.features.bow_vectorizer import BowFeatureExtractor

extractor = BowFeatureExtractor()
X_train = extractor.fit_transform(df_train["text_clean"])
X_val   = extractor.transform(df_val["text_clean"])
X_test  = extractor.transform(df_test["text_clean"])
"""

from __future__ import annotations

from typing import Iterable, Optional, Dict, Any

from sklearn.feature_extraction.text import CountVectorizer

from airline_sentiment.utils.config import load_ml_config


class BowFeatureExtractor:
    """
    Wrapper around scikit-learn's CountVectorizer, driven by ml.yaml.

    Parameters
    ----------
    ml_config : dict, optional
        Configuration dictionary loaded from `ml.yaml`. If None, it
        will be loaded automatically.

    Attributes
    ----------
    vectorizer : CountVectorizer
        The underlying scikit-learn vectorizer instance.
    """

    def __init__(self, ml_config: Optional[Dict[str, Any]] = None) -> None:
        cfg = ml_config or load_ml_config()
        bow_cfg = cfg.get("ml_experiments", {}).get("bow", {})

        self.vectorizer = CountVectorizer(
            max_features=bow_cfg.get("max_features", 5000),
            ngram_range=tuple(bow_cfg.get("ngram_range", [1, 2])),
            min_df=bow_cfg.get("min_df", 2),
            max_df=bow_cfg.get("max_df", 0.95),
            binary=bow_cfg.get("binary", False),
        )

    def fit(self, texts: Iterable[str]):
        """
        Fit the BoW vectorizer on a collection of texts.

        Parameters
        ----------
        texts : Iterable[str]
            Training texts (already cleaned).
        """
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: Iterable[str]):
        """
        Transform a collection of texts into BoW features.

        Parameters
        ----------
        texts : Iterable[str]
            Input texts.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse matrix of BoW features.
        """
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: Iterable[str]):
        """
        Fit the BoW vectorizer and transform the texts in one step.

        Parameters
        ----------
        texts : Iterable[str]
            Training texts.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse matrix of BoW features for the training set.
        """
        return self.vectorizer.fit_transform(texts)


__all__ = ["BowFeatureExtractor"]
