"""
airline_sentiment

End-to-end implementation of airline tweet sentiment analysis using:
- Classical machine learning models (TF–IDF / BoW features)
- A CNN–BiLSTM model with Word2Vec embeddings
- Transformer-based models (BERT, ELECTRA, ALBERT, RoBERTa, DistilBERT, etc.)

The package structure mirrors our experimental pipeline:
- data/          : dataset loading and management
- preprocessing/ : text cleaning, splitting, augmentation
- features/      : TF–IDF, BoW, Word2Vec embeddings
- models/        : ML, CNN–BiLSTM, and transformers
- evaluation/    : metrics and result analysis
- utils/         : configuration, logging, seeding
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

# Simple version placeholder; update as needed when tagging releases.
__version__ = "0.1.0"
