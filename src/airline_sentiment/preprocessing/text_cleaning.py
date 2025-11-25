"""
Text cleaning and normalization utilities for airline tweet sentiment.

We apply the preprocessing decisions specified in config/config.yaml:
- lowercasing
- URL / mention / hashtag cleanup
- punctuation and emoji removal
- optional number removal
- stopword removal
- stemming

These functions are shared across classical ML, CNN–BiLSTM, and
transformer-based experiments to ensure consistency.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Dict, Any

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from airline_sentiment.utils.config import load_global_config

# Ensure required NLTK resources are available.
# (If missing, user will need to run the downloads once in their environment.)
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
USER_PATTERN = re.compile(r"@\w+")
HASHTAG_SYMBOL_PATTERN = re.compile(r"#")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "]+",
    flags=re.UNICODE,
)
PUNCT_PATTERN = re.compile(r"[^\w\s]")
NUMBERS_PATTERN = re.compile(r"\d+")


class TextCleaner:
    """
    Configurable tweet/text cleaning pipeline.

    Parameters
    ----------
    config : dict, optional
        Global configuration dictionary. If None, config is loaded from
        config/config.yaml.

    Notes
    -----
    This class exposes `clean(text: str) -> str` for a single string,
    and `clean_many(texts: Iterable[str]) -> List[str]` for batches.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        cfg = config or load_global_config()
        pre_cfg = cfg.get("preprocessing", {})

        self.lowercase: bool = bool(pre_cfg.get("lowercase", True))
        self.remove_urls: bool = bool(pre_cfg.get("remove_urls", True))
        self.remove_user_mentions: bool = bool(pre_cfg.get("remove_user_mentions", True))
        self.remove_hashtags_symbol_only: bool = bool(
            pre_cfg.get("remove_hashtags_symbol_only", True)
        )
        self.remove_punctuation: bool = bool(pre_cfg.get("remove_punctuation", True))
        self.remove_numbers: bool = bool(pre_cfg.get("remove_numbers", False))
        self.remove_emojis: bool = bool(pre_cfg.get("remove_emojis", True))
        self.strip_whitespace: bool = bool(pre_cfg.get("strip_whitespace", True))

        self.stopwords_lang: str | None = pre_cfg.get("stopwords", "english")
        self.stemming_method: str | None = pre_cfg.get("stemming", "porter")

        # Initialize stopwords
        if self.stopwords_lang and self.stopwords_lang.lower() == "english":
            self.stopwords_set = set(stopwords.words("english"))
        else:
            self.stopwords_set = set()

        # Initialize stemmer
        if self.stemming_method and self.stemming_method.lower() == "porter":
            self.stemmer = PorterStemmer()
        else:
            self.stemmer = None

    def _basic_cleanup(self, text: str) -> str:
        """Apply basic regex-based cleanup operations."""
        if self.remove_urls:
            text = URL_PATTERN.sub(" ", text)
        if self.remove_user_mentions:
            text = USER_PATTERN.sub(" ", text)
        if self.remove_hashtags_symbol_only:
            # Remove only the "#" character; keep the word
            text = HASHTAG_SYMBOL_PATTERN.sub("", text)
        if self.remove_emojis:
            text = EMOJI_PATTERN.sub(" ", text)
        if self.remove_numbers:
            text = NUMBERS_PATTERN.sub(" ", text)
        if self.remove_punctuation:
            text = PUNCT_PATTERN.sub(" ", text)
        return text

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization."""
        return text.split()

    def _apply_stopwords_and_stemming(self, tokens: Iterable[str]) -> List[str]:
        """Remove stopwords and apply stemming if configured."""
        cleaned_tokens: List[str] = []
        for tok in tokens:
            if not tok:
                continue
            if self.stopwords_set and tok.lower() in self.stopwords_set:
                continue
            if self.stemmer is not None:
                tok = self.stemmer.stem(tok)
            cleaned_tokens.append(tok)
        return cleaned_tokens

    def clean(self, text: str) -> str:
        """
        Clean a single input string according to the configuration.

        Steps (in order):
        1. Optional lowercasing.
        2. Remove URLs, mentions, emoji, punctuation, etc.
        3. Tokenize on whitespace.
        4. Remove stopwords and apply stemming.
        5. Join back into a space-separated string.
        """
        if not isinstance(text, str):
            text = str(text)

        if self.lowercase:
            text = text.lower()

        text = self._basic_cleanup(text)

        tokens = self._tokenize(text)
        tokens = self._apply_stopwords_and_stemming(tokens)

        cleaned = " ".join(tokens)

        if self.strip_whitespace:
            cleaned = cleaned.strip()

        return cleaned

    def clean_many(self, texts: Iterable[str]) -> List[str]:
        """
        Clean many input strings.

        Parameters
        ----------
        texts : Iterable[str]
            Sequence of texts to clean.

        Returns
        -------
        List[str]
            Cleaned texts.
        """
        return [self.clean(t) for t in texts]


__all__ = ["TextCleaner"]
