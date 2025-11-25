"""
Text data augmentation utilities for airline tweet sentiment.

We implement simple EDA-style augmentation operations:
- Synonym replacement
- Random insertion
- Random swap
- Random deletion

These operations are controlled by probabilities defined in
config/config.yaml under the `augmentation` section, and can be used
to upsample minority classes to match the majority class size.

Typical usage:
--------------
from airline_sentiment.preprocessing.augmentation import (
    Augmenter,
    balance_classes_with_augmentation,
)

augmenter = Augmenter()
df_balanced = balance_classes_with_augmentation(df_train, augmenter)
"""

from __future__ import annotations

import random
from typing import List, Dict, Any, Iterable, Tuple

import nltk
import pandas as pd
from nltk.corpus import wordnet

from airline_sentiment.utils.config import load_global_config

# Ensure NLTK resources for synonyms are available
try:
    _ = wordnet.synsets("good")
except LookupError:
    nltk.download("wordnet")
    nltk.download("omw-1.4")


class Augmenter:
    """
    Simple text augmentation class implementing EDA-style operations.

    Parameters
    ----------
    config : dict, optional
        Global configuration dictionary. If None, config is loaded from
        config/config.yaml.

    Notes
    -----
    We assume the input text is already cleaned/tokenized into simple
    whitespace-separated tokens (e.g., from TextCleaner).
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        cfg = config or load_global_config()
        aug_cfg = cfg.get("augmentation", {})

        ops_cfg = aug_cfg.get("operations", {})
        self.synonym_replacement_prob: float = float(
            ops_cfg.get("synonym_replacement_prob", 0.1)
        )
        self.random_insertion_prob: float = float(
            ops_cfg.get("random_insertion_prob", 0.05)
        )
        self.random_swap_prob: float = float(
            ops_cfg.get("random_swap_prob", 0.05)
        )
        self.random_deletion_prob: float = float(
            ops_cfg.get("random_deletion_prob", 0.02)
        )

        project_cfg = cfg.get("project", {})
        self.random_seed: int = int(project_cfg.get("random_seed", 42))
        random.seed(self.random_seed)

    # ------------------------------------------------------------------
    # Synonym utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _get_synonyms(word: str) -> List[str]:
        """Retrieve a list of synonyms for a word using WordNet."""
        synonyms: List[str] = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                candidate = lemma.name().replace("_", " ").lower()
                if candidate != word.lower() and candidate not in synonyms:
                    synonyms.append(candidate)
        return synonyms

    def _replace_word_with_synonym(self, tokens: List[str]) -> List[str]:
        """Randomly pick a token and replace it with a synonym if available."""
        if not tokens:
            return tokens

        indices = list(range(len(tokens)))
        random.shuffle(indices)

        for idx in indices:
            word = tokens[idx]
            synonyms = self._get_synonyms(word)
            if synonyms:
                tokens[idx] = random.choice(synonyms)
                break

        return tokens

    def _insert_synonym(self, tokens: List[str]) -> List[str]:
        """Randomly pick a token, get a synonym, and insert it at a random position."""
        if not tokens:
            return tokens

        word = random.choice(tokens)
        synonyms = self._get_synonyms(word)
        if not synonyms:
            return tokens

        synonym = random.choice(synonyms)
        insert_pos = random.randint(0, len(tokens))
        tokens.insert(insert_pos, synonym)
        return tokens

    def _swap_two_words(self, tokens: List[str]) -> List[str]:
        """Randomly swap two distinct word positions."""
        if len(tokens) < 2:
            return tokens

        idx1, idx2 = random.sample(range(len(tokens)), 2)
        tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]
        return tokens

    def _random_deletion(self, tokens: List[str], deletion_prob: float = 0.1) -> List[str]:
        """
        Randomly delete each word with a given probability.

        If all words are deleted, we return the original tokens to avoid
        producing empty strings.
        """
        if not tokens:
            return tokens

        kept_tokens = [t for t in tokens if random.random() > deletion_prob]
        if not kept_tokens:
            return tokens
        return kept_tokens

    # ------------------------------------------------------------------
    # High-level augmentation interface
    # ------------------------------------------------------------------

    def augment_text(self, text: str) -> str:
        """
        Apply a random combination of augmentation operations to a text.

        Parameters
        ----------
        text : str
            Cleaned, whitespace-separated text.

        Returns
        -------
        str
            Augmented text.
        """
        if not isinstance(text, str):
            text = str(text)

        tokens = text.split()
        if not tokens:
            return text

        # Synonym replacement
        if random.random() < self.synonym_replacement_prob:
            tokens = self._replace_word_with_synonym(tokens)

        # Random insertion
        if random.random() < self.random_insertion_prob:
            tokens = self._insert_synonym(tokens)

        # Random swap
        if random.random() < self.random_swap_prob:
            tokens = self._swap_two_words(tokens)

        # Random deletion
        if random.random() < self.random_deletion_prob:
            tokens = self._random_deletion(tokens, deletion_prob=0.1)

        return " ".join(tokens)


def _compute_target_counts(
    df: pd.DataFrame,
    label_col: str,
    target_mode: str,
) -> Dict[Any, int]:
    """
    Compute target sample count per class given the balancing mode.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    label_col : str
        Column name containing class labels.
    target_mode : str
        Balancing strategy:
        - "max_class": upsample all classes to match the largest class.

    Returns
    -------
    Dict[label_value, int]
        Target sample count per class.
    """
    class_counts = df[label_col].value_counts().to_dict()

    if target_mode == "max_class":
        max_count = max(class_counts.values())
        return {cls: max_count for cls in class_counts}

    raise ValueError(f"Unknown target_mode for augmentation: {target_mode}")


def balance_classes_with_augmentation(
    df: pd.DataFrame,
    augmenter: Augmenter,
    label_col: str = "label_str",
    text_col: str = "text_clean",
    config: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Upsample minority classes via augmentation to achieve class balance.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing at least `text_col` and `label_col`.
    augmenter : Augmenter
        Augmenter instance to generate new texts.
    label_col : str, default "label_str"
        Name of the column containing class labels.
    text_col : str, default "text_clean"
        Name of the column containing cleaned text.
    config : dict, optional
        Global config. If None, loaded automatically.

    Returns
    -------
    pd.DataFrame
        New dataframe with original and augmented samples combined.
    """
    cfg = config or load_global_config()
    aug_cfg = cfg.get("augmentation", {})
    enabled = bool(aug_cfg.get("enabled", True))
    if not enabled:
        return df

    target_mode = aug_cfg.get("target_class_balance", "max_class")

    df = df.copy()
    target_counts = _compute_target_counts(df, label_col=label_col, target_mode=target_mode)

    augmented_rows = []
    for cls, target_count in target_counts.items():
        cls_df = df[df[label_col] == cls]
        current_count = len(cls_df)
        if current_count >= target_count:
            continue

        needed = target_count - current_count
        # We repeatedly sample from existing class rows and augment their text
        source_rows = cls_df.to_dict(orient="records")
        for _ in range(needed):
            base_row = random.choice(source_rows)
            base_text = base_row.get(text_col, "")
            new_text = augmenter.augment_text(base_text)

            new_row = dict(base_row)
            new_row[text_col] = new_text
            # Optionally mark as augmented
            new_row["is_augmented"] = True
            augmented_rows.append(new_row)

    if not augmented_rows:
        # Nothing to augment; return original dataframe
        return df

    df_aug = pd.DataFrame(augmented_rows, columns=list(df.columns) + ["is_augmented"])
    # Align columns: there might be missing "is_augmented" in original df
    if "is_augmented" not in df.columns:
        df["is_augmented"] = False

    combined = pd.concat([df, df_aug], ignore_index=True)
    return combined


__all__ = ["Augmenter", "balance_classes_with_augmentation"]
