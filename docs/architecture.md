# Project Architecture

This document describes the architecture of the **Airline Tweet Sentiment** repository, which implements the experiments from:

**Exploring Transformer Models for Sentiment Analysis in Airline Service Reviews**  
https://ieeexplore.ieee.org/abstract/document/10796289

The goal is to provide a clear end-to-end view of how data flows through the system and how each component fits into our experimental pipeline.

---

## 1. High-Level Overview

At a high level, the pipeline is:

1. **Raw data ingestion**  
   Raw airline tweet CSVs go into `data/raw/`.

2. **Preprocessing & splitting**  
   We clean the text, normalize labels, and create stratified train/val/test splits per dataset into `data/processed/<dataset>/`.

3. **Feature extraction**  
   - Classical ML: TF–IDF and Bag-of-Words (BoW).  
   - Deep learning: Word2Vec embeddings for CNN–BiLSTM.  
   - Transformers: Subword tokenization via HuggingFace tokenizers.

4. **Model training**  
   - Classical ML models: LR, KNN, SVM, Decision Tree, Random Forest, AdaBoost.  
   - CNN–BiLSTM: Word2Vec-initialized embedding + conv filters + BiLSTM.  
   - Transformers: Fine-tuned pre-trained models (e.g. BERT, ELECTRA, ALBERT, RoBERTa).

5. **Evaluation & analysis**  
   We compute accuracy, per-class precision/recall/F1, macro F1, and summarize results across models and datasets.

All behavior is controlled using YAML configs under `config/`.

---

## 2. Package Structure

```text
src/airline_sentiment/
├── __init__.py
│
├── data/
│   ├── __init__.py
│   ├── datasets.py
│   └── dataset_manager.py
│
├── preprocessing/
│   ├── __init__.py
│   ├── text_cleaning.py
│   ├── splitter.py
│   └── augmentation.py
│
├── features/
│   ├── __init__.py
│   ├── tfidf_vectorizer.py
│   ├── bow_vectorizer.py
│   └── word2vec_embeddings.py
│
├── models/
│   ├── __init__.py
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── logistic_regression.py
│   │   ├── knn.py
│   │   ├── svm.py
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   ├── adaboost.py
│   │   └── experiment_runner.py
│   │
│   ├── dl/
│   │   ├── __init__.py
│   │   └── cnn_bilstm.py
│   │
│   └── transformers/
│       ├── __init__.py
│       ├── dataset.py
│       └── trainer.py
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py
│   └── analysis.py
│
└── utils/
    ├── __init__.py
    ├── config.py
    ├── seed.py
    └── logging_utils.py
```
3. Data Layer
3.1 data/datasets.py

Responsibilities:
-- Read raw CSVs from data/raw/ based on filenames in config/config.yaml.
-- Normalize column names (e.g., text, tweet_text, content → text).
-- Map sentiment labels to unified string labels (negative, neutral, positive) and numeric IDs (0, 1, 2).
Output:
A pandas.DataFrame with columns:
-- text – original tweet text.
-- label_str – sentiment as string.
-- label_id – sentiment as integer.

3.2 data/dataset_manager.py

Responsibilities:
-- Ensure processed splits exist for a dataset (ensure_processed_splits).
-- Load processed train/val/test CSVs.
-- Optionally apply augmentation-based class balancing to the training set.
This acts as the single source of truth for experiments needing a dataset.

4. Preprocessing Layer
4.1 preprocessing/text_cleaning.py

Implements the TextCleaner:
-- Lowercasing
-- URL, mention, hashtag, number, and emoji handling
-- Punctuation removal / normalization
-- Stopword removal and light stemming
-- Handles “airline tweet” quirks while preserving sentiment cues as much as possible.

4.2 preprocessing/splitter.py

Pipeline to:
-- Load raw data via data/datasets.py.
-- Clean text → text_clean.
-- Perform stratified splits (train/val/test) based on label_id.
-- Save splits into data/processed/<dataset>/.

4.3 preprocessing/augmentation.py

Implements EDA-style augmentation:
-- Synonym replacement.
-- Random insertion / swap / deletion (with constraints).
-- Class balancing helper that over-samples minority classes via augmentation until target ratios are reached.
Used primarily to make class distributions more balanced in training.

5. Feature Layer
5.1 features/tfidf_vectorizer.py
-- Wraps sklearn.feature_extraction.text.TfidfVectorizer.
-- Controlled by config/ml.yaml (max_features, ngram_range, etc.).
--Produces sparse TF–IDF matrices.

5.2 features/bow_vectorizer.py
-- Wraps sklearn.feature_extraction.text.CountVectorizer.
-- Controlled by config/ml.yaml.
-- Produces sparse Bag-of-Words representations.

5.3 features/word2vec_embeddings.py
Word2VecTrainer:
-- Builds a tokenized corpus from cleaned texts.
-- Trains a Word2Vec model with hyperparameters from config/cnn_bilstm.yaml.
-- Saves / loads models to/from models/word2vec/.
build_embedding_matrix:
-- Transforms a word_index (token → ID) into a dense embedding matrix.
-- Uses trained Word2Vec vectors where available, random vectors otherwise.
-- Embedding matrix feeds directly into the CNN–BiLSTM embedding layer.

6. Model Layer
6.1 Classical ML – models/ml/
-- Wrappers (logistic_regression.py, knn.py, svm.py, decision_tree.py, random_forest.py, adaboost.py):
-- Each file exposes a build_* function that returns a configured sklearn model using config/ml.yaml.
Experiment runner (experiment_runner.py):
-- Loads train/val/test splits via dataset_manager.
-- Builds TF–IDF or BoW features.
-- Trains/evaluates all enabled models (per ml.yaml).
-- Computes accuracy, per-class metrics, macro F1.
-- Saves results to outputs/ml_results_<dataset>.csv.
6.2 CNN–BiLSTM – models/dl/cnn_bilstm.py
Core components:
- Vocabulary & tokenization:
-- build_vocab creates word_index from cleaned texts.
-- encode_text maps sentences to fixed-length ID sequences.
- Dataset:
-- TweetSequenceDataset wraps (text_clean, label_id) into tensors.
- Model:
- CNNBiLSTM:
-- Word2Vec-initialized embedding layer.
-- Multiple 1D conv layers with ReLU + max-pooling.
-- Concatenated features fed into BiLSTM.
-- Dropout + FC layer → logits for 3-class sentiment.
- Training pipeline:
-- train_and_evaluate_cnn_bilstm(dataset_name):
-- Loads splits, trains Word2Vec, builds embeddings.
-- Trains model with Adam + CrossEntropyLoss.
-- Early stopping based on validation macro F1.
-- Logs test metrics and saves to outputs/cnn_bilstm_results_<dataset>.csv.
6.3 Transformers – models/transformers/
- dataset.py:
- create_transformers_datasets:
-- Loads splits via dataset_manager.
-- Tokenizes text_clean with a HuggingFace tokenizer.
-- Wraps into TweetTransformersDataset for PyTorch DataLoader.
- trainer.py:
-- For each model key in config/transformers.yaml:
-- Builds AutoTokenizer and AutoModelForSequenceClassification.
-- Uses HuggingFace Trainer for fine-tuning.
-- Computes metrics via our shared compute_classification_metrics.
-- Saves aggregated results to outputs/transformers_results_<dataset>.csv.

7. Evaluation Layer
7.1 evaluation/metrics.py
- compute_classification_metrics(y_true, y_pred, label_names):
-- Accuracy
-- Per-class precision, recall, F1
-- Macro-averaged F1
Used by:
-- CNN–BiLSTM training loop.
-- Transformer trainer.
-- Optionally classical ML.
7.2 evaluation/analysis.py
- Utilities to:
-- Load ml_results_*.csv, cnn_bilstm_results_*.csv, transformers_results_*.csv.
-- Concatenate and summarize.
-- Extract best configuration per dataset based on a metric (typically macro F1).

8. Utilities
8.1 utils/config.py
- Resolves PROJECT_ROOT.
- Loads YAML configs:
-- config/config.yaml
-- config/ml.yaml
-- config/cnn_bilstm.yaml
-- config/transformers.yaml
Centralizes configuration logic so all modules read settings consistently.
8.2 utils/seed.py
- set_global_seed(seed):
-- Sets Python random, NumPy, and PyTorch seeds (CPU + CUDA).
-- Ensures experiments are as reproducible as possible.
8.3 utils/logging_utils.py
- get_logger(name, log_to_file=False, filename=None):
-- Console + optional file logging.
-- Controlled via paths.logs in config/config.yaml.
  
9. Scripts
```text
- scripts/prepare_datasets.py
Build processed splits for all or one dataset.

- scripts/run_ml_experiments.py
Run TF–IDF/BoW baselines over all enabled ML models.

- scripts/run_cnn_bilstm.py
Train and evaluate CNN–BiLSTM for each dataset.

- scripts/run_transformers.py
Fine-tune transformer models (single or multiple) for each dataset.
```
These scripts correspond to the main experimental blocks in our paper and allow full reproduction of the reported results.

This architecture is deliberately modular so that we can easily:
-- Swap in new models,
-- Extend augmentation and preprocessing,
-- Or plug in new datasets with minimal changes to the core pipeline.
