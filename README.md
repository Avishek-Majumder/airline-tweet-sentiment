# Airline Tweet Sentiment – End-to-End Implementation

This repository implements a complete airline tweet sentiment analysis pipeline based on our paper:

**Paper Title:** Exploring Transformer Models for Sentiment Analysis in Airline Service Reviews  
**Paper Link:** https://ieeexplore.ieee.org/abstract/document/10796289

We cover three families of models:

- **Classical ML models** with TF–IDF / Bag-of-Words features  
- **CNN–BiLSTM** model with **Word2Vec** embeddings  
- **Transformer-based models** (BERT, ELECTRA, ALBERT, RoBERTa, DistilBERT, etc.)

The code is organized so that all components (preprocessing, feature extraction, model training, and evaluation) are reproducible and configurable via YAML files.

---

## 1. Repository Structure

```text
airline-tweet-sentiment/
├── config/
│   ├── config.yaml             # Global paths, datasets, preprocessing, label mapping
│   ├── ml.yaml                 # Classical ML + feature extraction settings
│   ├── cnn_bilstm.yaml         # CNN–BiLSTM + Word2Vec settings
│   └── transformers.yaml       # Transformer models & training hyperparameters
│
├── data/
│   ├── raw/                    # Raw CSVs (US airline, global, merged)
│   └── processed/              # Auto-generated train/val/test splits per dataset
│       └── <dataset_name>/
│           ├── train.csv
│           ├── val.csv
│           └── test.csv
│
├── models/                     # (Optional) Saved models, e.g. Word2Vec
│   └── word2vec/
│       └── word2vec.model
│
├── outputs/
│   ├── ml_results_<dataset>.csv          # Classical ML results
│   ├── cnn_bilstm_results_<dataset>.csv  # CNN–BiLSTM results
│   └── transformers_results_<dataset>.csv# Transformer results
│
├── runs/                       # Log files (if enabled)
│
├── scripts/
│   ├── prepare_datasets.py     # Build processed train/val/test splits
│   ├── run_ml_experiments.py   # Run TF–IDF/BoW + classical ML experiments
│   ├── run_cnn_bilstm.py       # Train/evaluate CNN–BiLSTM
│   └── run_transformers.py     # Fine-tune transformer models
│
└── src/
    └── airline_sentiment/
        ├── __init__.py
        │
        ├── data/
        │   ├── datasets.py          # Load raw CSVs, normalize columns, map labels
        │   └── dataset_manager.py   # Ensure/load processed splits (+ optional augmentation)
        │
        ├── preprocessing/
        │   ├── text_cleaning.py     # TextCleaner: URLs, mentions, hashtags, emojis, stemming, etc.
        │   ├── splitter.py          # Clean + stratified train/val/test splitting
        │   └── augmentation.py      # EDA-style augmentations + class balancing
        │
        ├── features/
        │   ├── tfidf_vectorizer.py      # TF–IDF feature extractor
        │   ├── bow_vectorizer.py        # Bag-of-Words feature extractor
        │   └── word2vec_embeddings.py   # Word2VecTrainer + embedding matrix builder
        │
        ├── models/
        │   ├── ml/
        │   │   ├── logistic_regression.py
        │   │   ├── knn.py
        │   │   ├── svm.py
        │   │   ├── decision_tree.py
        │   │   ├── random_forest.py
        │   │   ├── adaboost.py
        │   │   └── experiment_runner.py   # Orchestrates all classical ML experiments
        │   │
        │   ├── dl/
        │   │   └── cnn_bilstm.py         # CNN–BiLSTM model + full training pipeline
        │   │
        │   └── transformers/
        │       ├── dataset.py            # Tokenization + Dataset wrappers
        │       └── trainer.py            # HF Trainer-based fine-tuning & evaluation
        │
        ├── evaluation/
        │   ├── metrics.py            # Accuracy + per-class precision/recall/F1 + macro F1
        │   └── analysis.py           # Aggregate & summarize results from CSVs
        │
        └── utils/
            ├── config.py             # YAML config loaders + project root resolution
            ├── seed.py               # Global random seeding (Python/NumPy/PyTorch)
            └── logging_utils.py      # Unified logging helper
```
2. Installation

We assume Python ≥ 3.9.

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# or
.\.venv\Scripts\activate           # Windows
```
2. Install dependencies (we use standard ML / DL libraries plus HuggingFace Transformers):
```bash
pip install -r requirements.txt
```
If you prefer, you can also use pip install -e . once the project is packaged with pyproject.toml / setup.cfg.

3. Data Setup
Place the raw airline tweet datasets in data/raw/ and ensure the filenames in config/config.yaml match your actual CSV names.
Example (from config/config.yaml):
```bash
datasets:
  airline_us:
    filename: "airline_tweets_us.csv"
  airline_global:
    filename: "airline_tweets_global.csv"
  airline_merged:
    filename: "airline_tweets_merged.csv"
```
Each raw CSV should contain at least:
-- A text column (e.g., text, tweet_text, or content)
-- A label column (e.g., airline_sentiment, sentiment, label, or polarity)
The loader normalizes these into:
-- text (original tweet)
-- label_str ∈ {negative, neutral, positive}
-- label_id ∈ {0, 1, 2}

4. Preprocessing & Splits
To build cleaned, stratified train/val/test splits:
```bash
# All datasets configured in config/config.yaml
python scripts/prepare_datasets.py

# Single dataset
python scripts/prepare_datasets.py --dataset airline_us
```
This will:

-- Load raw CSVs
-- Clean text using TextCleaner (URLs, mentions, hashtags, emojis, punctuation, stopwords, stemming)
-- Perform stratified train/val/test splits
-- Save them under data/processed/<dataset_name>/.

5. Classical ML Experiments (TF–IDF / BoW)
To run classical ML experiments (Logistic Regression, KNN, SVM, Decision Tree, Random Forest, AdaBoost):
```bash
# All datasets
python scripts/run_ml_experiments.py

# Single dataset
python scripts/run_ml_experiments.py --dataset airline_us
```
Configuration:
-- Feature extraction (TF–IDF / BoW) and model hyperparameters are controlled by config/ml.yaml.

The script saves:
-- outputs/ml_results_<dataset>.csv
(one row per (model, feature_type) with accuracy, per-class metrics, and macro F1).

6. CNN–BiLSTM + Word2Vec
To train and evaluate the CNN–BiLSTM model:
```bash
# All datasets
python scripts/run_cnn_bilstm.py

# Single dataset
python scripts/run_cnn_bilstm.py --dataset airline_us
```
The pipeline:

Loads processed train/val/test splits (optionally with augmentation on train).
Trains a Word2Vec model on cleaned tweets from one or more datasets.
Builds a vocabulary and embedding matrix.
Trains a CNN–BiLSTM classifier with early stopping (macro F1 on validation).
Evaluates on the test set and saves:
outputs/cnn_bilstm_results_<dataset>.csv
Key hyperparameters (filter sizes, BiLSTM hidden size, dropout, etc.) live in config/cnn_bilstm.yaml.

7. Transformer Models
To fine-tune transformer models (e.g., BERT, ELECTRA, ALBERT, RoBERTa, DistilBERT):
```bash
# All datasets + all enabled models from transformers.yaml
python scripts/run_transformers.py

# Single dataset, all enabled models
python scripts/run_transformers.py --dataset airline_us

# Single dataset, single model key (e.g., bert_base)
python scripts/run_transformers.py --dataset airline_us --model_key bert_base
```
Configuration:

config/transformers.yaml defines:
-- Shared training options (epochs, batch size, LR, warmup ratio, etc.)
-- Model list and their HuggingFace model_name strings.

For each (dataset, model_key), we:
-- Build tokenized datasets from cleaned text (text_clean).
-- Fine-tune using HuggingFace Trainer.

Evaluate on the test set and save:
-- outputs/transformers_results_<dataset>.csv

8. Evaluation & Result Analysis

Beyond the per-experiment CSVs, we provide simple utilities to aggregate and inspect results:
-- airline_sentiment.evaluation.metrics.compute_classification_metrics
– Shared accuracy + per-class precision/recall/F1 + macro F1.

airline_sentiment.evaluation.analysis:
```bash
load_results_df(pattern) – load all result CSVs (e.g., ml_results_*.csv).
summarize_best_per_dataset(results_df, metric="macro_f1", group_cols=[...]) – show the best configuration per dataset, according to a chosen metric.
```
Example quick analysis:
```bash
python -m airline_sentiment.evaluation.analysis
```
This loads ml_results_*.csv and prints the best ML configurations per dataset based on macro F1.

9. Reproducibility

We use a central project.random_seed in config/config.yaml.
airline_sentiment.utils.seed.set_global_seed sets seeds for:

Python’s random

NumPy

PyTorch (CPU + CUDA, if available)

We call this function in all major experiment entrypoints so runs are as reproducible as possible given hardware and library constraints.

10. How This Maps to Our Paper

Our implementation is designed to mirror the methodology, experiments, and metrics in our paper:

Exploring Transformer Models for Sentiment Analysis in Airline Service Reviews
(https://ieeexplore.ieee.org/abstract/document/10796289
)

Specifically, we provide:

Preprocessing & augmentation: consistent cleaning pipeline and class balancing.

Classical ML baseline: TF–IDF/BoW features with LR, KNN, SVM, DT, RF, AdaBoost.

CNN–BiLSTM: Word2Vec-based feature extractor + convolutional and recurrent layers.

Transformers: fine-tuned pre-trained language models for tweet-level sentiment.

Evaluation: accuracy and macro F1, along with class-wise precision/recall/F1.

This repository is ready to be pushed to GitHub as a complete, reproducible implementation of our airline tweet sentiment analysis framework aligned with the above paper.
