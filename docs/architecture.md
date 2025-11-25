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
