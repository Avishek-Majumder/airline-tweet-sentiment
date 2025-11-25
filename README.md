# Airline Tweet Sentiment Analysis – Traditional ML, CNN–BiLSTM, and Transformers

In this repository, we implement an end-to-end sentiment analysis pipeline for airline-related tweets using three families of models:
1. Classical machine learning models with TF–IDF and Bag-of-Words features.
2. A CNN–BiLSTM architecture with Word2Vec embeddings.
3. Pre-trained transformer models (BERT, ELECTRA, ALBERT, RoBERTa, DistilBERT, etc.) fine-tuned for tweet sentiment classification.

Our implementation follows the experimental design of a research study comparing these model families on multiple airline tweet datasets (including a merged dataset), evaluated with accuracy, precision, recall, and F1-score for negative, neutral, and positive sentiment.

## Repository structure (high level)

This is the structure we will gradually build:

- `src/airline_sentiment/`  
  Core Python package:
  - `data/` – dataset loaders and utilities  
  - `preprocessing/` – text cleaning, tokenization, augmentation  
  - `features/` – TF–IDF, Bag-of-Words, Word2Vec  
  - `models/` – classical ML, CNN–BiLSTM, and transformers  
  - `evaluation/` – metrics, result tables, plots  
  - `utils/` – config handling, logging, common helpers  

- `config/`  
  YAML configuration files for datasets, models, and experiments.

- `scripts/`  
  Entrypoints to:
  - Download datasets  
  - Run classical ML experiments  
  - Train the CNN–BiLSTM model  
  - Fine-tune transformer models  

- `notebooks/` (optional)  
  Exploratory data analysis and result visualization.

## Goals

- Provide a clean, reproducible implementation of the full experimental pipeline.
- Mirror the methodology and metrics from the paper as closely as possible.
- Make it straightforward to download the data, train models, and reproduce the reported comparisons.

## Getting started

We will add detailed setup and usage instructions later, including:

- Python environment and dependencies.
- Data download and preprocessing commands.
- How to run:
  - Classical ML experiments.
  - CNN–BiLSTM training.
  - Transformer fine-tuning and evaluation.

For now, you can clone this repository structure and follow the step-by-step instructions as we build out the codebase.
