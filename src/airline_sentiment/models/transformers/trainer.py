"""
Transformer training and evaluation for airline tweet sentiment.

This module:
- Loads processed train/val/test splits via `dataset_manager`.
- Builds TweetTransformersDataset objects using a HuggingFace tokenizer.
- Fine-tunes pre-trained transformer models for sentiment classification.
- Evaluates on the test set (accuracy, per-class precision/recall/F1, macro F1).
- Saves aggregated results to: outputs/transformers_results_<dataset>.csv

Configuration sources:
- Global config:         config/config.yaml
- Transformers config:   config/transformers.yaml

Typical CLI usage
-----------------
python -m airline_sentiment.models.transformers.trainer --dataset airline_us
python -m airline_sentiment.models.transformers.trainer --dataset airline_global
python -m airline_sentiment.models.transformers.trainer --dataset airline_merged

You can also restrict to a single model key defined in transformers.yaml, e.g.:

python -m airline_sentiment.models.transformers.trainer --dataset airline_us --model_key bert_base
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction

from airline_sentiment.evaluation.metrics import compute_classification_metrics
from airline_sentiment.models.transformers.dataset import create_transformers_datasets
from airline_sentiment.utils.config import (
    load_global_config,
    load_transformers_config,
    PROJECT_ROOT,
)
from airline_sentiment.utils.seed import set_global_seed


def _build_label_maps(global_cfg: Dict[str, Any]) -> Dict[str, Dict]:
    """
    Build label2id and id2label mappings from global config.

    Returns a dict with keys:
    - "label2id": mapping from label string to int
    - "id2label": mapping from int to label string
    """
    mapping = global_cfg.get("labels", {}).get("mapping", {})
    label2id = {str(name): int(idx) for name, idx in mapping.items()}
    id2label = {int(idx): str(name) for name, idx in mapping.items()}
    return {"label2id": label2id, "id2label": id2label}


def _run_single_transformer_experiment(
    dataset_name: str,
    model_key: str,
    model_cfg: Dict[str, Any],
    common_cfg: Dict[str, Any],
    global_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run a single transformer model training + evaluation experiment.

    Parameters
    ----------
    dataset_name : str
        Dataset name ("airline_us", "airline_global", "airline_merged").
    model_key : str
        Key in transformers.yaml (e.g., "bert_base", "electra_base").
    model_cfg : dict
        Model-specific configuration from transformers.yaml.
    common_cfg : dict
        Shared transformer training configuration from transformers.yaml.
    global_cfg : dict
        Global configuration from config.yaml.

    Returns
    -------
    Dict[str, Any]
        A dictionary of test metrics plus metadata for this experiment.
    """
    set_global_seed()

    model_name = model_cfg.get("model_name")
    if not model_name:
        raise ValueError(f"Model '{model_key}' in transformers.yaml is missing 'model_name'.")

    label_maps = _build_label_maps(global_cfg)
    label2id = label_maps["label2id"]
    id2label = label_maps["id2label"]
    num_labels = len(label2id)

    # ------------------------------------------------------------------
    # Tokenizer & datasets
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    datasets = create_transformers_datasets(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        transformers_config={"transformers": {"common": common_cfg}},
        global_config=global_cfg,
        apply_augmentation_to_train=True,
    )

    train_dataset = datasets["train"]
    val_dataset = datasets["val"]
    test_dataset = datasets["test"]

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    # ------------------------------------------------------------------
    # Training arguments
    # ------------------------------------------------------------------
    outputs_root = global_cfg.get("paths", {}).get("outputs", "outputs")
    out_dir = PROJECT_ROOT / outputs_root / f"transformers_{dataset_name}_{model_key}"
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=common_cfg.get("epochs", 4),
        per_device_train_batch_size=common_cfg.get("batch_size", 32),
        per_device_eval_batch_size=common_cfg.get("batch_size", 32),
        learning_rate=common_cfg.get("learning_rate", 2e-5),
        weight_decay=common_cfg.get("weight_decay", 0.0),
        warmup_ratio=common_cfg.get("warmup_ratio", 0.1),
        logging_steps=common_cfg.get("logging_steps", 50),
        evaluation_strategy="steps",
        eval_steps=common_cfg.get("eval_steps", 200),
        save_strategy="steps",
        save_steps=common_cfg.get("eval_steps", 200),
        save_total_limit=common_cfg.get("save_total_limit", 2),
        load_best_model_at_end=True,
        metric_for_best_model=common_cfg.get("metric_for_best", "macro_f1"),
        greater_is_better=common_cfg.get("greater_is_better", True),
        gradient_accumulation_steps=common_cfg.get("gradient_accumulation_steps", 1),
        max_grad_norm=common_cfg.get("max_grad_norm", 1.0),
        logging_dir=str(out_dir / "logs"),
        report_to=[],  # disable external loggers by default
    )

    # ------------------------------------------------------------------
    # Metrics callback for Trainer
    # ------------------------------------------------------------------
    def _compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        preds = eval_pred.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = preds.argmax(axis=-1)
        labels = eval_pred.label_ids

        # Use our shared metrics helper with readable label names
        metrics = compute_classification_metrics(labels, preds, label_names=id2label)
        # Trainer expects float values
        return {k: float(v) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )

    # ------------------------------------------------------------------
    # Train & evaluate
    # ------------------------------------------------------------------
    trainer.train()

    # Evaluate on test set
    test_output = trainer.predict(test_dataset)
    test_preds = test_output.predictions
    if isinstance(test_preds, tuple):
        test_preds = test_preds[0]
    test_pred_labels = test_preds.argmax(axis=-1)
    test_labels = test_output.label_ids

    test_metrics = compute_classification_metrics(test_labels, test_pred_labels, label_names=id2label)

    # Build record
    record: Dict[str, Any] = {
        "dataset": dataset_name,
        "model": model_key,
        "model_name": model_name,
    }
    record.update(test_metrics)

    return record


def run_transformer_experiments_for_dataset(
    dataset_name: str,
    single_model_key: Optional[str] = None,
    save_results: bool = True,
) -> pd.DataFrame:
    """
    Run transformer experiments on a dataset for all enabled models
    (or a single model if specified).

    Parameters
    ----------
    dataset_name : str
        One of "airline_us", "airline_global", "airline_merged".
    single_model_key : str, optional
        If provided, only this model key from transformers.yaml will be run.
    save_results : bool, default True
        If True, results are saved to:
            outputs/transformers_results_<dataset_name>.csv

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per model containing test metrics.
    """
    global_cfg = load_global_config()
    tcfg_all = load_transformers_config()
    tcfg = tcfg_all.get("transformers", {})

    common_cfg = tcfg.get("common", {})
    models_cfg = tcfg.get("models", {})

    records: List[Dict[str, Any]] = []

    if single_model_key is not None:
        if single_model_key not in models_cfg:
            raise KeyError(f"Model key '{single_model_key}' not found in transformers.yaml.")
        model_cfg = models_cfg[single_model_key]
        if not model_cfg.get("enabled", True):
            print(f"Model '{single_model_key}' is disabled in transformers.yaml. Skipping.")
        else:
            rec = _run_single_transformer_experiment(
                dataset_name=dataset_name,
                model_key=single_model_key,
                model_cfg=model_cfg,
                common_cfg=common_cfg,
                global_cfg=global_cfg,
            )
            records.append(rec)
    else:
        # Run all enabled models
        for model_key, model_cfg in models_cfg.items():
            if not model_cfg.get("enabled", True):
                continue
            rec = _run_single_transformer_experiment(
                dataset_name=dataset_name,
                model_key=model_key,
                model_cfg=model_cfg,
                common_cfg=common_cfg,
                global_cfg=global_cfg,
            )
            records.append(rec)

    results_df = pd.DataFrame(records)

    if save_results:
        outputs_root = global_cfg.get("paths", {}).get("outputs", "outputs")
        out_dir = PROJECT_ROOT / outputs_root
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"transformers_results_{dataset_name}.csv"
        results_df.to_csv(out_path, index=False)
        print(f"Saved transformer test results for {dataset_name} to: {out_path}")

    return results_df


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune transformer models for airline tweet sentiment."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="airline_us",
        choices=["airline_us", "airline_global", "airline_merged"],
        help="Dataset name to train on.",
    )
    parser.add_argument(
        "--model_key",
        type=str,
        default=None,
        help="Optional model key from transformers.yaml (e.g., 'bert_base'). "
             "If omitted, all enabled models are run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_transformer_experiments_for_dataset(
        dataset_name=args.dataset,
        single_model_key=args.model_key,
        save_results=True,
    )


if __name__ == "__main__":
    main()
