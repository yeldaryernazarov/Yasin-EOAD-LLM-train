"""
Configuration file for AD detection training and inference
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    "model_name": "roberta-base",
    "max_length": 512,
    "num_labels": 2,
    "learning_rate": 2e-5,
    "batch_size": 8,
    "eval_batch_size": 16,
    "num_epochs": 5,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 1,
    "fp16": True,
    "save_strategy": "epoch",
    "evaluation_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "roc_auc",
    "save_total_limit": 3,
}

# Data configuration
DATA_CONFIG = {
    "text_column": "text",
    "label_column": "label",
    "id_column": "id",
    "age_column": "age",
    "text_path_column": "text_path",
    "test_size": 0.2,
    "val_size": 0.1,
    "random_state": 42,
    "stratify": True,
}

# Feature extraction configuration
FEATURE_CONFIG = {
    "linguistic_features": True,
    "paralinguistic_features": True,
    "max_features": 1000,
    "ngram_range": (1, 3),
    "min_df": 2,
    "max_df": 0.95,
}

# Evaluation configuration
EVAL_CONFIG = {
    "cv_folds": 5,
    "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
    "calibration_method": "isotonic",  # or "platt"
}

# Inference configuration
INFERENCE_CONFIG = {
    "model_path": MODELS_DIR / "best_model",
    "calibration_model_path": MODELS_DIR / "calibration_model.pkl",
    "confidence_threshold": 0.5,
    "explain_predictions": True,
    "top_k_features": 10,
}