"""
Central runtime settings for the StratoCrypto ML/evaluation pipeline.

This module replaces the scattered hardcoded constants used across the
prediction stack (ml_trainer, calibrate_predictor, validate_ml, etc.)
with one typed source of truth. The live bot/dashboard still keep their
own env-driven thresholds, but every training/eval entry point should
read from here so paths, symbols, lookbacks and seeds stay consistent.
"""
from __future__ import annotations

import os
import sys

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
STATE_DIR = BASE_DIR

BINANCE_BASE = "https://api.binance.com"

# One shared history depth so the trained scaler and live inference see the
# same feature warm-up. EWM/rolling features (ema_50_200, bb_50_*, ...) need
# hundreds of bars to stabilise; training on 1500 bars and inferring on 100
# distorts the same feature between the two paths.
TRAIN_LOOKBACK = 1500
INFERENCE_LOOKBACK = TRAIN_LOOKBACK

# Default prediction universe used by every ML entry point.
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

# (tf_hours, interval) mapping shared by training and inference.
TIMEFRAME_INTERVALS = [(2, "5m"), (6, "15m"), (8, "30m"), (24, "1h")]
INTERVAL_TF = {"5m": 2, "15m": 6, "30m": 8, "1h": 24}
HORIZON_BARS = {2: 24, 6: 24, 8: 16, 24: 24}

# Feature selection cap.
MAX_FEATURES = 50

# Reproducibility.
RANDOM_STATE = 42

# Model artifact names (bare filenames).
ENSEMBLE_FILE = "ensemble_v2.pkl"
META_LABELER_FILE = "meta_labeler.pkl"
CALIBRATION_FILE = "calibration.json"
TRACKER_FILE = "prediction_accuracy.json"
TRAINING_BUFFER_FILE = "training_buffer.json"

# Full paths for the modules that open them directly.
ENSEMBLE_PATH = os.path.join(MODELS_DIR, ENSEMBLE_FILE)
META_LABELER_PATH = os.path.join(MODELS_DIR, META_LABELER_FILE)
CALIBRATION_PATH = os.path.join(MODELS_DIR, CALIBRATION_FILE)
TRACKER_PATH = os.path.join(STATE_DIR, TRACKER_FILE)
TRAINING_BUFFER_PATH = os.path.join(BASE_DIR, TRAINING_BUFFER_FILE)

# A human-readable version stamped on every artifact the pipeline writes.
# Bump it whenever the feature schema or the ensemble architecture changes.
PIPELINE_VERSION = "2.1.0"


def seed_everything(seed: int = RANDOM_STATE) -> None:
    """Seed numpy (and, if installed, torch/sklearn) for reproducibility."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass
    try:
        import random
        random.seed(seed)
    except Exception:
        pass


def ensure_dirs() -> None:
    for d in (MODELS_DIR, DATA_DIR):
        os.makedirs(d, exist_ok=True)


def _env_bool(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_list(name: str, default: list) -> list:
    raw = os.environ.get(name, "")
    if not raw.strip():
        return list(default)
    return [x.strip().upper() for x in raw.replace(";", ",").split(",") if x.strip()]


# Environment overrides (keep the existing env knobs working).
ML_OPTUNA = _env_bool("ML_OPTUNA")
FOUNDATION_MODELS = _env_bool("FOUNDATION_MODELS", "1")
SYMBOLS = _env_list("ML_SYMBOLS", DEFAULT_SYMBOLS)

# Guard against importing anything heavy at import time.
sys = sys  # noqa: F841 (import kept for parity with other modules)
