"""
Small, dependency-free helpers for reading/writing JSON state files.

The project writes ~8 machine-local JSON state files from ~6 modules, each
with its own copy of the try/except json.load boilerplate. New code should
use these helpers; existing callers can adopt them incrementally.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


def load_json(path: str, default: Any = None) -> Any:
    """Load a JSON file, returning ``default`` on any failure."""
    if default is None:
        default = {}
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, ValueError) as exc:
        logger.debug(f"load_json failed for {path}: {exc}")
        return default


def save_json(path: str, data: Any, indent: int = 2) -> bool:
    """Atomically write JSON data to ``path`` (tmp + replace)."""
    try:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        os.replace(tmp, path)
        return True
    except (OSError, TypeError, ValueError) as exc:
        logger.warning(f"save_json failed for {path}: {exc}")
        return False


def safe_float(value: Any, default: float = 0.0) -> float:
    """Return a finite float or ``default`` (None/NaN/Inf/invalid -> default)."""
    try:
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge ``override`` into ``base`` (new dict)."""
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out
