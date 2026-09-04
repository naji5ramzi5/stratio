"""
Foundation Models – Chronos-2 (Amazon) + TimesFM 2.5 (Google)
Zero-shot time-series foundation models added as ensemble members.
Lazy-loaded on first use; safe to import without internet or weights.

Each model maps a closing-price series to a predicted % change over a horizon.
Confidence is derived from the forecast quantile spread (wider = less sure).
"""

import os
import threading
import logging
import numpy as np

logger = logging.getLogger("FoundationModels")

ENABLE_FOUNDATION = os.environ.get("FOUNDATION_MODELS", "1").strip().lower() not in ("0", "false", "off")
CHRONOS_MODEL = os.environ.get("CHRONOS_MODEL", "amazon/chronos-2")
TIMESFM_MODEL = os.environ.get("TIMESFM_MODEL", "google/timesfm-2.5-200m-pytorch")
MAX_CONTEXT = int(os.environ.get("FOUNDATION_MAX_CONTEXT", "512"))

_chronos = None
_timesfm = None
_attempted = {"chronos": False, "timesfm": False}
_lock = threading.Lock()
_result_cache = {}


# ─── lazy loaders ──────────────────────────────────────────────────

def _load_chronos():
    global _chronos
    if _chronos is not None or _attempted["chronos"]:
        return _chronos
    with _lock:
        if _chronos is not None or _attempted["chronos"]:
            return _chronos
        _attempted["chronos"] = True
        try:
            from chronos import Chronos2Pipeline
            _chronos = Chronos2Pipeline.from_pretrained(CHRONOS_MODEL)
            logger.info(f"Chronos-2 loaded: {CHRONOS_MODEL}")
        except Exception as e:
            logger.warning(f"Chronos-2 load failed: {e}")
            _chronos = None
    return _chronos


def _load_timesfm():
    global _timesfm
    if _timesfm is not None or _attempted["timesfm"]:
        return _timesfm
    with _lock:
        if _timesfm is not None or _attempted["timesfm"]:
            return _timesfm
        _attempted["timesfm"] = True
        try:
            from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
            from timesfm import configs
            model = TimesFM_2p5_200M_torch.from_pretrained(
                TIMESFM_MODEL, torch_compile=False
            )
            fc = configs.ForecastConfig(
                max_context=MAX_CONTEXT,
                max_horizon=128,
                normalize_inputs=True,
                per_core_batch_size=1,
                force_flip_invariance=False,
                infer_is_positive=True,
                use_continuous_quantile_head=True,
                fix_quantile_crossing=True,
            )
            model.compile(fc)
            _timesfm = model
            logger.info(f"TimesFM loaded: {TIMESFM_MODEL}")
        except Exception as e:
            logger.warning(f"TimesFM load failed: {e}")
            _timesfm = None
    return _timesfm


def _clamp_conf(conf):
    try:
        return int(max(15, min(95, float(conf))))
    except Exception:
        return 50


def _spread_conf(spread_pct):
    if spread_pct is None:
        return 50
    if spread_pct <= 0.3:
        return 90
    if spread_pct <= 1.0:
        return 80
    if spread_pct <= 2.5:
        return 65
    if spread_pct <= 5.0:
        return 45
    return 30


# ─── forecasters ───────────────────────────────────────────────────

def chronos_forecast(closes, n_steps):
    """Chronos-2 zero-shot forecast. Returns dict or None."""
    if not ENABLE_FOUNDATION:
        return None
    pipe = _load_chronos()
    if pipe is None:
        return None
    try:
        import torch
        arr = np.asarray(closes, dtype=np.float64)
        if len(arr) < 20 or n_steps < 1:
            return None
        ctx = torch.tensor(arr[-MAX_CONTEXT:], dtype=torch.float32).reshape(1, 1, -1)
        out = pipe.predict(ctx, prediction_length=int(n_steps))
        t = out[0]  # (1, n_quantiles, pred_len)
        q = pipe.quantiles
        med_i = list(q).index(0.5)
        q10_i = list(q).index(0.1)
        q90_i = list(q).index(0.9)
        last_price = float(arr[-1])
        med = float(t[0, med_i, -1])
        q10 = float(t[0, q10_i, -1])
        q90 = float(t[0, q90_i, -1])
        change_pct = (med / last_price - 1) * 100 if last_price > 0 else 0.0
        spread_pct = (q90 - q10) / last_price * 100 if last_price > 0 else 0.0
        return {
            "change_pct": round(float(change_pct), 4),
            "predicted_price": float(med),
            "q10": float(q10),
            "q90": float(q90),
            "spread_pct": round(float(spread_pct), 4),
            "confidence": _spread_conf(spread_pct),
        }
    except Exception as e:
        logger.debug(f"Chronos-2 forecast failed: {e}")
        return None


def timesfm_forecast(closes, n_steps):
    """TimesFM 2.5 zero-shot forecast. Returns dict or None."""
    if not ENABLE_FOUNDATION:
        return None
    model = _load_timesfm()
    if model is None:
        return None
    try:
        arr = np.asarray(closes, dtype=np.float64)
        if len(arr) < 20 or n_steps < 1:
            return None
        inputs = [arr[-MAX_CONTEXT:]]
        points, quantiles = model.forecast(horizon=int(n_steps), inputs=inputs)
        last_price = float(arr[-1])
        med = float(points[0][-1])
        q_lo = float(quantiles[0, -1, 1])
        q_hi = float(quantiles[0, -1, 9])
        change_pct = (med / last_price - 1) * 100 if last_price > 0 else 0.0
        spread_pct = (q_hi - q_lo) / last_price * 100 if last_price > 0 else 0.0
        return {
            "change_pct": round(float(change_pct), 4),
            "predicted_price": float(med),
            "q10": float(q_lo),
            "q90": float(q_hi),
            "spread_pct": round(float(spread_pct), 4),
            "confidence": _spread_conf(spread_pct),
        }
    except Exception as e:
        logger.debug(f"TimesFM forecast failed: {e}")
        return None


def _cache_key(closes, n_steps):
    arr = np.asarray(closes, dtype=np.float64)
    tail = tuple(round(float(x), 6) for x in arr[-4:])
    return (n_steps, tail)


def foundation_forecast(closes, n_steps, use_cache=True):
    """
    Combined Chronos-2 + TimesFM forecast.
    Returns dict {chronos, timesfm, change_pct, confidence} or None.
    `change_pct` is the confidence-weighted blend of both models.
    """
    if not ENABLE_FOUNDATION:
        return None
    if use_cache:
        key = _cache_key(closes, n_steps)
        if key in _result_cache:
            return _result_cache[key]

    c_res = chronos_forecast(closes, n_steps)
    t_res = timesfm_forecast(closes, n_steps)

    if c_res is None and t_res is None:
        return None

    parts = []
    for res in (c_res, t_res):
        if res is not None:
            parts.append(res)

    w_sum = sum(p["confidence"] for p in parts) or 1.0
    blend = sum(p["change_pct"] * p["confidence"] for p in parts) / w_sum
    avg_conf = sum(p["confidence"] for p in parts) / len(parts)

    result = {
        "chronos": c_res,
        "timesfm": t_res,
        "change_pct": round(float(blend), 4),
        "confidence": _clamp_conf(avg_conf),
        "num_models": len(parts),
    }
    if use_cache:
        if len(_result_cache) > 128:
            _result_cache.clear()
        _result_cache[key] = result
    return result


def availability():
    return {
        "enabled": ENABLE_FOUNDATION,
        "chronos": _chronos is not None,
        "timesfm": _timesfm is not None,
        "chronos_attempted": _attempted["chronos"],
        "timesfm_attempted": _attempted["timesfm"],
        "chronos_model": CHRONOS_MODEL,
        "timesfm_model": TIMESFM_MODEL,
    }
