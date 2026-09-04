"""Regime & Volatility Predictor — a fundamentally different approach.

The old directional model (predicting +3% / -3% moves) is dead: 45-48% accuracy
at every horizon because short-term crypto direction is near-random.

This model predicts what IS predictable:
  1. Volatility regime: will the next N bars be high-vol or low-vol?
  2. Market regime: TRENDING vs MEAN-REVERTING vs VOLATILE?
  3. Trade quality score: given current conditions, what's the probability
     that a mean-reversion (pairs) trade profits?

These are the inputs that actually matter for the pairs edge:
  - Trade MORE when regime is mean-reverting and vol is moderate
  - Trade LESS/SKIP when regime is trending or vol is exploding
  - Size by predicted volatility

Output: a single "regime_score" 0..100 that gates the pairs book.
"""
import json
import logging
import os
import pickle
import numpy as np
import pandas as pd

from settings import MODELS_DIR, CALIBRATION_PATH, RANDOM_STATE, seed_everything
from utils.state_store import load_json, save_json

logger = logging.getLogger("RegimePredictor")
MODEL_PATH = os.path.join(MODELS_DIR, "regime_v1.pkl")

_REGIME_LABELS = {0: "TRENDING", 1: "MEAN_REVERTING", 2: "VOLATILE"}


def compute_regime_features(klines):
    """Compute features that predict regime/volatility (not direction).

    These are DIFFERENT from the old directional features: they measure
    the SHAPE of price action (vol persistence, mean-reversion strength,
    trend consistency), not where price is going."""
    if not klines or len(klines) < 50:
        return None
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_vol",
        "taker_buy_quote", "ignore",
    ])
    for c in ["open", "high", "low", "close", "volume", "trades"]:
        df[c] = df[c].astype(float)

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    n = len(close)
    if n < 50:
        return None

    # --- Returns & volatility ---
    rets = np.diff(np.log(close + 1e-12))
    vol_5 = np.std(rets[-5:]) * np.sqrt(24 * 365) if len(rets) >= 5 else 0.5
    vol_20 = np.std(rets[-20:]) * np.sqrt(24 * 365) if len(rets) >= 20 else 0.5
    vol_50 = np.std(rets[-50:]) * np.sqrt(24 * 365) if len(rets) >= 50 else 0.5
    vol_ratio = vol_5 / max(vol_50, 1e-9)  # vol acceleration

    # --- Trend consistency (how persistent is the trend?) ---
    # Positive autocorrelation in returns = trending
    # Negative autocorrelation = mean-reverting
    if len(rets) >= 20:
        autocorr = np.corrcoef(rets[:-1], rets[1:])[0, 1]
        if np.isnan(autocorr):
            autocorr = 0.0
    else:
        autocorr = 0.0

    # ADX-like: directional consistency
    ups = np.sum(rets[-20:] > 0) if len(rets) >= 20 else 10
    downs = 20 - ups
    dx = abs(ups - downs) / 20.0  # 0 = balanced (mean-reverting), 1 = all one way (trending)

    # --- Mean-reversion strength (Hurst-like) ---
    # Variance ratio: var(2-period) / (2 * var(1-period))
    # < 1 = mean-reverting, > 1 = trending
    if len(rets) >= 20:
        var_short = np.var(rets[-10:])
        rets_2 = rets[-20::2]  # every-other bar = 2-period returns
        var_long = np.var(rets_2) if len(rets_2) > 1 else var_short
        var_ratio = var_long / max(2 * var_short, 1e-9)
    else:
        var_ratio = 1.0

    # --- Volatility of volatility (vol clustering) ---
    vol_series = [np.std(rets[i:i + 5]) for i in range(0, min(50, len(rets) - 5), 5)]
    vol_of_vol = np.std(vol_series) if len(vol_series) > 1 else 0.0

    # --- Range expansion ---
    tr = np.maximum(high[1:] - low[1:],
                    np.maximum(abs(high[1:] - close[:-1]),
                               abs(low[1:] - close[:-1])))
    atr_14 = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
    atr_ratio = atr_14 / max(np.mean(tr), 1e-9)

    # --- Volume trend ---
    vol_mean = np.mean(df["volume"].values[-20:])
    vol_trend = np.mean(df["volume"].values[-5:]) / max(vol_mean, 1e-9)

    # --- Skewness of returns (fat tails indicator) ---
    if len(rets) >= 20:
        skew = float(np.mean(((rets[-20:] - np.mean(rets[-20:])) / (np.std(rets[-20:]) + 1e-9)) ** 3))
    else:
        skew = 0.0

    features = np.array([
        vol_5, vol_20, vol_50, vol_ratio, autocorr, dx, var_ratio,
        vol_of_vol, atr_ratio, vol_trend, skew,
        np.log1p(ups), np.log1p(downs),
    ], dtype=float)

    names = [
        "vol_5", "vol_20", "vol_50", "vol_ratio", "autocorr", "directional_dx",
        "var_ratio", "vol_of_vol", "atr_ratio", "vol_trend", "return_skew",
        "log_ups", "log_downs",
    ]
    return features, names, {
        "vol_5": round(vol_5, 4), "vol_20": round(vol_20, 4),
        "vol_ratio": round(vol_ratio, 3), "autocorr": round(autocorr, 3),
        "directional_dx": round(dx, 3), "var_ratio": round(var_ratio, 3),
        "regime_autolabel": _autolabel_regime(vol_ratio, autocorr, vol_ratio),
    }


def _autolabel_regime(autocorr, dx, vol_ratio, vol_20=0.5):
    """Auto-label regime for training: 0=trending, 1=mean-reverting, 2=volatile."""
    if vol_20 > 1.0 or vol_ratio > 1.5:
        return 2
    if dx > 0.65 and autocorr > -0.08:
        return 0
    return 1


def predict_regime(klines):
    """Predict current regime + vol forecast. Returns a dict.

    This is what replaces the dead directional prediction."""
    result = compute_regime_features(klines)
    if result is None:
        return {"status": "error", "note": "insufficient data"}
    features, names, diagnostics = result

    # Rule-based regime (no model file needed — these are causal statistics)
    vol_ratio = diagnostics["vol_ratio"]
    autocorr = diagnostics["autocorr"]
    dx = diagnostics["directional_dx"]
    vol_20 = diagnostics["vol_20"]

    # VOLATILE: very high absolute vol
    if vol_20 > 1.0 or vol_ratio > 1.5:
        regime_id = 2
    # TRENDING: most returns same direction
    elif dx > 0.65 and autocorr > -0.08:
        regime_id = 0
    else:
        regime_id = 1  # MEAN_REVERTING

    regime = _REGIME_LABELS[regime_id]

    # Volatility forecast (next 24h) — simple but effective
    vol_current = diagnostics["vol_20"]
    # Vol is persistent: high vol tends to stay high for a bit, then revert
    vol_forecast = vol_current * 0.7 + 0.5 * 0.3  # reverts toward 0.5 annualized

    # Regime score 0..100: how favorable for pairs mean-reversion?
    score = 50.0
    if regime == "MEAN_REVERTING":
        score += 30
    elif regime == "VOLATILE":
        score -= 25
    elif regime == "TRENDING":
        score -= 15
    # Moderate vol is best for pairs (not too dead, not exploding)
    if 0.3 <= vol_current <= 0.8:
        score += 10
    elif vol_current > 1.2:
        score -= 15
    score = float(np.clip(score, 0, 100))

    # Recommendation
    if score >= 70:
        action = "TRADE"  # favorable for pairs
    elif score >= 50:
        action = "CAUTION"  # ok but smaller size
    else:
        action = "SKIP"  # trending or too volatile — pairs edge is weak

    return {
        "status": "ok",
        "regime": regime,
        "score": round(score, 1),
        "action": action,
        "vol_current": round(vol_current, 4),
        "vol_forecast_24h": round(vol_forecast, 4),
        "autocorr": autocorr,
        "diagnostics": diagnostics,
    }


def regime_score_for_pair(symbol_a, symbol_b, fetch_fn):
    """Combined regime score for a pair: average of both assets' regimes,
    penalized if they disagree strongly."""
    ka = fetch_fn(symbol_a, "1h", 200)
    kb = fetch_fn(symbol_b, "1h", 200)
    ra = predict_regime(ka)
    rb = predict_regime(kb)
    if ra.get("status") != "ok" or rb.get("status") != "ok":
        return None
    combined = (ra["score"] + rb["score"]) / 2
    # penalize if regimes disagree (one trending, one mean-reverting)
    if ra["regime"] != rb["regime"]:
        combined *= 0.7
    return {
        "pair": f"{symbol_a}/{symbol_b}",
        "score": round(combined, 1),
        "regime_a": ra["regime"], "regime_b": rb["regime"],
        "action_a": ra["action"], "action_b": rb["action"],
    }


if __name__ == "__main__":
    # quick demo on BTC
    from data_loader.binance_ohlcv import fetch_klines
    k = fetch_klines("BTCUSDT", "1h", 200)
    r = predict_regime(k)
    print(json.dumps(r, indent=2))
