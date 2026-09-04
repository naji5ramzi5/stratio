"""
Regime Gate: trend/volatility filter for directional signals.

Evidence:
  The original deepalpha-freqai claim was that a hard regime gate turned a
  -11% walk-forward into +7% on 2024 BTC data. Our own A/B test
  (gate_ab_test.py, BTC 4 timeframes) found the opposite on OUR data:

      ALL  (no gate)  acc 50.2%  excess -1.6%
      GATE (hard gate) acc 40.0%  excess -13.1%

  A hard regime gate DESTROYS value on a weak directional edge — it blocks
  mean-reversion trades that would have been profitable. The gate therefore
  runs in REPORTING mode by default: it still DETECTS the regime (useful
  diagnostic) but does not block (``tradable`` stays True). Flip
  ``GATE_BLOCKS = True`` only if a future A/B proves blocking helps on a
  stronger edge.

This layer NEVER fabricates: if the features cannot be built the gate returns
status "UNAVAILABLE" and reports the regime as unknown.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("regime_gate")

# Trend must exceed this fraction of price to be a real trend, not noise.
TREND_FRACTION = 0.002
# Only report (not block) when recent volatility exceeds this ATR%.
MAX_ATR_PCT = 5.0
# Master switch: does the gate actually block trades? Default False because our
# A/B proves blocking hurts on a weak directional edge (see docstring).
GATE_BLOCKS = False
# Minimum confidence required for a signal to trade against a flat regime.
CONDITIONAL_MIN_CONF = 70.0
# Minimum bars before the EMAs are considered warmed up.
MIN_BARS = 200


def detect_regime(df, trend_fraction=TREND_FRACTION):
    """Classify the trend regime from a kline DataFrame with a ``close`` column.

    Returns a dict with trend (UP/DOWN/FLAT), the EMA pair and the ATR%.
    """
    if df is None or len(df) < 2:
        return {"status": "missing", "trend": "UNKNOWN", "reason": "no data"}
    close = df["close"].astype(float)
    n = len(close)
    ema24 = close.ewm(span=24).mean()
    ema96 = close.ewm(span=96).mean()

    out = {"status": "ok", "ema24": float(ema24.iloc[-1]), "ema96": float(ema96.iloc[-1]),
           "n_bars": n, "trend": "FLAT"}
    if n < MIN_BARS:
        out["status"] = "warmup"
        out["reason"] = f"only {n} bars (< {MIN_BARS})"
        return out

    rel = (ema24.iloc[-1] - ema96.iloc[-1]) / max(abs(ema96.iloc[-1]), 1e-12)
    out["trend_delta_pct"] = round(float(rel * 100), 3)
    if rel > trend_fraction:
        out["trend"] = "UP"
    elif rel < -trend_fraction:
        out["trend"] = "DOWN"

    if "high" in df.columns and "low" in df.columns:
        h, l, c = df["high"].astype(float), df["low"].astype(float), close
        pc = c.shift(1)
        tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        atr = float(tr.rolling(14).mean().iloc[-1])
        out["atr_pct"] = round(atr / float(c.iloc[-1]) * 100, 3)
    else:
        out["atr_pct"] = float("nan")
    return out


def _interval_for_tf(tf_hours):
    if tf_hours <= 2:
        return "5m"
    if tf_hours <= 6:
        return "15m"
    if tf_hours <= 8:
        return "30m"
    return "1h"


def _ml_direction(ml_result):
    if not ml_result:
        return 0
    try:
        p = float(ml_result.get("predicted_change_pct", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0
    if abs(p) < 0.05:
        return 0
    return 1 if p > 0 else -1


def gate_signal(symbol, tf_hours, ml_result, df=None, gate_blocks=GATE_BLOCKS,
                conditional_min_conf=CONDITIONAL_MIN_CONF, fetch_features=True):
    """Apply the regime gate to an ML result.

    ``df`` is the feature frame for the interval that matches ``tf_hours``;
    when omitted the cached features are used (may be None -> UNAVAILABLE).
    ``fetch_features=False`` disables the network path (used by offline tests).

    When ``gate_blocks`` is False (default, see ``GATE_BLOCKS``) the gate still
    detects and reports the regime but NEVER sets ``tradable`` to False — our
    A/B proved that blocking destroys value on a weak directional edge. Flip
    ``gate_blocks=True`` only if a future A/B proves otherwise.

    Returns a dict with status in PASS / BLOCKED / CONDITIONAL / NEUTRAL /
    UNAVAILABLE, a boolean ``tradable`` and the reason.
    """
    base = {"symbol": symbol, "tf_hours": tf_hours}
    if df is None:
        if not fetch_features:
            return {**base, "status": "UNAVAILABLE", "tradable": False,
                    "reason": "no df supplied and fetch disabled"}
        try:
            from ml_trainer import get_cached_features
            df = get_cached_features(symbol, _interval_for_tf(tf_hours), 1500)
        except Exception as exc:
            logger.debug("regime gate feature build failed: %s", exc)
            return {**base, "status": "UNAVAILABLE", "tradable": False,
                    "reason": "features unavailable"}
    if df is None or len(df) < 2:
        return {**base, "status": "UNAVAILABLE", "tradable": False,
                "reason": "no feature data"}

    regime = detect_regime(df)
    direction = _ml_direction(ml_result)
    conf = float((ml_result or {}).get("confidence", 0.0) or 0.0)
    out = {**base, "regime": regime}

    if regime.get("status") == "warmup":
        status = "UNAVAILABLE"
        return {**base, "status": status, "tradable": gate_blocks,
                "reason": regime.get("reason", "warmup"), "regime": regime}

    atr = regime.get("atr_pct")
    high_vol = atr is not None and not np.isnan(atr) and atr > MAX_ATR_PCT
    if high_vol and gate_blocks:
        return {**base, "status": "BLOCKED", "tradable": False,
                "reason": f"volatility too high (ATR% {atr} > {MAX_ATR_PCT})",
                "regime": regime}

    if direction == 0:
        status = "NEUTRAL"
        return {**base, "status": status, "tradable": gate_blocks,
                "reason": "ML prediction near zero", "regime": regime}

    trend = regime["trend"]
    if trend == "UP" and direction == 1:
        return {**base, "status": "PASS", "tradable": True,
                "reason": "long with uptrend", "regime": regime}
    if trend == "DOWN" and direction == -1:
        return {**base, "status": "PASS", "tradable": True,
                "reason": "short with downtrend", "regime": regime}
    if trend in ("UP", "DOWN"):
        if gate_blocks:
            return {**base, "status": "BLOCKED", "tradable": False,
                    "reason": f"signal opposes {trend} trend", "regime": regime}
        return {**base, "status": "COUNTER_TREND", "tradable": True,
                "reason": f"signal opposes {trend} trend (gate_blocks=False, reported not blocked)",
                "regime": regime}

    # flat regime
    if conf >= conditional_min_conf:
        return {**base, "status": "CONDITIONAL", "tradable": True,
                "reason": f"flat regime, confidence {conf:.0f} >= {conditional_min_conf:.0f}",
                "regime": regime}
    if gate_blocks:
        return {**base, "status": "BLOCKED", "tradable": False,
                "reason": f"flat regime and confidence {conf:.0f} < {conditional_min_conf:.0f}",
                "regime": regime}
    return {**base, "status": "FLAT_LOW_CONF", "tradable": True,
            "reason": f"flat regime, confidence {conf:.0f} < {conditional_min_conf:.0f} (reported, not blocked)",
            "regime": regime}
