"""
Hybrid signal — fuse the PPO agent's latest action with the existing
predictor engine (predict_price_movement) into a single trade recommendation
compatible with paper_trading.PaperTradingEngine.open_trade().

Rationale: the predictor already blends technicals + order flow + ML +
foundation models (per the project's strong prediction bots), while the PPO
agent adds a trained action policy.  When both agree the signal is boosted;
when the PPO abstains (|action| below the hold zone) the predictor dominates;
when they conflict the score cancels toward HOLD.

Usage:
    from reinforcement import hybrid_signal
    rec = hybrid_signal.generate("BTCUSDT", tf_hours=24,
                                 model_dir="crypto/ppo_models/BTCUSDT_1h")
    # -> {signal, side, confidence, size_pct, stop_loss_pct, take_profit_pct,
    #     ppo, predictor, agreement, ...} or {"error": ...}
"""

import os
import sys

import numpy as np

try:
    from . import ppo_signal as ps
except ImportError:
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if _pkg_dir not in sys.path:
        sys.path.insert(0, _pkg_dir)
    import ppo_signal as ps  # noqa: F401

__all__ = ["ppo_score", "pred_score", "fuse", "generate"]

PPO_HOLD_ZONE = 0.15  # |action| below this fraction of norm_action = abstain
DEFAULT_PPO_MODEL_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "ppo_models", "BTCUSDT_1h_sb3"))


def _ensure_crypto_path():
    crypto_dir = os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".."))
    if crypto_dir not in sys.path:
        sys.path.insert(0, crypto_dir)


def _interval_for_tf(tf_hours):
    if tf_hours <= 2:
        return "5m"
    if tf_hours <= 6:
        return "15m"
    if tf_hours <= 8:
        return "30m"
    return "1h"


def ppo_score(sig):
    """Continuous PPO score in [-1, 1] from the last raw action (0 = abstain)."""
    if not sig or "error" in sig or not sig.get("action"):
        return None
    action = np.atleast_1d(sig["action"])[0]
    norm_action = sig.get("norm_action", 1.0)
    if not norm_action:
        return None
    score = float(np.clip(float(action) / float(norm_action), -1.0, 1.0))
    return 0.0 if abs(score) < PPO_HOLD_ZONE else score


def pred_score(pred):
    """Continuous predictor score in [-1, 1] (3% move = full strength)."""
    if not pred or "error" in pred:
        return None
    pct = pred.get("predicted_change_pct")
    if pct is None:
        return None
    return float(np.clip(float(pct) / 3.0, -1.0, 1.0))


def fuse(ppo_sig, pred, ppo_weight=0.5):
    """Combine a ppo_signal dict and a predictor dict into a recommendation.

    Returns a dict with: signal, side, confidence, size_pct, stop_loss_pct,
    take_profit_pct, fused_score, agreement, ppo, predictor.
    """
    p_s = ppo_score(ppo_sig)
    d_s = pred_score(pred)

    if p_s is None and d_s is None:
        return {"error": "no usable signal from PPO or predictor"}

    if p_s is None:
        fused = d_s
        agreement = "predictor"
    elif d_s is None:
        fused = p_s
        agreement = "ppo"
    elif abs(p_s) < PPO_HOLD_ZONE:
        fused = d_s
        agreement = "predictor"  # PPO abstains
    else:
        fused = float(np.clip(ppo_weight * p_s + (1.0 - ppo_weight) * d_s, -1.0, 1.0))
        agreement = "both" if np.sign(p_s) == np.sign(d_s) else "conflict"

    if fused >= 0.6:
        signal = "STRONG BUY"
    elif fused >= 0.25:
        signal = "BUY"
    elif fused <= -0.6:
        signal = "STRONG SELL"
    elif fused <= -0.25:
        signal = "SELL"
    else:
        signal = "HOLD"

    side = "LONG" if fused > 0.25 else ("SHORT" if fused < -0.25 else None)

    base_conf = pred.get("confidence", 50) if (pred and "error" not in pred) else 50
    if agreement == "both":
        base_conf += 12
    elif agreement == "conflict":
        base_conf *= 0.55
    elif agreement == "ppo":
        base_conf = 40 + abs(fused) * 55
    confidence = int(max(15, min(95, base_conf)))

    size_pct = min(confidence / 200, 0.25)
    stop_loss_pct = max(0.5, 5 - confidence / 20)
    take_profit_pct = stop_loss_pct * 2

    rec = {
        "signal": signal,
        "side": side,
        "confidence": confidence,
        "size_pct": round(size_pct, 4),
        "stop_loss_pct": round(stop_loss_pct, 2),
        "take_profit_pct": round(take_profit_pct, 2),
        "fused_score": round(fused, 4),
        "agreement": agreement,
        "ppo": (None if ppo_sig is None or "error" in ppo_sig else {
            "signal": ppo_sig.get("signal"),
            "action": ppo_sig.get("action"),
            "equity_change_pct": round(ppo_sig.get("equity_change_pct", 0.0), 3),
            "score": round(p_s, 4) if p_s is not None else None,
        }),
        "predictor": (None if pred is None or "error" in pred else {
            "predicted_change_pct": pred.get("predicted_change_pct"),
            "grade": pred.get("grade"),
            "confidence": pred.get("confidence"),
            "market_regime": pred.get("market_regime"),
            "entry_timing": pred.get("entry_timing"),
            "ml_prediction": pred.get("ml_prediction"),
            "foundation_prediction": pred.get("foundation_prediction"),
            "reasons": pred.get("reasons"),
        }),
    }
    return rec


def generate(symbol, tf_hours=24, model_dir=None, interval=None, limit=300,
             lookback=50, net_dimension=128, ppo_weight=0.5):
    """One-call hybrid: predictor engine + trained PPO model -> recommendation.

    Never raises on bad data; returns a dict (possibly with 'error').
    """
    _ensure_crypto_path()
    model_dir = model_dir or DEFAULT_PPO_MODEL_DIR
    if interval is None:
        interval = _interval_for_tf(tf_hours)

    pred = None
    try:
        from predictor import predict_price_movement
        res = predict_price_movement(symbol, tf_hours)
        if res and "error" not in res:
            pred = res
    except Exception as e:
        pred = {"error": str(e)}

    ppo_sig = ps.generate(symbol, model_dir, interval=interval, limit=limit,
                          lookback=lookback, net_dimension=net_dimension)

    rec = fuse(ppo_sig, pred, ppo_weight=ppo_weight)
    rec["symbol"] = symbol
    rec["timeframe_hours"] = tf_hours
    rec["interval"] = interval
    rec["current_price"] = (
        ppo_sig.get("price") if (ppo_sig and "error" not in ppo_sig)
        else (pred.get("current_price") if pred and "error" not in pred else None))
    return rec
