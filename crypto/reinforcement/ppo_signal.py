"""
PPO signal helper — bridges a trained PPO model to a simple buy / sell / hold
signal for paper trading (or dashboards), without touching the bot internals.

Flow: fetch recent OHLCV -> build features -> run predict_ppo -> map the last
policy action to a discrete signal.

Usage:
    from reinforcement import ppo_signal
    sig = ppo_signal.generate("BTCUSDT", model_dir="ppo_models/btc_1h",
                              interval="1h", limit=300, lookback=30)
    print(sig["signal"], sig["price"], sig["equity_change_pct"])
"""

import os

import numpy as np

try:
    from . import finrl_ppo as fp
    from . import sb3_ppo as sp
except ImportError:
    import sys
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)
    import finrl_ppo as fp  # noqa: F401
    import sb3_ppo as sp  # noqa: F401

__all__ = [
    "signal_from_action",
    "load_trained",
    "generate",
]

# Hold zone: |action| below this fraction of norm_action is treated as "hold".
HOLD_THRESHOLD_FRAC = 0.15


def signal_from_action(action, norm_action, hold_threshold_frac=HOLD_THRESHOLD_FRAC):
    """Map one raw policy action (for one asset) to 'buy' / 'hold' / 'sell'.

    A positive action means the agent wants to accumulate, a negative one to
    reduce. Very small actions relative to norm_action are "hold".
    """
    if norm_action <= 0:
        raise ValueError("norm_action must be positive")
    mag = abs(float(action)) / float(norm_action)
    if mag < hold_threshold_frac:
        return "hold"
    return "buy" if float(action) > 0 else "sell"


def load_trained(symbol, model_dir, interval="1h", limit=300, lookback=50,
                 net_dimension=128, hold_threshold_frac=HOLD_THRESHOLD_FRAC):
    """Fetch klines, build features and run the trained model on them.

    Returns a dict with:
        symbol, interval, price, equity, equity_change_pct, action (last),
        signal (per-asset discrete), account (curve), model_dir.
    Raises ValueError when there is no trained actor.pth or data is too short.
    """
    if not fp.is_available():
        raise RuntimeError("PPO unavailable: torch is not installed")

    has_sb3 = (os.path.isfile(os.path.join(model_dir, "ppo_sb3.zip"))
               and os.path.isfile(os.path.join(model_dir, "vecnormalize.pkl")))
    has_elegantrl = os.path.isfile(os.path.join(model_dir, "actor.pth"))
    if not has_sb3 and not has_elegantrl:
        raise ValueError(f"No trained model found in {model_dir} "
                         f"(need actor.pth or ppo_sb3.zip + vecnormalize.pkl)")

    df = fp.fetch_klines(symbol, interval, limit)
    if df is None or len(df) < fp.WARMUP_ROWS + lookback + 5:
        raise ValueError(f"Not enough market data for {symbol} {interval} "
                         f"(got {0 if df is None else len(df)} rows)")

    env_params = fp.default_env_params(lookback=lookback)
    norm_action = env_params["norm_action"]

    if has_sb3:
        env_params["reward_mode"] = "relative"
        env_params["action_mode"] = "weight"
        price_array, tech_array = sp.build_rich_features({symbol: df})
        account, actions = sp.predict_ppo_sb3(
            price_array, tech_array, env_params, model_dir, return_actions=True)
        backend = "sb3"
    else:
        price_array, tech_array = fp.build_features({symbol: df})
        account, actions = fp.predict_ppo(
            price_array, tech_array, env_params, model_dir,
            net_dimension=net_dimension, if_log=False, return_actions=True)
        backend = "elegantrl"

    if actions:
        last_action = np.atleast_1d(np.asarray(actions[-1]))
    else:
        last_action = np.zeros(price_array.shape[1])

    price = float(price_array[-1, 0])
    equity = float(account[-1])
    equity_change_pct = (account[-1] / account[0] - 1.0) * 100.0 if account else 0.0

    signals = [signal_from_action(a, norm_action, hold_threshold_frac) for a in last_action]
    primary = signals[0] if len(signals) == 1 else "mixed"

    return {
        "symbol": symbol,
        "interval": interval,
        "backend": backend,
        "price": price,
        "equity": equity,
        "equity_change_pct": float(equity_change_pct),
        "action": last_action.tolist(),
        "signal": primary,
        "signals": signals,
        "account": account,
        "model_dir": model_dir,
        "norm_action": norm_action,
    }


def generate(symbol, model_dir, interval="1h", limit=300, lookback=50,
             net_dimension=128, hold_threshold_frac=HOLD_THRESHOLD_FRAC):
    """One-call helper: same as load_trained, never raises on bad data.

    On any failure returns a dict with 'error' set (safe for bot call sites).
    """
    try:
        return load_trained(symbol, model_dir, interval=interval, limit=limit,
                            lookback=lookback, net_dimension=net_dimension,
                            hold_threshold_frac=hold_threshold_frac)
    except Exception as e:
        return {"symbol": symbol, "interval": interval, "error": str(e)}
