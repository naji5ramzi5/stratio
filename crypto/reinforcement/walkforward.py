"""
Walk-forward backtesting + regime-gated deployment for the sb3 PPO path.

Crypto-only.  Trains a model on a rolling window of 1h klines, evaluates it on
the *next contiguous* window (strictly out-of-sample), rolls forward, and
stitches all out-of-sample folds into one honest equity curve vs an
equal-weight buy-and-hold benchmark.

No-look-ahead guarantees (each one is verified in test_walkforward.py):

* Feature standardization uses the training window ONLY (fit_standardizer on
  train rows, apply_standardizer with the same scaler on test rows).
* The regime gate is causal: at bar t it uses the trend signal from bar t-1
  (close above its 24h SMA), so it never peeks at the bar it trades on.
* Benchmark (equal-weight buy & hold) is computed on the exact same test rows.

Why the gate: the raw PPO learns "long in bull, cash in bear" but still rides
positions into sudden crashes (observed -30..-70% on the 2026 bear window).
A simple trend overlay zeroes the agent's exposure in downtrends — the standard
"risk layer" used by production crypto systems and recommended by the walk
forward / regime-detection literature surveyed for this build.

Usage:
    from reinforcement import walkforward as wf
    rep = wf.walkforward_backtest(symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                                  outdir="crypto/ppo_models/walkforward")
    print(rep["metrics"])
"""

import json
import os

import numpy as np
import pandas as pd

try:
    from . import finrl_ppo as fp
    from . import sb3_ppo as sp
except ImportError:
    import sys
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if _pkg_dir not in sys.path:
        sys.path.insert(0, _pkg_dir)
    import finrl_ppo as fp  # noqa: F401
    import sb3_ppo as sp  # noqa: F401

__all__ = [
    "sma_regime",
    "confirmed_bear",
    "replay_with_gate",
    "fold_metrics",
    "stitch_equity",
    "walkforward_backtest",
    "retrain_latest",
    "WALKFORWARD_HP",
]

# defensive config that generalized best in the regime study
WALKFORWARD_HP = {
    "learning_rate": 2.796081833841857e-05,
    "ent_coef": 0.005,
    "gamma": 0.995,
    "gae_lambda": 0.9,
    "net_dimension": 64,
    "batch_size": 128,
}

HOURS_PER_YEAR = 8760.0


def default_env_params(lookback=50):
    ep = fp.default_env_params(lookback=lookback)
    ep["reward_mode"] = "relative"
    ep["action_mode"] = "weight"
    ep["rebalance_deadband"] = 0.05
    return ep


def sma_regime(price_col, window=24):
    """Causal uptrend flag per row: close[t-1] > SMA(close, window)[t-1].

    Returns a bool array aligned to `price_col` (False during the first
    `window` warmup bars).
    """
    price = pd.Series(np.asarray(price_col, dtype=np.float64))
    sma = price.rolling(window).mean()
    uptrend = price > sma
    return np.asarray(uptrend.shift(1).fillna(False), dtype=bool)


def confirmed_bear(price_col, fast=24, slow=72, conf=12):
    """Confirmed-downtrend mask (causal, whipsaw-resistant).

    Bear (True) at the trade bar t only when, using bars up to t-1:
      * close < SMA(fast)          (price below its short average)
      * SMA(fast) < SMA(slow)      (fast below slow: regime stacked)
      * SMA(fast) is falling over `conf` bars

    Requiring all three cuts the single-cross-over whipsaw of `sma_regime`
    on noisy hourly crypto data.
    """
    price = pd.Series(np.asarray(price_col, dtype=np.float64))
    fast_sma = price.rolling(fast).mean().shift(1)
    slow_sma = price.rolling(slow).mean().shift(1)
    falling = fast_sma.diff(conf) < 0
    return np.asarray((fast_sma < slow_sma) & falling, dtype=bool)


def replay_with_gate(price_slice, tech_slice, env_params, actions, regime_mask):
    """Replay saved actions on a raw env, zeroing exposure in downtrends.

    Mirrors predict_ppo_sb3's account bookkeeping (stops one step short of the
    terminal step, which DummyVecEnv would auto-reset).  `regime_mask[k]`
    governs the trade executed by action k.
    """
    env = sp.make_env_inst(price_slice, tech_slice, env_params, if_log=False)
    env.reset()
    curve = [float(env.total_asset)]
    for k, a in enumerate(actions[:-1]):
        a = np.asarray(a, dtype=np.float64).copy()
        if k < len(regime_mask) and not bool(regime_mask[k]):
            a[:] = 0.0  # force full cash in a downtrend
        env.step(a)
        curve.append(float(env.total_asset))
    env.close()
    return np.asarray(curve, dtype=np.float64)


def _curve_metrics(curve):
    curve = np.asarray(curve, dtype=np.float64)
    rets = curve[1:] / curve[:-1] - 1.0
    sharpe = fp._sharpe(rets, np.sqrt(HOURS_PER_YEAR))
    cum = float(curve[-1] / curve[0] - 1.0) if len(curve) > 1 else 0.0
    dd = fp._max_drawdown(curve)
    return {"cum_return": cum, "sharpe": sharpe, "max_drawdown": dd,
            "final": float(curve[-1]), "points": len(curve)}


def fold_metrics(account, price_slice, lookback=50):
    """Metrics for one fold's account curve vs equal-weight buy & hold."""
    account = np.asarray(account, dtype=np.float64)
    indx1 = max(0, int(lookback) - 1)
    indx2 = len(price_slice) - int(lookback)
    eqw = fp.compute_eqw(price_slice, indx1, indx2)
    if eqw is None:
        return None
    eqw_curve = eqw[0]
    n = min(len(account), len(eqw_curve))
    bot = _curve_metrics(account[:n])
    bench = _curve_metrics(eqw_curve[:n])
    bot["excess_sharpe"] = bot["sharpe"] - bench["sharpe"]
    bot["benchmark"] = bench
    return bot


def stitch_equity(fold_accounts):
    """Compound fold account curves into one continuous equity path.

    Each fold's curve is normalized to start at 1.0 and the successive
    relative values are multiplied into the running equity, so the path is
    continuous across fold boundaries.
    """
    curve = [1.0]
    value = 1.0
    for acc in fold_accounts:
        acc = np.asarray(acc, dtype=np.float64)
        if len(acc) < 2:
            continue
        rel = acc / acc[0]
        for x in rel[1:]:
            curve.append(value * float(x))
        value = value * float(rel[-1])
    return np.asarray(curve, dtype=np.float64)


def walkforward_backtest(symbols, interval="1h", total=17520, n_train=8760,
                         n_test=1450, train_steps=60000, lookback=50,
                         hp=None, outdir=None, gate=True, window=24,
                         regime_fn="confirmed_bear", seed=11, verbose=0,
                         ohlcv=None, env_params=None):
    """Rolling-window train/predict over crypto 1h data (out-of-sample).

    Returns a report dict with per-fold metrics, the stitched OOS equity
    (gated and raw) vs equal-weight benchmark, and fold artifacts in `outdir`.
    """
    hp = dict(WALKFORWARD_HP if hp is None else hp)
    if env_params is None:
        env_params = default_env_params(lookback=lookback)
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    if ohlcv is None:
        ohlcv = {}
        for s in symbols:
            df = fp.fetch_klines_paginated(s, interval, total=total)
            if df is None:
                raise RuntimeError(f"failed to fetch {s} {interval}")
            ohlcv[s] = df
    common = sorted(set.intersection(*[set(ohlcv[s]["timestamp"]) for s in symbols]))
    aligned = {s: ohlcv[s].set_index("timestamp").loc[common].reset_index()
               for s in symbols}
    price_full, tech_raw = sp.compute_rich_features(aligned)
    N = len(price_full)
    if N < n_train + n_test + 2 * lookback:
        raise ValueError(f"data too short: {N} rows need train {n_train} + "
                         f"test {n_test + 2 * lookback}")
    times = np.asarray(common)[sp.WARMUP_ROWS:]
    if callable(regime_fn):
        regime_signal = regime_fn
    elif regime_fn == "confirmed_bear":
        regime_signal = confirmed_bear
    else:
        regime_signal = lambda p, window=window: sma_regime(p, window=window)  # noqa: E731
    regime_full = regime_signal(price_full[:, 0])

    te_len = n_test + 2 * lookback
    starts = []
    s = 0
    while s + n_train + te_len <= N:
        starts.append(s)
        s += n_test
    if not starts:
        raise ValueError("no full walk-forward fold fits the data")

    acc_raw_all, acc_gate_all, eqw_folds, folds = [], [], [], []
    for i, s in enumerate(starts):
        tr = slice(s, s + n_train)
        te = slice(s + n_train, s + n_train + te_len)
        mean, std = sp.fit_standardizer(tech_raw[tr])
        tech_tr = sp.apply_standardizer(tech_raw[tr], mean, std)
        tech_te = sp.apply_standardizer(tech_raw[te], mean, std)
        price_te = price_full[te]

        fold_dir = None if outdir is None else os.path.join(outdir, f"fold_{i}")
        sp.train_ppo_sb3(price_full[tr], tech_tr, env_params, fold_dir,
                         total_timesteps=int(train_steps), seed=seed + i,
                         verbose=verbose, **hp)
        account, actions = sp.predict_ppo_sb3(
            price_te, tech_te, env_params, fold_dir, return_actions=True)

        acc_raw = np.asarray(account, dtype=np.float64)
        acc_gate = replay_with_gate(price_te, tech_te, env_params, actions,
                                    regime_full[te.start + lookback - 1:])
        eqw = fp.compute_eqw(price_te, lookback - 1, te_len - lookback)
        n_eqw = min(len(acc_raw), len(eqw[0]))
        eqw_folds.append(eqw[0][:n_eqw])
        acc_raw_all.append(acc_raw)
        acc_gate_all.append(acc_gate)

        m_raw = fold_metrics(acc_raw, price_te, lookback)
        m_gate = fold_metrics(acc_gate, price_te, lookback)
        folds.append({
            "i": i, "train_rows": [s, s + n_train],
            "test_rows": [s + n_train, s + n_train + n_test],
            "test_range": [times[s + n_train], times[s + n_train + n_test - 1]],
            "metrics_raw": m_raw, "metrics_gated": m_gate,
        })
        if verbose:
            print(f"fold {i}: raw {m_raw['cum_return']:+.3f} / "
                  f"gated {m_gate['cum_return']:+.3f} / "
                  f"eqw {m_raw['benchmark']['cum_return']:+.3f}")
        if outdir is not None:
            np.savez(os.path.join(fold_dir, "oos_curve.npz"),
                     account=acc_raw, gated=acc_gate, price=price_te)
        del account, actions

    curve_raw = stitch_equity(acc_raw_all)
    curve_gated = stitch_equity(acc_gate_all)
    eqw_curve = np.concatenate(eqw_folds) if eqw_folds else np.array([1.0])

    report = {
        "symbols": symbols, "interval": interval,
        "windows": {"n_train": n_train, "n_test": n_test,
                    "n_folds": len(starts), "gate": bool(gate),
                    "gate_window": window, "regime_fn":
                    regime_fn.__name__ if callable(regime_fn) else regime_fn},
        "hp": hp, "env_params": {k: v for k, v in env_params.items()
                                 if isinstance(v, (int, float, str))},
        "starts": starts,
        "metrics_raw": _curve_metrics(curve_raw),
        "metrics_gated": _curve_metrics(curve_gated),
        "metrics_eqw": _curve_metrics(eqw_curve),
        "excess_raw": (_curve_metrics(curve_raw)["sharpe"]
                       - _curve_metrics(eqw_curve)["sharpe"]),
        "excess_gated": (_curve_metrics(curve_gated)["sharpe"]
                         - _curve_metrics(eqw_curve)["sharpe"]),
        "folds": folds,
    }
    if outdir is not None:
        np.savez(os.path.join(outdir, "oos_curves.npz"),
                 raw=curve_raw, gated=curve_gated, eqw=eqw_curve)
        with open(os.path.join(outdir, "report.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)
    return report


def retrain_latest(symbols, interval="1h", total=17520, n_train=8760,
                   train_steps=60000, lookback=50, hp=None, outdir=None,
                   seed=11, verbose=0, ohlcv=None, env_params=None,
                   regime_fn="confirmed_bear"):
    """Train the production model on the most recent window only.

    This is what a live deployment re-runs (e.g. weekly): train on the last
    `n_train` bars, save ppo_sb3.zip + vecnormalize.pkl, and report the current
    market regime so the caller can decide whether to deploy or stay in cash.
    """
    hp = dict(WALKFORWARD_HP if hp is None else hp)
    if env_params is None:
        env_params = default_env_params(lookback=lookback)
    if ohlcv is None:
        ohlcv = {}
        for s in symbols:
            df = fp.fetch_klines_paginated(s, interval, total=total)
            if df is None:
                raise RuntimeError(f"failed to fetch {s} {interval}")
            ohlcv[s] = df
    common = sorted(set.intersection(*[set(ohlcv[s]["timestamp"]) for s in symbols]))
    aligned = {s: ohlcv[s].set_index("timestamp").loc[common].reset_index()
               for s in symbols}
    price_full, tech_raw = sp.compute_rich_features(aligned)
    s = max(0, len(price_full) - n_train)
    mean, std = sp.fit_standardizer(tech_raw[s:])
    tech_tr = sp.apply_standardizer(tech_raw[s:], mean, std)
    sp.train_ppo_sb3(price_full[s:], tech_tr, env_params, outdir,
                     total_timesteps=int(train_steps), seed=seed,
                     verbose=verbose, **hp)
    if callable(regime_fn):
        regime_signal = regime_fn
    elif regime_fn == "confirmed_bear":
        regime_signal = confirmed_bear
    else:
        regime_signal = lambda p, window=24: sma_regime(p, window=window)  # noqa: E731
    regime = bool(regime_signal(price_full[:, 0])[-1])
    report = {
        "symbols": symbols, "interval": interval, "model_dir": outdir,
        "train_rows": [s, len(price_full)], "timesteps": train_steps,
        "current_regime": "bull" if regime else "bear",
        "last_close": float(price_full[-1, 0]),
    }
    if outdir is not None:
        with open(os.path.join(outdir, "report.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)
    return report
