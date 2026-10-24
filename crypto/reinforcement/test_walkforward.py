"""
Tests for reinforcement/walkforward.py (walk-forward + regime gate).
Run: python reinforcement/test_walkforward.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from walkforward import (confirmed_bear, replay_with_gate, sma_regime,
                         stitch_equity, walkforward_backtest, retrain_latest)


def _ohlcv(n=800, seed=3, drift=0.0005, trend=None):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    if trend is not None:
        close = np.asarray(trend, dtype=np.float64)
    else:
        close = 100 * np.exp(np.cumsum(np.full(n, drift) + rng.normal(0, 0.004, n)))
    df = pd.DataFrame({
        "timestamp": (pd.Timestamp("2024-01-01") + pd.to_timedelta(t, unit="h")).astype("int64") // 10**6,
        "open": close, "high": close * 1.004, "low": close * 0.996,
        "close": close, "volume": rng.uniform(10, 50, n),
    })
    return df


def test_confirmed_bear_causal():
    n = 300
    trend = np.concatenate([np.linspace(100, 200, 150), np.linspace(200, 120, 150)])
    r = confirmed_bear(trend, fast=12, slow=36, conf=6)
    assert r.dtype == bool and r.shape == (n,)
    # no signal before the slow warmup
    assert not r[:36].any()
    # falling half must be flagged as bear
    assert r[180:].mean() > 0.5
    # rising half must be almost never bear
    assert r[36:140].mean() < 0.2
    print("OK test_confirmed_bear_causal")


def test_sma_regime_causal():
    n = 300
    # a rising then falling series
    trend = np.concatenate([np.linspace(100, 200, 150), np.linspace(200, 120, 150)])
    r = sma_regime(trend, window=24)
    assert r.dtype == bool and r.shape == (n,)
    # uptrend must be True only after warmup (first `window` bars False)
    assert not r[:24].any()
    # rising half should be mostly uptrend
    assert r[24:140].mean() > 0.6
    # falling half should be mostly downtrend
    assert r[160:].mean() < 0.4
    print("OK test_sma_regime_causal")


def test_stitch_equity():
    a1 = np.array([1e6, 1.05e6, 1.1e6])
    a2 = np.array([1e6, 0.98e6])
    stitched = stitch_equity([a1, a2])
    expected = np.array([1.0, 1.05, 1.1, 1.1 * 0.98])
    assert np.allclose(stitched, expected)
    print("OK test_stitch_equity")


def test_replay_gate_all_ones_matches_predict():
    from sb3_ppo import compute_rich_features, predict_ppo_sb3, train_ppo_sb3
    from walkforward import default_env_params
    price, tech = compute_rich_features({"BTCUSDT": _ohlcv(n=500, drift=0.0005)})
    split = int(len(price) * 0.8)
    ep = default_env_params(20)
    md = tempfile.mkdtemp(prefix="wf_gate_")
    train_ppo_sb3(price[:split], tech[:split], ep, md,
                  total_timesteps=300, net_dimension=32, batch_size=64,
                  n_steps=128, verbose=0)
    account, actions = predict_ppo_sb3(price[split:], tech[split:], ep, md,
                                       return_actions=True)
    mask = np.ones(len(actions), dtype=bool)
    replay = replay_with_gate(price[split:], tech[split:], ep, actions, mask)
    n = min(len(account), len(replay))
    assert np.allclose(np.asarray(account)[:n], replay[:n], rtol=1e-6), (
        "all-true gate replay must match the raw predict account")
    print("OK test_replay_gate_all_ones_matches_predict", round(replay[-1], 2))


def test_walkforward_tiny():
    ohlcv = {"BTCUSDT": _ohlcv(n=650, drift=0.0005),
             "ETHUSDT": _ohlcv(n=650, drift=0.0005, seed=7)}
    out = tempfile.mkdtemp(prefix="wf_tiny_")
    rep = walkforward_backtest(list(ohlcv), n_train=280, n_test=50,
                               train_steps=250, outdir=out, ohlcv=ohlcv,
                               lookback=20, verbose=0)
    for key in ("metrics_raw", "metrics_gated", "metrics_eqw",
                "excess_raw", "excess_gated", "folds"):
        assert key in rep, key
    assert rep["windows"]["n_folds"] >= 1
    assert os.path.exists(os.path.join(out, "report.json"))
    assert os.path.exists(os.path.join(out, "oos_curves.npz"))
    for k in ("cum_return", "sharpe", "max_drawdown"):
        assert np.isfinite(rep["metrics_gated"][k])
    print("OK test_walkforward_tiny", "folds:", rep["windows"]["n_folds"],
          "excess_gated:", round(rep["excess_gated"], 3))


def test_walkforward_gate_protects_in_crash():
    n = 700
    # training = mild uptrend, test = crash
    train = 100 * np.exp(np.cumsum(np.full(n, 0.001)))
    test_start = train[-1]
    test = test_start * np.exp(np.cumsum(np.full(400, -0.004)))
    trend = np.concatenate([train, test])
    ohlcv = {"BTCUSDT": _ohlcv(n=n + 400, trend=trend)}
    out = tempfile.mkdtemp(prefix="wf_crash_")
    rep = walkforward_backtest(list(ohlcv), n_train=300, n_test=40,
                               train_steps=250, outdir=out, ohlcv=ohlcv,
                               lookback=20, verbose=0, gate=True)
    # eqw (buy & hold) must crash
    assert rep["metrics_eqw"]["cum_return"] < -0.05
    # the gated agent should lose (much) less than the benchmark
    gated = rep["metrics_gated"]["cum_return"]
    eqw = rep["metrics_eqw"]["cum_return"]
    assert gated > eqw, f"gate failed: gated {gated} < eqw {eqw}"
    print(f"OK test_walkforward_gate_protects_in_crash gated {gated:+.3f} eqw {eqw:+.3f}")


def test_retrain_latest():
    ohlcv = {"BTCUSDT": _ohlcv(n=400, drift=0.001)}
    out = tempfile.mkdtemp(prefix="wf_retrain_")
    rep = retrain_latest(list(ohlcv), n_train=260, train_steps=250,
                         outdir=out, ohlcv=ohlcv, lookback=20)
    assert os.path.exists(os.path.join(out, "ppo_sb3.zip"))
    assert os.path.exists(os.path.join(out, "vecnormalize.pkl"))
    assert os.path.exists(os.path.join(out, "report.json"))
    assert rep["current_regime"] in ("bull", "bear")
    print("OK test_retrain_latest", "regime:", rep["current_regime"])


if __name__ == "__main__":
    test_confirmed_bear_causal()
    test_sma_regime_causal()
    test_stitch_equity()
    test_replay_gate_all_ones_matches_predict()
    test_walkforward_tiny()
    test_walkforward_gate_protects_in_crash()
    test_retrain_latest()
    print("All walkforward tests passed.")
