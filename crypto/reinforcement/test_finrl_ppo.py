"""
Tests for reinforcement/finrl_ppo.py (FinRL_Crypto PPO layer).
Run: python reinforcement/test_finrl_ppo.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from finrl_ppo import (is_available, build_features, default_env_params,
                       train_ppo, predict_ppo, evaluate_ppo,
                       compute_eqw, compute_data_points_per_year,
                       fetch_klines)


def _ohlcv(n=400, seed=0, drift=0.002, start=100.0):
    rng = np.random.default_rng(seed)
    close = start * np.exp(np.cumsum(rng.normal(drift, 0.002, n)))
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    volume = np.abs(rng.normal(1000, 300, n))
    return pd.DataFrame({"timestamp": np.arange(n), "open": open_,
                         "high": high, "low": low, "close": close,
                         "volume": volume})


def _trend_ohlcv(n=800, start=100.0):
    close = start * np.exp(np.linspace(0, 2.0, n))
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    volume = np.abs(np.random.default_rng(2).normal(1000, 300, n))
    return pd.DataFrame({"timestamp": np.arange(n), "open": open_,
                         "high": high, "low": low, "close": close,
                         "volume": volume})


def test_available():
    assert is_available(), "torch must be importable for PPO"
    print("OK test_available")


def test_build_features():
    price, tech = build_features({"BTCUSDT": _ohlcv(n=400)})
    assert price.ndim == 2 and price.shape[1] == 1
    assert tech.shape[1] == 14, f"expected 14 features, got {tech.shape[1]}"
    assert price.shape[0] == 400 - 100, "warmup rows should be dropped"
    assert np.isnan(tech).sum() == 0, "tech_array must be NaN-free"
    assert np.isnan(price).sum() == 0
    print("OK test_build_features", price.shape, tech.shape)


def test_build_features_multiasset():
    price, tech = build_features({"BTCUSDT": _ohlcv(seed=0), "ETHUSDT": _ohlcv(seed=1)})
    assert price.shape[1] == 2
    assert tech.shape[1] == 28
    print("OK test_build_features_multiasset", price.shape)


def test_default_env_params():
    ep = default_env_params(20)
    assert ep["lookback"] == 20
    # norm_action fix: max action must exceed minimum order size to allow trading
    assert ep["norm_action"] >= 1_000_000 * 0.1, "norm_action must scale with capital"
    ep2 = default_env_params(30, initial_capital=100_000, max_position_frac=0.5)
    assert ep2["norm_action"] == 50_000.0
    print("OK test_default_env_params", ep["norm_action"])


def test_env_mechanics():
    from finrl_crypto import CryptoEnvCCXT
    price, tech = build_features({"BTCUSDT": _trend_ohlcv(n=200)})
    ep = default_env_params(20)
    config = {"price_array": price, "tech_array": tech, "ticker_list": ["BTC/USDT"]}
    env = CryptoEnvCCXT(config=config, env_params=ep, if_log=False)

    s0 = env.reset()
    assert s0.shape == (1 + 1 + 14 * 20,), f"state shape {s0.shape}"
    assert env.stocks.sum() == 0 and env.cash == 1_000_000

    # big buy should hold a position
    _, _, done, _ = env.step(np.array([100.0]))
    assert env.stocks[0] > 0, "buy must increase holdings"
    assert env.cash < 1_000_000, "cash must decrease after buy"

    # holding through an uptrend grows portfolio value
    v0 = env.get_portfolio_value()
    for _ in range(5):
        _, _, done, _ = env.step(np.array([0.0]))
    assert env.get_portfolio_value() > v0, "portfolio must grow in an uptrend"

    # full sell clears holdings
    _, _, done, _ = env.step(np.array([-1e6]))
    assert env.stocks[0] == 0, "sell must clear holdings"
    assert abs(env.cash - env.get_portfolio_value()) < 1e-3
    print("OK test_env_mechanics")


def test_train_and_predict():
    if not is_available():
        print("SKIP test_train_and_predict (torch missing)")
        return
    price, tech = build_features({"BTCUSDT": _trend_ohlcv(n=800)})
    split = int(len(price) * 0.8)
    ep = default_env_params(20)
    model_dir = tempfile.mkdtemp(prefix="ppo_test_")
    res = train_ppo(price[:split], tech[:split], ep, model_dir,
                    total_timesteps=800, net_dimension=64, batch_size=128,
                    target_step=128, eval_time_gap=0.5)
    assert os.path.exists(os.path.join(model_dir, "actor.pth")), "actor.pth must be saved"
    assert res["state_dim"] == 1 + 1 + 14 * 20
    assert res["action_dim"] == 1

    account = predict_ppo(price[split:], tech[split:], ep, model_dir, net_dimension=64)
    assert len(account) >= 1
    assert account[0] == 1_000_000
    assert len(set(np.round(account, 4))) > 3, "agent should trade and vary the account"
    print("OK test_train_and_predict", res, "final:", account[-1])


def test_predict_too_short_raises():
    price, tech = build_features({"BTCUSDT": _trend_ohlcv(n=800)})
    ep = default_env_params(50)
    try:
        predict_ppo(price[:40], tech[:40], ep, "nope", net_dimension=32)
        raise AssertionError("expected ValueError for too-short test data")
    except ValueError:
        pass
    print("OK test_predict_too_short_raises")


def test_evaluate_metrics():
    price, tech = build_features({"BTCUSDT": _trend_ohlcv(n=800)})
    split = int(len(price) * 0.8)
    price_test = price[split:]
    eqw = compute_eqw(price_test, 0, len(price_test))
    assert eqw is not None and len(eqw[0]) == len(price_test)

    synthetic_account = 1_000_000 * np.exp(np.linspace(0, 0.3, len(price_test)))
    m = evaluate_ppo(synthetic_account, price_test, "1h", 20)
    for key in ("sharpe_bot", "sharpe_eqw", "excess_sharpe", "cum_return_bot",
                "cum_return_eqw", "max_drawdown_bot", "max_drawdown_eqw",
                "final_asset", "trades"):
        assert key in m, f"missing metric {key}"
    assert np.isfinite(m["sharpe_bot"])
    assert m["final_asset"] > 1_000_000
    print("OK test_evaluate_metrics", {k: round(v, 3) for k, v in m.items()
                                       if isinstance(v, float)})


def test_timeframe_map():
    assert compute_data_points_per_year("1d") == 365
    assert compute_data_points_per_year("5m") == 12 * 24 * 365
    assert compute_data_points_per_year("1h") == 24 * 365
    try:
        compute_data_points_per_year("xx")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    print("OK test_timeframe_map")


def test_signal_mapping():
    from ppo_signal import signal_from_action
    assert signal_from_action(5e4, 1e5) == "buy"
    assert signal_from_action(-3e4, 1e5) == "sell"
    assert signal_from_action(1e3, 1e5) == "hold"
    assert signal_from_action(0.0, 1e5) == "hold"
    print("OK test_signal_mapping")


def test_signal_error_path():
    from ppo_signal import generate
    bad = generate("BTCUSDT", "C:/does/not/exist", limit=50, lookback=20)
    assert "error" in bad, bad
    print("OK test_signal_error_path")


def test_hybrid_scoring():
    from hybrid_signal import ppo_score, pred_score, fuse
    assert ppo_score({"action": [5e4], "norm_action": 1e5}) == 0.5
    assert ppo_score({"action": [1e3], "norm_action": 1e5}) == 0.0
    assert ppo_score({"error": "x"}) is None
    assert pred_score({"predicted_change_pct": 3.0}) == 1.0
    assert pred_score({"predicted_change_pct": -1.5}) == -0.5
    assert pred_score({"error": "x"}) is None
    print("OK test_hybrid_scoring")


def test_hybrid_fuse():
    from hybrid_signal import fuse
    rec = fuse({"action": [5e4], "norm_action": 1e5, "signal": "buy"},
               {"predicted_change_pct": 2.0, "confidence": 70})
    assert rec["side"] == "LONG" and rec["agreement"] == "both"
    assert rec["confidence"] >= 70

    rec2 = fuse({"action": [5e4], "norm_action": 1e5, "signal": "buy"},
                {"predicted_change_pct": -2.0, "confidence": 80})
    assert rec2["signal"] == "HOLD" and rec2["side"] is None
    assert rec2["agreement"] == "conflict" and rec2["confidence"] <= 50

    rec3 = fuse({"action": [1e3], "norm_action": 1e5, "signal": "hold"},
                {"predicted_change_pct": 3.5, "confidence": 65})
    assert rec3["signal"] == "STRONG BUY" and rec3["agreement"] == "predictor"

    rec4 = fuse({"action": [7e4], "norm_action": 1e5, "signal": "buy"}, {"error": "x"})
    assert rec4["agreement"] == "ppo"

    assert "error" in fuse({"error": "x"}, {"error": "y"})
    print("OK test_hybrid_fuse")


def test_fetch_klines_graceful():
    df = fetch_klines("BTCUSDT", "1h", 5)
    if df is None:
        print("SKIP test_fetch_klines_graceful (offline)")
        return
    assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    assert len(df) == 5
    print("OK test_fetch_klines_graceful", len(df), "rows")


if __name__ == "__main__":
    test_available()
    test_build_features()
    test_build_features_multiasset()
    test_default_env_params()
    test_env_mechanics()
    test_train_and_predict()
    test_predict_too_short_raises()
    test_evaluate_metrics()
    test_timeframe_map()
    test_signal_mapping()
    test_signal_error_path()
    test_hybrid_scoring()
    test_hybrid_fuse()
    test_fetch_klines_graceful()
    print("\nAll finrl_ppo tests passed.")
