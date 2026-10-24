"""
Tests for reinforcement/sb3_ppo.py (stable-baselines3 PPO path).
Run: python reinforcement/test_sb3_ppo.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

import finrl_ppo as fp
from sb3_ppo import (build_rich_features, make_env_inst, train_ppo_sb3,
                     predict_ppo_sb3)


def _ohlcv(n=600, seed=0, drift=0.0003):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift, 0.004, n)))
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) * 1.002
    low = np.minimum(open_, close) * 0.998
    volume = np.abs(rng.normal(1000, 300, n))
    return pd.DataFrame({"timestamp": np.arange(n), "open": open_,
                         "high": high, "low": low, "close": close,
                         "volume": volume})


def test_rich_features():
    price, tech = build_rich_features({"BTCUSDT": _ohlcv()})
    assert price.shape[1] == 1
    assert tech.shape[1] == 23, f"expected 23 features/asset, got {tech.shape[1]}"
    assert price.shape[0] == 600 - fp.WARMUP_ROWS
    assert np.isnan(tech).sum() == 0
    assert abs(float(np.mean(tech))) < 0.05, "features should be ~z-scored"
    assert abs(float(np.std(tech)) - 1.0) < 0.2, "features should be ~unit std"
    print("OK test_rich_features", price.shape, tech.shape)


def test_env_adapter():
    from finrl_crypto.sb3_adapter import SB3CryptoEnv
    price, tech = build_rich_features({"BTCUSDT": _ohlcv()})
    ep = fp.default_env_params(30)
    config = {"price_array": price, "tech_array": tech, "ticker_list": ["ASSET0/USDT"]}
    env = SB3CryptoEnv(config=config, env_params=ep, if_log=False)
    assert env.observation_space.shape == (1 + 1 + 23 * 30,)
    assert env.action_space.shape == (1,)
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    obs, reward, term, trunc, _ = env.step(env.action_space.sample())
    assert isinstance(term, bool) and isinstance(trunc, bool)
    print("OK test_env_adapter", env.observation_space.shape)


def test_sb3_train_predict():
    price, tech = build_rich_features({"BTCUSDT": _ohlcv(n=800, drift=0.002)})
    split = int(len(price) * 0.8)
    ep = fp.default_env_params(20)
    md = tempfile.mkdtemp(prefix="sb3_test_")
    res = train_ppo_sb3(price[:split], tech[:split], ep, md,
                        total_timesteps=600, net_dimension=32, batch_size=64,
                        n_steps=256, verbose=0)
    assert os.path.exists(os.path.join(md, "ppo_sb3.zip"))
    assert os.path.exists(os.path.join(md, "vecnormalize.pkl"))
    assert res["backend"] == "sb3"
    account, actions = predict_ppo_sb3(price[split:], tech[split:], ep, md,
                                       return_actions=True)
    assert len(account) >= 1 and account[0] == 1_000_000
    # regression: DummyVecEnv auto-resets the env once done=True, so a naive
    # reader appended the post-reset initial state (1e6) as the final equity.
    # Replay the exact same actions on a raw env (no auto-reset) and compare.
    ref = make_env_inst(price[split:], tech[split:], ep, if_log=False)
    ref.reset()
    for a in actions[:-1]:
        ref.step(np.asarray(a, dtype=np.float64))
    true_final = ref.total_asset
    ref.close()
    assert abs(account[-1] - true_final) < 0.01 * true_final, (
        f"predicted final equity {account[-1]:.2f} != true final {true_final:.2f} "
        "(post-reset equity leak)")
    print("OK test_sb3_train_predict", res["state_dim"], len(account),
          round(account[-1], 2))


def test_relative_reward():
    from finrl_crypto.sb3_adapter import SB3CryptoEnv
    price, tech = build_rich_features({"BTCUSDT": _ohlcv()})
    ep = fp.default_env_params(20)
    ep["reward_mode"] = "relative"
    config = {"price_array": price, "tech_array": tech, "ticker_list": ["ASSET0/USDT"]}
    env = SB3CryptoEnv(config=config, env_params=ep, if_log=False)
    assert env.reward_mode == "relative"
    env.reset()
    rewards = []
    for _ in range(30):
        _, r, _, _, _ = env.step(np.array([2e4]))
        rewards.append(r)
    assert np.isfinite(rewards).all()
    print("OK test_relative_reward", "mean", round(float(np.mean(rewards)), 6))


def test_fetch_paginated_graceful():
    df = fp.fetch_klines_paginated("BTCUSDT", "1h", total=1500)
    if df is None:
        print("SKIP test_fetch_paginated_graceful (offline)")
        return
    assert len(df) == 1500
    assert df["timestamp"].nunique() == 1500, "pages must not overlap"
    print("OK test_fetch_paginated_graceful", len(df), "rows")


if __name__ == "__main__":
    test_rich_features()
    test_env_adapter()
    test_sb3_train_predict()
    test_relative_reward()
    test_fetch_paginated_graceful()
    print("\nAll sb3_ppo tests passed.")
