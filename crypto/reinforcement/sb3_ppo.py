"""
Stronger PPO path using stable-baselines3 with richer, z-scored features.

CryptoEnvCCXT + sb3 = battle-tested PPO implementation with VecNormalize
(observation + reward normalization) instead of the hand-rolled ElegantRL fork.

Features are enriched per asset (log returns, rolling volatility, trailing
z-scores) on top of the base indicator set, then standardized sample-wide
(standard FinRL practice).  `price_array` stays as raw prices (the env needs
them for position valuation); the standardized block is `tech_array`.

Usage:
    from reinforcement import sb3_ppo as sp
    price, tech = sp.build_rich_features({"BTCUSDT": df})
    sp.train_ppo_sb3(price, tech, fp.default_env_params(50), model_dir)
    account = sp.predict_ppo_sb3(price, tech, fp.default_env_params(50), model_dir)
"""

import os

import numpy as np
import pandas as pd

try:
    from . import finrl_ppo as fp
    from .finrl_crypto.sb3_adapter import SB3CryptoEnv, SB3WeightEnv
except ImportError:
    import sys
    _pkg_dir = os.path.dirname(os.path.abspath(__file__))
    if _pkg_dir not in sys.path:
        sys.path.insert(0, _pkg_dir)
    import finrl_ppo as fp  # noqa: F401
    from finrl_crypto.sb3_adapter import SB3CryptoEnv, SB3WeightEnv  # noqa: F401

__all__ = ["build_rich_features", "compute_rich_features", "fit_standardizer",
           "apply_standardizer", "train_ppo_sb3", "predict_ppo_sb3", "make_env_inst"]

WARMUP_ROWS = fp.WARMUP_ROWS

RICH_FEATURE_COLUMNS = fp.FEATURE_COLUMNS + [
    "log_ret_1", "log_ret_3", "log_ret_6", "log_ret_12", "log_ret_24",
    "vol_12", "vol_24", "close_z", "momentum_z",
]

# Best hyper-parameters found by optuna tuning on 1y of BTCUSDT 1h data
# (train 70% / val 10% / test 20%, relative-reward mode).
SB3_PPO_BEST = {
    "learning_rate": 0.0020263302316476015,
    "ent_coef": 0.005,
    "gamma": 0.98,
    "gae_lambda": 0.9,
    "net_dimension": 128,
    "batch_size": 512,
}


def _rolling(df, col, w, agg):
    return df[col].rolling(w).agg(agg).to_numpy()


def compute_rich_features(ohlcv):
    """Compute enriched indicators per asset (NO standardization, warmup dropped).

    Returns (price_array, tech_raw): aligned closes and the raw feature block.
    Pair with fit_standardizer / apply_standardizer for leakage-free
    walk-forward folds (standardize using only the training window).
    """
    if not ohlcv:
        raise ValueError("ohlcv must be a non-empty dict of DataFrames")
    I = fp._get_indicators()
    n = min(len(df) for df in ohlcv.values())
    if n < WARMUP_ROWS + 2:
        raise ValueError(f"Not enough rows per asset ({n}) for warmup {WARMUP_ROWS}")

    prices_list, techs_list = [], []
    for sym, df in ohlcv.items():
        df = df.tail(n).copy()
        open_ = df["open"].to_numpy(np.float64)
        high = df["high"].to_numpy(np.float64)
        low = df["low"].to_numpy(np.float64)
        close = df["close"].to_numpy(np.float64)
        volume = df["volume"].to_numpy(np.float64)
        mean = (open_ + high + low + close) / 4.0

        log_close = pd.Series(np.log(np.maximum(close, 1e-8)))
        ret = log_close.diff().to_numpy().copy()
        ret[0] = 0.0

        df2 = pd.DataFrame({
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "ret": ret,
        })
        base = {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume,
            "macd": I.macd(mean, 12, 26),
            "rsi": I.rsi(mean, 14),
            "stoch1": I.stoch(close, high, low, 5, 3)[0],
            "stoch2": I.stoch(close, high, low, 5, 3)[1],
            "wpr": I.wpr(close, high, low, 14),
            "atr": I.atr(open_, high, low, 14),
            "roc": I.roc(mean, 12),
            "chop": I.chop(close, open_, high, low, 14),
            "momentum": I.momentum(mean, 40),
        }
        rich = dict(base)
        rich["log_ret_1"] = ret
        rich["log_ret_3"] = _rolling(df2, "ret", 3, "sum")
        rich["log_ret_6"] = _rolling(df2, "ret", 6, "sum")
        rich["log_ret_12"] = _rolling(df2, "ret", 12, "sum")
        rich["log_ret_24"] = _rolling(df2, "ret", 24, "sum")
        rich["vol_12"] = _rolling(df2, "ret", 12, "std")
        rich["vol_24"] = _rolling(df2, "ret", 24, "std")
        close_ma = _rolling(df2, "close", 100, "mean")
        close_std = _rolling(df2, "close", 100, "std")
        rich["close_z"] = (close - close_ma) / np.where(close_std < 1e-12, np.nan, close_std)
        mom_ma = _rolling(df2, "close", 40, "mean")
        rich["momentum_z"] = (close - mom_ma) / np.where(mom_ma < 1e-12, np.nan, mom_ma)

        tech = np.column_stack([rich[c] for c in RICH_FEATURE_COLUMNS])
        techs_list.append(tech)
        prices_list.append(close)

    tech_array = np.hstack(techs_list)
    price_array = np.column_stack(prices_list)[WARMUP_ROWS:]
    tech_array = tech_array[WARMUP_ROWS:]
    return (np.nan_to_num(price_array, nan=0.0).astype(np.float32),
            tech_array.astype(np.float32))


def fit_standardizer(tech_raw):
    """Per-asset-block mean/std of the raw feature block (train window only)."""
    block = len(RICH_FEATURE_COLUMNS)
    n_assets = tech_raw.shape[1] // block
    mean = np.zeros(tech_raw.shape[1], dtype=np.float64)
    std = np.ones(tech_raw.shape[1], dtype=np.float64)
    for i in range(n_assets):
        b = tech_raw[:, i * block:(i + 1) * block]
        m = np.nanmean(b, axis=0)
        s = np.nanstd(b, axis=0)
        s = np.where(s < 1e-12, 1.0, s)
        mean[i * block:(i + 1) * block] = m
        std[i * block:(i + 1) * block] = s
    return mean, std


def apply_standardizer(tech_raw, mean, std):
    """Z-score a feature block with a previously fitted scaler."""
    tech = (tech_raw - mean) / std
    tech = np.nan_to_num(tech, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return tech


def build_rich_features(ohlcv):
    """Enriched, standardized features: (price_array, tech_array).

    Sample-wide standardization (uses every row for the scaler) — fine for a
    single train/test split, but for walk-forward use compute_rich_features +
    fit_standardizer(train-only) + apply_standardizer to avoid look-ahead.
    """
    price_array, tech_raw = compute_rich_features(ohlcv)
    mean, std = fit_standardizer(tech_raw)
    return price_array, apply_standardizer(tech_raw, mean, std)


def make_env_inst(price_array, tech_array, env_params, **kwargs):
    """Build the sb3 environment matching env_params['action_mode'].

    Default action_mode is 'weight' (target-weight rebalancing, SB3WeightEnv);
    'quantity' uses the legacy SB3CryptoEnv (raw crypto-quantity actions).
    """
    n_assets = price_array.shape[1]
    config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "if_train": False,
        "ticker_list": [f"ASSET{i}/USDT" for i in range(n_assets)],
    }
    mode = env_params.get("action_mode", "weight")
    cls = SB3WeightEnv if mode == "weight" else SB3CryptoEnv
    return cls(config=config, env_params=env_params, **kwargs)


def _make_env(price_array, tech_array, env_params):
    return lambda: make_env_inst(price_array, tech_array, env_params, if_log=False)


def train_ppo_sb3(price_array, tech_array, env_params, model_dir,
                  total_timesteps=40000, net_dimension=None, learning_rate=None,
                  gamma=None, n_steps=2048, batch_size=None, gae_lambda=None,
                  clip_range=0.2, ent_coef=None, seed=42, verbose=0):
    """Train PPO via stable-baselines3 (VecNormalize obs+reward).

    Hyper-parameters default to the optuna-tuned SB3_PPO_BEST.  Returns a
    metadata dict; saves ppo_sb3.zip + vecnormalize.pkl in model_dir.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    if not fp.is_available():
        raise RuntimeError("PPO unavailable: torch is not installed")
    import torch
    torch.set_grad_enabled(True)  # guard against ElegantRL leaving grad disabled
    model_dir = os.path.normpath(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    hp = SB3_PPO_BEST
    net_dimension = int(net_dimension or hp["net_dimension"])
    learning_rate = learning_rate if learning_rate is not None else hp["learning_rate"]
    gamma = gamma if gamma is not None else hp["gamma"]
    batch_size = int(batch_size or hp["batch_size"])
    gae_lambda = gae_lambda if gae_lambda is not None else hp["gae_lambda"]
    ent_coef = ent_coef if ent_coef is not None else hp["ent_coef"]

    venv = DummyVecEnv([_make_env(price_array, tech_array, env_params)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True,
                        clip_obs=10.0, clip_reward=10.0)

    model = PPO(
        "MlpPolicy", venv,
        policy_kwargs={"net_arch": [int(net_dimension), int(net_dimension)]},
        learning_rate=learning_rate, gamma=gamma,
        n_steps=int(n_steps), batch_size=int(batch_size),
        gae_lambda=gae_lambda, clip_range=clip_range, ent_coef=ent_coef,
        seed=seed, verbose=verbose)
    model.learn(total_timesteps=int(total_timesteps))

    model_path = os.path.join(model_dir, "ppo_sb3.zip")
    model.save(model_path)
    venv_path = os.path.join(model_dir, "vecnormalize.pkl")
    venv.save(venv_path)

    env = venv.envs[0].inner
    return {
        "model_dir": model_dir, "backend": "sb3",
        "state_dim": env.state_dim, "action_dim": env.action_dim,
        "max_step": env.max_step, "total_timesteps": int(total_timesteps),
        "vecnormalize": venv_path,
    }


def predict_ppo_sb3(price_array, tech_array, env_params, model_dir,
                    deterministic=True, verbose=0, return_actions=False):
    """Run a trained sb3 PPO on (test) data -> account-value curve.

    If `return_actions` is True, returns (account, actions) where actions is
    the list of raw policy actions (in norm_action units) per step.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    if not fp.is_available():
        raise RuntimeError("PPO unavailable: torch is not installed")
    lookback = env_params["lookback"]
    if price_array.shape[0] < lookback + 2:
        raise ValueError(
            f"Test data too short: {price_array.shape[0]} rows with lookback {lookback} "
            f"(need >= {lookback + 2})")

    model_dir = os.path.normpath(model_dir)
    venv = VecNormalize.load(
        os.path.join(model_dir, "vecnormalize.pkl"),
        DummyVecEnv([_make_env(price_array, tech_array, env_params)]))
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(os.path.join(model_dir, "ppo_sb3.zip"))
    env = venv.envs[0]

    obs = venv.reset()
    account = [float(env.cash + (env.price_array[env.time] * env.stocks).sum())]
    actions = []
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        actions.append(np.atleast_1d(np.asarray(action[0], dtype=np.float64)))
        obs, _, done, _ = venv.step(action)
        # DummyVecEnv auto-resets the env once done=True, so the env state is
        # no longer the episode end; only record equity while the episode runs.
        if bool(done[0]):
            break
        account.append(float(env.cash + (env.price_array[env.time] * env.stocks).sum()))

    if return_actions:
        return account, actions
    return account
