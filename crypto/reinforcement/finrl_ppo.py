"""
طبقة PPO من FinRL_Crypto — high-level API for training / evaluating a PPO
trading agent on historical OHLCV, using the vendored ElegantRL PPO fork.

Typical flow:

    from reinforcement import finrl_ppo as fp
    if not fp.is_available():
        print("PPO unavailable (torch not installed)")
        return

    price, tech = fp.build_features({"BTCUSDT": fp.fetch_klines("BTCUSDT", "1h", 2000)})
    split = int(len(price) * 0.8)
    fp.train_ppo(price[:split], tech[:split], fp.default_env_params(50),
                 model_dir="ppo_models/btc_1h", total_timesteps=3000)
    account = fp.predict_ppo(price[split:], tech[split:], fp.default_env_params(50),
                             model_dir="ppo_models/btc_1h")
    metrics = fp.evaluate_ppo(account, price[split:], "1h", 50)
    print(metrics)

NOTE: this module must stay import-light (no torch at module import time).
"""

import os

import numpy as np

__all__ = [
    "is_available",
    "fetch_klines",
    "build_features",
    "FEATURE_COLUMNS",
    "default_env_params",
    "ppo_hyperparams",
    "train_ppo",
    "predict_ppo",
    "evaluate_ppo",
    "compute_eqw",
    "compute_data_points_per_year",
    "run_pipeline",
]

BINANCE_BASE = "https://api.binance.com"

# Technical features computed per asset (no TA-Lib, uses data_loader/indicators.py)
FEATURE_COLUMNS = [
    "open", "high", "low", "close", "volume",
    "macd", "rsi", "stoch1", "stoch2", "wpr", "atr", "roc", "chop", "momentum",
]

WARMUP_ROWS = 100  # rows dropped at the start (max indicator warmup)


def is_available():
    """True if torch is importable (PPO training requires it)."""
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _get_indicators():
    """Load data_loader/indicators.py directly (its package __init__ is broken)."""
    global _indicators_mod
    if _indicators_mod is None:
        import importlib.util
        path = os.path.normpath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data_loader", "indicators.py"))
        spec = importlib.util.spec_from_file_location("_stratocrypto_indicators", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _indicators_mod = mod
    return _indicators_mod


_indicators_mod = None


def fetch_klines(symbol, interval="1h", limit=200):
    """Fetch OHLCV candles from the public Binance API.

    Returns a pandas DataFrame with columns
    [timestamp, open, high, low, close, volume], or None on failure.
    """
    import requests
    import pandas as pd
    try:
        r = requests.get(f"{BINANCE_BASE}/api/v3/klines",
                         params={"symbol": symbol, "interval": interval, "limit": int(limit)},
                         timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None
    if not isinstance(data, list) or not data:
        return None
    cols = ["timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_av", "trades", "tb_base_av", "tb_quote_av", "ignore"]
    df = pd.DataFrame(data, columns=cols)[["timestamp", "open", "high", "low", "close", "volume"]]
    return df.apply(pd.to_numeric, errors="coerce")


def fetch_klines_paginated(symbol, interval="1h", total=2000, per=1000,
                           max_retries=4, pause=0.3):
    """Fetch more candles than one API call allows, by walking backwards.

    Pages are accumulated oldest-first using `endTime` of the oldest candle
    already fetched, so pages never overlap.  Returns a DataFrame like
    fetch_klines, or None on persistent failure.
    """
    import time
    import requests
    import pandas as pd
    out = []
    params = {"symbol": symbol, "interval": interval, "limit": int(per)}
    while len(out) < int(total):
        if out:
            params["endTime"] = int(out[0][0]) - 1
        batch = None
        for attempt in range(max_retries):
            try:
                r = requests.get(f"{BINANCE_BASE}/api/v3/klines", params=params, timeout=20)
                r.raise_for_status()
                batch = r.json()
                break
            except Exception:
                if attempt == max_retries - 1:
                    return None
                time.sleep(3 + attempt * 3)
        if not batch:
            break
        out = batch + out
        if len(batch) < int(per):
            break
        time.sleep(pause)
    if not out:
        return None
    cols = ["timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_av", "trades", "tb_base_av", "tb_quote_av", "ignore"]
    df = pd.DataFrame(out, columns=cols)[["timestamp", "open", "high", "low", "close", "volume"]]
    return df.apply(pd.to_numeric, errors="coerce").tail(int(total)).reset_index(drop=True)


def build_features(ohlcv):
    """Build price/tech arrays for the RL environment from raw OHLCV.

    Args:
        ohlcv: dict {symbol: DataFrame[timestamp, open, high, low, close, volume]}

    Returns:
        (price_array [n, n_assets], tech_array [n, n_assets * n_features])
        aligned across assets, warmup rows dropped, NaN -> 0.
    """
    if not ohlcv:
        raise ValueError("ohlcv must be a non-empty dict of DataFrames")
    I = _get_indicators()
    n = min(len(df) for df in ohlcv.values())
    if n < WARMUP_ROWS + 2:
        raise ValueError(f"Not enough rows per asset ({n}) for warmup {WARMUP_ROWS}")

    prices_list, techs_list = [], []
    for sym, df in ohlcv.items():
        df = df.tail(n)
        open_ = df["open"].to_numpy(np.float64)
        high = df["high"].to_numpy(np.float64)
        low = df["low"].to_numpy(np.float64)
        close = df["close"].to_numpy(np.float64)
        volume = df["volume"].to_numpy(np.float64)
        mean = (open_ + high + low + close) / 4.0
        stoch1, stoch2 = I.stoch(close, high, low, 5, 3)
        feats = {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "macd": I.macd(mean, 12, 26),
            "rsi": I.rsi(mean, 14),
            "stoch1": stoch1,
            "stoch2": stoch2,
            "wpr": I.wpr(close, high, low, 14),
            "atr": I.atr(open_, high, low, 14),
            "roc": I.roc(mean, 12),
            "chop": I.chop(close, open_, high, low, 14),
            "momentum": I.momentum(mean, 40),
        }
        techs_list.append(np.column_stack([feats[c] for c in FEATURE_COLUMNS]))
        prices_list.append(close)

    price_array = np.column_stack(prices_list)[WARMUP_ROWS:]
    tech_array = np.hstack(techs_list)[WARMUP_ROWS:]
    tech_array = np.nan_to_num(tech_array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    price_array = np.nan_to_num(price_array, nan=0.0).astype(np.float32)
    return price_array, tech_array


def default_env_params(lookback=50, crypto_limits=None,
                       initial_capital=1_000_000, max_position_frac=0.1):
    """Environment normalization params (FinRL_Crypto defaults, fixed action scale).

    NOTE: FinRL_Crypto ships `norm_action=1`, which makes the max buy order
    (1 x 10^-floor(log10(price)) BTC) smaller than the minimum order size, so
    the agent can never trade. We default `norm_action` to
    `initial_capital * max_position_frac`, i.e. a full-strength action deploys
    `max_position_frac` of the portfolio in one order.
    """
    norm_action = max(1.0, float(initial_capital * max_position_frac))
    ep = {
        "lookback": int(lookback),
        "norm_cash": 1e-6,
        "norm_stocks": 100,
        "norm_tech": 1,
        "norm_reward": 1,
        "norm_action": norm_action,
    }
    if crypto_limits is not None:
        ep["crypto_limits"] = crypto_limits
    return ep


def ppo_hyperparams(net_dimension=128, learning_rate=1e-4, batch_size=256,
                    gamma=0.99, target_step=1024, eval_time_gap=2.0):
    """ElegantRL PPO hyper-parameters (FinRL_Crypto style)."""
    return {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gamma": gamma,
        "net_dimension": net_dimension,
        "target_step": target_step,
        "eval_time_gap": eval_time_gap,
    }


def _import_finrl():
    try:
        from .finrl_crypto import CryptoEnvCCXT, DRLAgent
    except ImportError:
        import sys
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        if pkg_dir not in sys.path:
            sys.path.insert(0, pkg_dir)
        from finrl_crypto import CryptoEnvCCXT, DRLAgent
    return CryptoEnvCCXT, DRLAgent


def train_ppo(price_array, tech_array, env_params, model_dir, total_timesteps=2000,
              gpu_id=-1, model_name="ppo", net_dimension=128, learning_rate=1e-4,
              batch_size=256, gamma=0.99, target_step=1024, eval_time_gap=2.0,
              if_log=False):
    """Train a PPO agent on historical data and save it under model_dir.

    Returns a dict with training metadata.
    """
    if not is_available():
        raise RuntimeError("PPO unavailable: torch is not installed")
    CryptoEnvCCXT, DRLAgent = _import_finrl()
    model_dir = os.path.normpath(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    hp = ppo_hyperparams(net_dimension=net_dimension, learning_rate=learning_rate,
                         batch_size=batch_size, gamma=gamma,
                         target_step=target_step, eval_time_gap=eval_time_gap)

    agent = DRLAgent(env=CryptoEnvCCXT, price_array=price_array, tech_array=tech_array,
                     env_params=env_params, if_log=if_log)
    model = agent.get_model(model_name, gpu_id, model_kwargs=hp)
    agent.train_model(model=model, cwd=model_dir, total_timesteps=int(total_timesteps))

    # ElegantRL's learner leaves torch gradients globally disabled; restore so
    # any later torch training (e.g. stable-baselines3) in the same process works.
    import torch
    torch.set_grad_enabled(True)

    return {
        "model_dir": model_dir,
        "model_name": model_name,
        "net_dimension": int(net_dimension),
        "state_dim": int(model.state_dim),
        "action_dim": int(model.action_dim),
        "max_step": int(model.max_step),
    }


def predict_ppo(price_array, tech_array, env_params, model_dir,
                net_dimension=128, gpu_id=-1, model_name="ppo", if_log=False,
                return_actions=False):
    """Run the trained agent on (test) data.

    Returns the account-value curve (list of floats) over the episode.
    If `return_actions` is True, returns (account_curve, actions) where
    actions is the list of per-step raw actions (shape [n_steps, n_assets]).
    """
    if not is_available():
        raise RuntimeError("PPO unavailable: torch is not installed")
    CryptoEnvCCXT, DRLAgent = _import_finrl()
    n_assets = price_array.shape[1]
    lookback = env_params["lookback"]
    if price_array.shape[0] < lookback + 2:
        raise ValueError(
            f"Test data too short: {price_array.shape[0]} rows with lookback {lookback} "
            f"(need >= {lookback + 2})")
    ticker_list = [f"ASSET{i}/USDT" for i in range(n_assets)]
    config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "if_train": False,
        "ticker_list": ticker_list,
    }
    env = CryptoEnvCCXT(config=config, env_params=env_params, if_log=if_log)
    out = DRLAgent.DRL_prediction(
        model_name=model_name, cwd=model_dir,
        net_dimension=net_dimension, environment=env, gpu_id=gpu_id,
        return_actions=return_actions)
    if return_actions:
        account, actions = out
        return (list(np.asarray(account, dtype=np.float64)),
                [np.asarray(a, dtype=np.float64) for a in actions])
    return list(np.asarray(out, dtype=np.float64))


def compute_data_points_per_year(timeframe):
    """Annual number of candles for a Binance timeframe string."""
    table = {
        "1m": 60 * 24 * 365, "3m": 20 * 24 * 365, "5m": 12 * 24 * 365,
        "10m": 6 * 24 * 365, "15m": 4 * 24 * 365, "30m": 2 * 24 * 365,
        "1h": 24 * 365, "2h": 12 * 365, "4h": 6 * 365, "6h": 4 * 365,
        "12h": 2 * 365, "1d": 365,
    }
    if timeframe not in table:
        raise ValueError(f"Timeframe {timeframe!r} not supported")
    return table[timeframe]


def compute_eqw(price_array, indx1=0, indx2=None):
    """Equal-weight buy-and-hold benchmark curve over a price slice.

    Returns (account_curve, pct_returns, cum_returns).
    """
    price_ary = price_array[indx1:indx2]
    if price_ary.ndim != 2 or price_ary.shape[0] == 0:
        return None
    initial = price_ary[0, :]
    if np.any(initial <= 0):
        return None
    weights = np.array([1e6 / len(initial) / initial[i] for i in range(len(initial))])
    account = np.sum(weights * price_ary, axis=1)
    cumrets = account / account[0] - 1.0
    rets = account[1:] / account[:-1] - 1.0
    return account, rets, cumrets


def _sharpe(rets, factor):
    rets = np.asarray(rets, dtype=np.float64)
    if len(rets) < 2:
        return 0.0
    mean = np.nanmean(rets)
    vol = np.nanstd(rets, ddof=1)
    if not np.isfinite(vol) or vol == 0:
        return 0.0
    return float(np.sqrt(factor) * mean / vol)


def _max_drawdown(curve):
    curve = np.asarray(curve, dtype=np.float64)
    if len(curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(curve)
    dd = curve / np.where(peak == 0, np.nan, peak) - 1.0
    dd = np.nan_to_num(dd, nan=0.0)
    return float(dd.min())


def evaluate_ppo(account_values, price_test, timeframe, lookback=50):
    """Compare the PPO agent vs an equal-weight benchmark on the test window.

    Returns a dict of metrics (sharpe_bot, sharpe_eqw, excess_sharpe, ...).
    """
    account = np.asarray(account_values, dtype=np.float64)
    factor = compute_data_points_per_year(timeframe)

    if len(account) > 1:
        drl_rets = account[1:] / account[:-1] - 1.0
    else:
        drl_rets = np.array([0.0])
    sharpe_bot = _sharpe(drl_rets, factor)

    indx1 = max(0, int(lookback) - 1)
    indx2 = len(price_test) - int(lookback)
    eqw = compute_eqw(price_test, indx1, indx2)
    if eqw is None:
        return {
            "sharpe_bot": sharpe_bot, "sharpe_eqw": 0.0, "excess_sharpe": sharpe_bot,
            "cum_return_bot": float(account[-1] / account[0] - 1) if len(account) > 1 else 0.0,
            "cum_return_eqw": 0.0, "max_drawdown_bot": _max_drawdown(account),
            "max_drawdown_eqw": 0.0, "final_asset": float(account[-1]),
            "trades": len(account),
        }
    _, eqw_rets, eqw_cumrets = eqw
    sharpe_eqw = _sharpe(eqw_rets, factor)
    eqw_curve = eqw_cumrets + 1.0

    return {
        "sharpe_bot": sharpe_bot,
        "sharpe_eqw": sharpe_eqw,
        "excess_sharpe": sharpe_bot - sharpe_eqw,
        "cum_return_bot": float(account[-1] / account[0] - 1) if len(account) > 1 else 0.0,
        "cum_return_eqw": float(eqw_cumrets[-1]),
        "max_drawdown_bot": _max_drawdown(account),
        "max_drawdown_eqw": _max_drawdown(eqw_curve),
        "final_asset": float(account[-1]),
        "trades": len(account),
    }


def run_pipeline(symbols, interval="1h", limit=2000, lookback=50, total_timesteps=2000,
                 train_frac=0.8, model_dir=None, net_dimension=128, gpu_id=-1,
                 eval_time_gap=2.0, if_log=False):
    """End-to-end: fetch data -> build features -> train PPO -> test -> evaluate.

    Returns a dict {ohlcv, price_array, tech_array, split, train, account_values, metrics}.
    """
    if not is_available():
        raise RuntimeError("PPO unavailable: torch is not installed")

    ohlcv = {sym: fetch_klines(sym, interval, limit) for sym in symbols}
    missing = [sym for sym, df in ohlcv.items() if df is None]
    if missing:
        raise RuntimeError(f"Failed to fetch klines for: {missing}")
    price_array, tech_array = build_features(ohlcv)
    if price_array.shape[0] < lookback + 20:
        raise RuntimeError(f"Too little data: {price_array.shape[0]} rows")

    split = int(price_array.shape[0] * train_frac)
    if split < lookback + 5 or price_array.shape[0] - split < lookback + 5:
        raise RuntimeError("train/test split leaves too little data per side")

    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "ppo_models", f"{symbols[0]}_{interval}")

    env_params = default_env_params(lookback=lookback)

    train = train_ppo(price_array[:split], tech_array[:split], env_params, model_dir,
                      total_timesteps=total_timesteps, gpu_id=gpu_id,
                      net_dimension=net_dimension, eval_time_gap=eval_time_gap,
                      if_log=if_log)

    account_values = predict_ppo(price_array[split:], tech_array[split:], env_params,
                                 model_dir, net_dimension=net_dimension, gpu_id=gpu_id,
                                 if_log=if_log)

    metrics = evaluate_ppo(account_values, price_array[split:], interval, lookback)

    return {
        "ohlcv": ohlcv,
        "price_array": price_array,
        "tech_array": tech_array,
        "split": split,
        "train": train,
        "account_values": account_values,
        "metrics": metrics,
    }
