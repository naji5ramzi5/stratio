"""
ML PREDICTOR v2.0 – Multi-Model Ensemble + Online Learning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Models: MLP Neural Net + Random Forest + Gradient Boosting + XGBoost + Linear
Training: Offline (initial) + Online (every scan after verification)
Features: 50+ technical indicators + market context
"""

import os, sys, json, pickle, warnings, logging
from datetime import datetime
from copy import deepcopy
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger("MLTrainer")

from settings import (MODELS_DIR, TRAIN_LOOKBACK, INFERENCE_LOOKBACK,
                      MAX_FEATURES, HORIZON_BARS, INTERVAL_TF,
                      RANDOM_STATE, PIPELINE_VERSION, CALIBRATION_PATH,
                      TRAINING_BUFFER_FILE, seed_everything)
from utils.state_store import load_json, save_json, safe_float

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_BUFFER = os.path.join(BASE_DIR, TRAINING_BUFFER_FILE)
os.makedirs(MODELS_DIR, exist_ok=True)

_calibration_cache = None


def _load_calibration():
    global _calibration_cache
    if _calibration_cache is None:
        try:
            with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
                _calibration_cache = json.load(f)
        except Exception:
            _calibration_cache = {}
    return _calibration_cache


def _calibration_bin(agreement_conf):
    """Map a raw model-agreement % to the realised out-of-sample accuracy
    recorded for that bin in calibration.json (returns 0-1 or None)."""
    cal = _load_calibration()
    if not cal:
        return None
    a = float(agreement_conf)
    for lo, hi in [(50, 60), (60, 70), (70, 80), (80, 90), (90, 101)]:
        if lo <= a < hi:
            key = f"{lo}-{hi}"
            if key == "90-101":
                key = "90-100"
            if key in cal:
                return float(cal[key])
            return None
    return None

# ─── IMPORTS ──────────────────────────────────────────────────────
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_absolute_error
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

BINANCE_BASE = "https://api.binance.com"  # kept for back-compat

KLINES_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume",
                  "close_time", "quote_vol", "trades", "taker_buy_vol",
                  "taker_buy_quote", "ignore"]
NUMERIC_COLS = ["open", "high", "low", "close", "volume"]


def fetch_klines(symbol, interval="1h", limit=500):
    """Fetch klines (oldest -> newest) with backward pagination.

    Delegates to the canonical loader in data_loader.binance_ohlcv so the
    whole ML/eval stack shares one network implementation.
    """
    from data_loader.binance_ohlcv import fetch_klines as _fetch
    return _fetch(symbol, interval=interval, limit=limit)


def fetch_btc_returns(interval="1h", lookback=1500, min_samples=200):
    """Fetch BTCUSDT close returns as a time-aligned frame (timestamp, btc_ret_1)."""
    raw = fetch_klines("BTCUSDT", interval, lookback)
    if not raw or len(raw) < min_samples:
        return None
    btc_df = pd.DataFrame(raw, columns=KLINES_COLUMNS)
    for c in NUMERIC_COLS:
        btc_df[c] = btc_df[c].astype(float)
    out = btc_df[["timestamp"]].copy()
    out["btc_ret_1"] = btc_df["close"].pct_change()
    return out


def merge_btc_features(df, btc_ret_df):
    """Add time-aligned BTC features to ``df``.

    Uses the SAME causal formula as live inference (get_cached_features): a
    rolling correlation of the symbol's own returns against BTC's, aligned on
    the timestamp column. Rows without a BTC bar become NaN and must be
    dropped by the caller. (The old training code correlated against the tail
    of the whole series — a look-ahead leak.)
    """
    if btc_ret_df is None:
        return df
    df = df.merge(btc_ret_df, on="timestamp", how="left")
    df["btc_corr_10"] = df["ret_1"].rolling(10).corr(df["btc_ret_1"])
    return df


# ═══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (50+ features)
# ═══════════════════════════════════════════════════════════════════

FEATURE_NAMES = None  # set after first training

def compute_features(df):
    c = df["close"]
    h = df["high"]
    l_ = df["low"]
    o = df["open"]
    v = df["volume"]

    # ── Price returns ─────────────────────────────────────────
    for p in [1, 2, 3, 5, 8, 10, 13, 21]:
        df[f"ret_{p}"] = c.pct_change(p)
    df["log_ret_1"] = np.log(c / c.shift(1))
    df["hl_pct"] = (h - l_) / l_ * 100
    df["co_pct"] = (c - o) / o * 100
    df["ho_pct"] = (h - o) / o * 100
    df["lo_pct"] = (l_ - o) / o * 100

    # ── Volume ────────────────────────────────────────────────
    df["v_ret_1"] = v.pct_change(1)
    df["v_ret_5"] = v.pct_change(5)
    df["v_ma5"] = v.rolling(5).mean()
    df["v_ma20"] = v.rolling(20).mean()
    df["v_ratio_5_20"] = df["v_ma5"] / (df["v_ma20"] + 1e-10)
    df["v_ratio_1_5"] = v / (df["v_ma5"] + 1e-10)
    df["v_dollar"] = v * c

    # ── RSI (3 variants) ──────────────────────────────────────
    for period in [6, 14, 21]:
        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    # ── MACD variants ─────────────────────────────────────────
    for fast, slow, signal in [(12, 26, 9), (8, 21, 5), (5, 13, 3)]:
        ema_f = c.ewm(span=fast).mean()
        ema_s = c.ewm(span=slow).mean()
        macd = ema_f - ema_s
        sig = macd.ewm(span=signal).mean()
        df[f"macd_{fast}_{slow}"] = macd
        df[f"macd_sig_{fast}_{slow}"] = sig
        df[f"macd_h_{fast}_{slow}"] = macd - sig
        df[f"macd_cross_{fast}_{slow}"] = ((macd > sig).astype(int).diff())

    # ── Bollinger Bands ───────────────────────────────────────
    for period in [10, 20, 50]:
        sma = c.rolling(period).mean()
        std = c.rolling(period).std()
        df[f"bb_{period}_w"] = ((sma + 2*std) - (sma - 2*std)) / sma * 100
        df[f"bb_{period}_p"] = (c - sma) / (2*std + 1e-10)
        df[f"bb_{period}_b"] = (c - (sma - 2*std)) / ((sma + 2*std) - (sma - 2*std) + 1e-10)
        df[f"bb_{period}_bw"] = ((c > (sma + 2*std)) | (c < (sma - 2*std))).astype(int)

    # ── EMAs ──────────────────────────────────────────────────
    for p in [5, 8, 9, 13, 21, 34, 50, 200]:
        df[f"ema_{p}"] = c.ewm(span=p).mean()
        df[f"p_ema_{p}"] = (c - df[f"ema_{p}"]) / df[f"ema_{p}"] * 100
    df["ema_9_21"] = (df["ema_9"] - df["ema_21"]) / df["ema_21"] * 100
    df["ema_21_50"] = (df["ema_21"] - df["ema_50"]) / df["ema_50"] * 100
    df["ema_50_200"] = (df["ema_50"] - df["ema_200"]) / df["ema_200"] * 100
    df["ema_short_mid"] = df["ema_9"] / df["ema_50"] - 1
    df["ema_mid_long"] = df["ema_50"] / df["ema_200"] - 1

    # ── ATR variants ──────────────────────────────────────────
    tr = pd.concat([
        h - l_,
        (h - c.shift(1)).abs(),
        (l_ - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    for p in [7, 14, 21]:
        df[f"atr_{p}"] = tr.rolling(p).mean()
        df[f"atr_{p}_pct"] = df[f"atr_{p}"] / c * 100
    df["atr_ratio_7_21"] = df["atr_7"] / (df["atr_21"] + 1e-10)

    # ── OBV ───────────────────────────────────────────────────
    obv = (v * np.where(c > o, 1, np.where(c < o, -1, 0))).cumsum()
    df["obv"] = obv
    for p in [5, 10, 20]:
        df[f"obv_ma_{p}"] = obv.rolling(p).mean()
        df[f"obv_ratio_{p}"] = obv / (df[f"obv_ma_{p}"] + 1e-10)

    # ── Stochastic ────────────────────────────────────────────
    for p in [7, 14, 21]:
        ll = l_.rolling(p).min()
        hh = h.rolling(p).max()
        df[f"stoch_k_{p}"] = ((c - ll) / (hh - ll + 1e-10)) * 100
        df[f"stoch_d_{p}"] = df[f"stoch_k_{p}"].rolling(3).mean()

    # ── Williams %R ───────────────────────────────────────────
    for p in [7, 14]:
        ll = l_.rolling(p).min()
        hh = h.rolling(p).max()
        df[f"williams_r_{p}"] = ((hh - c) / (hh - ll + 1e-10)) * -100

    # ── CCI ───────────────────────────────────────────────────
    for p in [10, 20]:
        tp = (h + l_ + c) / 3
        sma_tp = tp.rolling(p).mean()
        mad_tp = tp.rolling(p).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        df[f"cci_{p}"] = (tp - sma_tp) / (0.015 * mad_tp + 1e-10)

    # ── ADX ───────────────────────────────────────────────────
    up = h.diff()
    down = -l_.diff()
    tr14 = tr.rolling(14).mean()
    for p in [14]:
        plus_dm = ((up > down) & (up > 0)).astype(int) * up
        minus_dm = ((down > up) & (down > 0)).astype(int) * down
        adx_plus = (plus_dm.rolling(p).mean() / tr14).fillna(0)
        adx_minus = (minus_dm.rolling(p).mean() / tr14).fillna(0)
        df[f"adx_{p}"] = 100 * ((adx_plus - adx_minus).abs() / (adx_plus + adx_minus + 1e-10))
        df[f"adx_di_plus_{p}"] = adx_plus
        df[f"adx_di_minus_{p}"] = adx_minus

    # ── MFI ───────────────────────────────────────────────────
    for p in [7, 14]:
        tp = (h + l_ + c) / 3
        mf = tp * v
        pos = mf.where(tp > tp.shift(1), 0).rolling(p).sum()
        neg = mf.where(tp < tp.shift(1), 0).rolling(p).sum()
        mfr = pos / (neg + 1e-10)
        df[f"mfi_{p}"] = 100 - (100 / (1 + mfr))

    # ── Volatility ────────────────────────────────────────────
    df["vty_5"] = c.pct_change().rolling(5).std() * 100
    df["vty_10"] = c.pct_change().rolling(10).std() * 100
    df["vty_20"] = c.pct_change().rolling(20).std() * 100
    df["vty_ratio_5_20"] = df["vty_5"] / (df["vty_20"] + 1e-10)
    df["vty_ratio_10_20"] = df["vty_10"] / (df["vty_20"] + 1e-10)

    # ── Price patterns ────────────────────────────────────────
    df["body"] = (c - o).abs()
    df["upper_wick"] = h - pd.concat([c, o], axis=1).max(axis=1)
    df["lower_wick"] = pd.concat([c, o], axis=1).min(axis=1) - l_
    df["body_ratio"] = df["body"] / (h - l_ + 1e-10)
    df["doji"] = ((df["body"] < (df["high"] - df["low"]) * 0.1)).astype(int)
    df["marubozu"] = ((df["upper_wick"] < df["body"] * 0.1) & (df["lower_wick"] < df["body"] * 0.1)).astype(int)
    df["hammer"] = ((df["lower_wick"] > df["body"] * 2) & (df["upper_wick"] < df["body"] * 0.3)).astype(int)
    df["shooting_star"] = ((df["upper_wick"] > df["body"] * 2) & (df["lower_wick"] < df["body"] * 0.3)).astype(int)

    # ── Correlation with BTC ──────────────────────────────────
    # (added externally in prepare_data)

    return df.replace([np.inf, -np.inf], np.nan)


# ═══════════════════════════════════════════════════════════════════
# MODEL FACTORY
# ═══════════════════════════════════════════════════════════════════

def create_models(random_state=42):
    models = {
        "mlp": MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu", solver="adam", alpha=0.001,
            batch_size=64, learning_rate="adaptive",
            max_iter=200, early_stopping=True,
            validation_fraction=0.15, random_state=random_state
        ),
        "rf": RandomForestRegressor(
            n_estimators=200, max_depth=12,
            min_samples_leaf=5, max_features='sqrt',
            n_jobs=2, random_state=random_state
        ),
        "gbr": GradientBoostingRegressor(
            n_estimators=120, max_depth=5,
            learning_rate=0.05, subsample=0.8,
            min_samples_leaf=5, max_features='sqrt',
            random_state=random_state
        ),
        "lr": Ridge(alpha=1.0),
        "huber": HuberRegressor(epsilon=1.35, alpha=0.0001),
    }
    if HAS_XGB:
        models["xgb"] = XGBRegressor(
            n_estimators=120, max_depth=5,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=random_state, verbosity=0
        )
    return models


# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════

def prepare_training_data(symbols, interval="1h", lookback=1500, min_samples=300):
    all_X, all_y = [], []
    global FEATURE_NAMES

    btc_ret_df = fetch_btc_returns(interval, lookback, min_samples=min_samples)

    for sym in symbols:
        raw = fetch_klines(sym, interval, lookback)
        if not raw or len(raw) < min_samples:
            continue
        df = pd.DataFrame(raw, columns=KLINES_COLUMNS)
        for c in NUMERIC_COLS:
            df[c] = df[c].astype(float)
        df = compute_features(df)

        # BTC correlation feature (time-aligned, causal, same formula as live).
        df = merge_btc_features(df, btc_ret_df)

        df = df.dropna().reset_index(drop=True)
        if len(df) < 60:
            continue

        # Map interval to the correct tf_hours
        target_tf = INTERVAL_TF.get(interval, 24)
        n_fwd = HORIZON_BARS[target_tf]
        fwd_ret = df["close"].shift(-n_fwd) / df["close"] - 1
        target_col = f"target_{target_tf}h"
        df[target_col] = fwd_ret * 100

        exclude = {"timestamp", "open", "high", "low", "close", "volume",
                   "close_time", "quote_vol", "trades", "taker_buy_vol",
                   "taker_buy_quote", "ignore"}
        target_cols = [c for c in df.columns if c.startswith("target_")]
        feats = [c for c in df.columns if c not in exclude and c not in target_cols]
        FEATURE_NAMES = feats
        X = df[feats].astype(float).values
        y = df[target_col].astype(float).values

        mask = ~np.isnan(y)
        all_X.append(X[mask])
        all_y.append(y[mask])

    if not all_X:
        return None, None
    return np.vstack(all_X), np.concatenate(all_y)


def _walk_forward_validate(X, y, model_class, model_params, n_splits=3):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        m = model_class(**model_params)
        m.fit(X_tr, y_tr)
        pred = m.predict(X_val)
        scores.append(-float(mean_absolute_error(y_val, pred)))
    return np.mean(scores) if scores else -999


def _quick_oos_report(X_scaled, y, fit_cut):
    """Directional accuracy + MAE of a fast RF on the last 20% (honest OOS).

    The fold is never part of this report's fit, so the number is a fair
    (if rough) estimate of the ensemble's generalisation.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        n_test = len(X_scaled) - fit_cut
        if n_test < 50 or fit_cut < 200:
            return None, None
        m = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(X_scaled[:fit_cut], y[:fit_cut])
        pred = m.predict(X_scaled[fit_cut:])
        actual = y[fit_cut:]
        correct = int(np.mean((pred > 0) == (actual > 0)) * n_test)
        accuracy = correct / n_test * 100.0
        mae = float(mean_absolute_error(actual, pred))
        logger.info(f"  OOS (last 20%): direction_accuracy={accuracy:.1f}% "
                    f"mae={mae:.3f} (n={n_test})")
        return round(accuracy, 1), round(mae, 3)
    except Exception as exc:
        logger.debug(f"  OOS report skipped: {exc}")
        return None, None


def _select_features(X, y, max_features=50, fit_mask=None):
    if fit_mask is None:
        fit_mask = np.ones(len(X), dtype=bool)
    if X.shape[1] <= max_features:
        return X, list(range(X.shape[1]))
    try:
        selector = SelectKBest(score_func=mutual_info_regression, k=min(max_features, X.shape[1] - 1))
        selector.fit(X[fit_mask], y[fit_mask])
        X_selected = selector.transform(X)
        selected_indices = selector.get_support(indices=True)
        logger.info(f"  Feature selection: {X.shape[1]} -> {X_selected.shape[1]} features")
        return X_selected, selected_indices
    except Exception as e:
        logger.warning(f"  Feature selection failed: {e}")
        return X, list(range(X.shape[1]))


def train_models(X, y, tf_hours=24):
    models = create_models()
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    if len(X) < 100:
        logger.warning(f"  [{tf_hours}h] Too few samples after NaN removal: {len(X)}")
        return {"models": {}, "scaler": StandardScaler(), "features": FEATURE_NAMES, "feature_count": 0, "weights": {}, "performance": {}, "tf_hours": tf_hours, "trained_at": datetime.now().isoformat()}

    # Fit the selector and scaler on the PAST 80% only (temporal split), then
    # transform the whole set. Previously both were fit on the full matrix,
    # letting validation/weight steps see statistics from their own fold.
    fit_cut = int(len(X) * 0.8)
    fit_mask = np.zeros(len(X), dtype=bool)
    fit_mask[:fit_cut] = True

    X, selected_features = _select_features(X, y, fit_mask=fit_mask)

    scaler = StandardScaler()
    scaler.fit(X[:fit_cut])
    X_scaled = scaler.transform(X)

    # Honest out-of-sample report: a quick RandomForest trained on the first
    # 80% and evaluated on the last 20% (never touched by that fold's fit).
    oos_accuracy, oos_mae = _quick_oos_report(X_scaled, y, fit_cut)

    results = {}
    wf_scores = {}
    for name, model in models.items():
        try:
            model.fit(X_scaled, y)
            score = model.score(X_scaled, y)
            results[name] = {"model": model, "score": score}
            logger.info(f"  [{tf_hours}h] {name}: score={score:.4f}")
        except Exception as e:
            logger.warning(f"  [{tf_hours}h] {name} failed: {e}")

    # Optimize best models with Optuna (off by default; enable via ML_OPTUNA=1).
    # Previously this block ran the trials but never applied the tuned params,
    # silently wasting minutes per timeframe.
    if (os.environ.get("ML_OPTUNA") == "1" and HAS_OPTUNA
            and len(X) >= 200):
        from optimizer import optimize_xgb, optimize_rf
        for model_name, opt_func in [('xgb', optimize_xgb), ('rf', optimize_rf)]:
            if model_name in results:
                opt_params = opt_func(X_scaled, y, n_trials=20)
                if opt_params and 'random_state' in opt_params:
                    logger.info(f"  [{tf_hours}h] Applying Optuna for {model_name}")
                    try:
                        from xgboost import XGBRegressor
                        opt_params["random_state"] = 42
                        results[model_name] = {"model": XGBRegressor(**opt_params),
                                               "score": 0.0}
                        results[model_name]["model"].fit(X_scaled, y)
                    except Exception as e:
                        logger.warning(f"  [{tf_hours}h] Optuna refit failed: {e}")

    # Walk-forward MAE per model to set ensemble weights (best generalizers win)
    try:
        logger.info(f"  Walk-forward validation [{tf_hours}h]:")
        for name, md in results.items():
            try:
                from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                from sklearn.linear_model import Ridge
                model_map = {
                    'rf': (RandomForestRegressor, {'n_estimators': 100, 'random_state': 42}),
                    'gbr': (GradientBoostingRegressor, {'n_estimators': 80, 'random_state': 42}),
                    'lr': (Ridge, {'alpha': 1.0}),
                }
                if name in model_map:
                    cls, params = model_map[name]
                    wf = _walk_forward_validate(X_scaled, y, cls, params)
                    wf_scores[name] = wf
                    logger.info(f"    {name}: wf_mae={-wf:.4f}")
            except Exception as e:
                logger.debug(f"    {name}: wf skipped ({e})")
    except Exception as e:
        logger.debug(f"  Walk-forward skipped: {e}")

    weights = {}
    if wf_scores:
        min_wf = min(wf_scores.values())
        for name in results:
            wf = wf_scores.get(name, min_wf)
            weights[name] = max(0.1, (wf - min_wf + 0.001))
        total_w = sum(weights.values())
        if total_w > 0:
            for name in weights:
                weights[name] /= total_w
        logger.info(f"  Dynamic weights: { {k: round(v, 3) for k, v in weights.items()} }")

    return {
        "models": results,
        "scaler": scaler,
        "features": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES) if FEATURE_NAMES else 0,
        "selected_features": selected_features if FEATURE_NAMES else [],
        "weights": weights or {name: 1.0 / len(results) for name in results},
        "performance": {name: {"correct": 0, "total": 0} for name in results},
        "tf_hours": tf_hours,
        "trained_at": datetime.now().isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "oos_direction_accuracy_pct": oos_accuracy,
        "oos_mae": oos_mae,
        "oos_n": len(X) - fit_cut if fit_cut is not None else None,
    }


def train_all(symbols=None, lookback=1500):
    logger.info("=" * 50)
    logger.info("ML TRAINER v2.0 – Training multi-model ensemble")
    logger.info("=" * 50)
    seed_everything(RANDOM_STATE)

    if symbols is None:
        from advanced_bot import get_qualified_symbols
        symbols = get_qualified_symbols()[:50]
    symbols = [s.upper().replace("USDT", "USDT") if s.upper().endswith("USDT") else s.upper() + "USDT"
               for s in symbols]
    logger.info(f"Symbols: {len(symbols)} | lookback: {lookback}")

    all_models = {}
    for tf_hours, interval in [(2, "5m"), (6, "15m"), (8, "30m"), (24, "1h")]:
        logger.info(f"\n─── Training {tf_hours}h models ({interval}) ───")
        X, y = prepare_training_data(symbols, interval, lookback=lookback,
                                     min_samples=200)
        if X is None:
            logger.warning(f"  [{tf_hours}h] No data, skipping")
            continue
        logger.info(f"  Samples: {len(X)}, Features: {len(FEATURE_NAMES) if FEATURE_NAMES else 0}")
        result = train_models(X, y, tf_hours)
        all_models[f"{tf_hours}h"] = result

    if not all_models:
        logger.warning("No models trained!")
        return False

    model_path = os.path.join(MODELS_DIR, "ensemble_v2.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(all_models, f)
    logger.info(f"\n✅ Models saved: {list(all_models.keys())}")
    return True


# ═══════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════

_model_cache = None
_feature_cache = {}  # {(symbol, interval): df}

def load_models():
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    path = os.path.join(MODELS_DIR, "ensemble_v2.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        _model_cache = pickle.load(f)
    return _model_cache


def get_cached_features(symbol, interval, lookback):
    key = (symbol, interval)
    if key in _feature_cache:
        return _feature_cache[key]
    raw = fetch_klines(symbol, interval, lookback)
    if not raw or len(raw) < 60:
        _feature_cache[key] = None
        return None
    df = pd.DataFrame(raw, columns=KLINES_COLUMNS)
    for c in NUMERIC_COLS:
        df[c] = df[c].astype(float)
    df = compute_features(df)

    # Add BTC features (time-aligned, cached BTC ret frame)
    btc_key = ("BTCUSDT", interval)
    if btc_key not in _feature_cache:
        _feature_cache[btc_key] = fetch_btc_returns(interval, lookback, min_samples=20)
    df = merge_btc_features(df, _feature_cache[btc_key])

    df = df.dropna().reset_index(drop=True)
    _feature_cache[key] = df
    return df


def predict_all(symbol, tf_hours):
    """Run all models and return ensemble prediction + feature vector."""
    all_models = load_models()
    if all_models is None:
        return None, None

    key = f"{tf_hours}h"
    if key not in all_models:
        return None, None

    ensemble = all_models[key]
    if not ensemble["models"]:
        return None, None

    if tf_hours <= 2:
        interval, lookback = "5m", INFERENCE_LOOKBACK
    elif tf_hours <= 6:
        interval, lookback = "15m", INFERENCE_LOOKBACK
    elif tf_hours <= 8:
        interval, lookback = "30m", INFERENCE_LOOKBACK
    else:
        interval, lookback = "1h", INFERENCE_LOOKBACK

    df = get_cached_features(symbol, interval, lookback)
    if df is None or len(df) < 2:
        return None, None

    feats = ensemble["features"]
    available = [f for f in feats if f in df.columns]
    if len(available) < len(feats) * 0.3:
        return None, None

    # Models were trained on the features selected by SelectKBest at training
    # time.  Build the vector ONLY from those columns, otherwise the scaler /
    # estimators (fit on the reduced matrix) raise and the ensemble silently
    # returns None (the bug that killed ml_pct in production).
    selected = ensemble.get("selected_features")
    if selected is not None and len(selected) > 0 and len(selected) < len(feats):
        selected = [int(i) for i in selected]
    else:
        selected = list(range(len(feats)))

    X = np.zeros((1, len(selected)))
    for j, idx in enumerate(selected):
        f = feats[idx]
        if f in df.columns:
            try:
                X[0, j] = float(df[f].values[-1])
            except (ValueError, TypeError):
                X[0, j] = 0.0

    scaler = ensemble["scaler"]
    X_scaled = scaler.transform(X)

    predictions = []
    weights = ensemble.get("weights", {})
    for name, md in ensemble["models"].items():
        try:
            pred = float(md["model"].predict(X_scaled)[0])
            if np.isnan(pred) or np.isinf(pred):
                continue
            w = weights.get(name, 1.0)
            predictions.append((pred, w, name))
        except Exception:
            continue

    if not predictions:
        return None, None

    # Weighted ensemble
    total_w = sum(w for _, w, _ in predictions)
    ensemble_pred = sum(p * w / total_w for p, w, _ in predictions) if total_w > 0 else 0
    if np.isnan(ensemble_pred) or np.isinf(ensemble_pred):
        ensemble_pred = 0

    # Confidence based on model agreement
    directions = [p > 0 for p, _, _ in predictions]
    agreement = sum(directions) / len(directions) if directions else 0.5
    agreement_conf = max(agreement, 1 - agreement) * 100

    # Calibrate the agreement against realised out-of-sample accuracy
    # (models/calibration.json from calibrate_predictor.py). The calibrated
    # value is the honest expected hit-rate for this confidence bin. Missing
    # bins fall back to the overall realised accuracy rather than the raw,
    # uncalibrated agreement.
    cal = _calibration_bin(agreement_conf)
    if cal is None:
        cal = _load_calibration().get("overall")
    honest_conf = float(cal) * 100.0 if cal is not None else agreement_conf

    feature_vec = X[0].tolist()
    feature_vec = [float(x) if not (np.isnan(x) or np.isinf(x)) else 0.0 for x in feature_vec]

    result = {
        "predicted_change_pct": round(float(ensemble_pred), 2),
        "confidence": round(float(honest_conf), 1),
        "raw_agreement": round(float(agreement_conf), 1),
        "model_predictions": {name: round(float(p), 4) for p, _, name in predictions},
        "num_models": len(predictions),
        "feature_vector": feature_vec
    }

    try:
        from regime_gate import gate_signal
        gate = gate_signal(symbol, tf_hours, result, df=df)
        result["regime_trend"] = gate.get("regime", {}).get("trend")
        result["regime_gate"] = gate["status"]
        result["gate_reason"] = gate["reason"]
        result["tradable"] = gate["tradable"]
    except Exception as exc:
        logger.debug("regime gate failed: %s", exc)
        result["regime_gate"] = "UNAVAILABLE"
        result["gate_reason"] = "gate error"
        result["tradable"] = False

    return result, feature_vec


# ═══════════════════════════════════════════════════════════════════
# ONLINE LEARNING
# ═══════════════════════════════════════════════════════════════════

def clear_feature_cache():
    global _feature_cache
    _feature_cache = {}

def save_training_sample(feature_vector, symbol, tf_hours, predicted_pct, actual_pct=None):
    """Save a training sample to buffer for later online learning."""
    samples = []
    if os.path.exists(TRAINING_BUFFER):
        try:
            with open(TRAINING_BUFFER, encoding="utf-8") as f:
                samples = json.load(f)
        except Exception:
            samples = []
    samples.append({
        "features": feature_vector,
        "symbol": symbol,
        "tf_hours": tf_hours,
        "predicted_pct": predicted_pct,
        "actual_pct": actual_pct,
        "learned": False,
        "created_at": datetime.now().isoformat()
    })
    samples = samples[-20000:]  # keep last 20k
    try:
        with open(TRAINING_BUFFER, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False)
    except Exception:
        pass


def update_outcome(symbol, tf_hours, created_at, actual_pct):
    """Find a matching sample and update with actual outcome."""
    if not os.path.exists(TRAINING_BUFFER):
        return
    try:
        with open(TRAINING_BUFFER, encoding="utf-8") as f:
            samples = json.load(f)
        updated = False
        for s in samples:
            if (s.get("symbol") == symbol and s.get("tf_hours") == tf_hours
                    and s.get("actual_pct") is None and not s.get("learned")):
                s["actual_pct"] = actual_pct
                updated = True
                break
        if updated:
            with open(TRAINING_BUFFER, "w", encoding="utf-8") as f:
                json.dump(samples, f, ensure_ascii=False)
    except Exception:
        pass


def online_learn():
    """Update models with newly verified samples using partial_fit."""
    all_models = load_models()
    if all_models is None:
        logger.info("[ONLINE] No ensemble model loaded, skipping online learning")
        return 0

    if not os.path.exists(TRAINING_BUFFER):
        return 0

    try:
        with open(TRAINING_BUFFER, encoding="utf-8") as f:
            samples = json.load(f)
    except Exception:
        return 0

    unlearned = [s for s in samples if not s.get("learned") and s.get("actual_pct") is not None]
    if not unlearned:
        return 0

    # Group by timeframe
    by_tf = {}
    for s in unlearned:
        tf = s["tf_hours"]
        fv = s["features"]
        if not fv or len(fv) == 0:
            continue
        if tf not in by_tf:
            by_tf[tf] = {"X": [], "y": []}
        by_tf[tf]["X"].append(fv)
        by_tf[tf]["y"].append(s["actual_pct"])

    total_learned = 0
    for tf_key, data in by_tf.items():
        key = f"{tf_key}h"
        if key not in all_models:
            continue
        ensemble = all_models[key]
        if not ensemble["models"]:
            continue

        X_new = np.array(data["X"])
        y_new = np.array(data["y"])
        scaler = ensemble["scaler"]

        # Feature count the scaler/models actually expect (post-selection)
        selected = ensemble.get("selected_features")
        expected = len(selected) if (selected is not None
                                     and len(selected) > 0) else ensemble["feature_count"]
        if X_new.shape[1] != expected:
            logger.warning(f"[ONLINE] {key}: feature mismatch {X_new.shape[1]} vs {expected}")
            continue

        X_scaled = scaler.transform(X_new)

        for name, md in ensemble["models"].items():
            model = md["model"]
            try:
                if hasattr(model, "partial_fit"):
                    model.partial_fit(X_scaled, y_new)
                elif hasattr(model, "fit"):
                    # For models without partial_fit, do quick retrain
                    model.fit(X_scaled, y_new)
                total_learned += len(X_new)

                # Update performance tracking
                preds = model.predict(X_scaled)
                for p, a in zip(preds, y_new):
                    perf = ensemble["performance"][name]
                    perf["total"] = perf.get("total", 0) + 1
                    if (p > 0 and a > 0) or (p < 0 and a < 0):
                        perf["correct"] = perf.get("correct", 0) + 1

                # Adjust weight based on recent accuracy
                perf = ensemble["performance"][name]
                total = perf.get("total", 1)
                correct = perf.get("correct", 0)
                accuracy = correct / total if total > 0 else 0.5
                ensemble["weights"][name] = max(0.5, accuracy * 2)
            except Exception as e:
                logger.warning(f"[ONLINE] {key}/{name}: {e}")

    # Mark samples as learned
    for s in samples:
        if not s.get("learned") and s.get("actual_pct") is not None:
            s["learned"] = True
    try:
        with open(TRAINING_BUFFER, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False)
        with open(os.path.join(MODELS_DIR, "ensemble_v2.pkl"), "wb") as f:
            pickle.dump(all_models, f)
    except Exception as e:
        logger.warning(f"[ONLINE] Save failed: {e}")

    if total_learned:
        logger.info(f"[ONLINE] Learned from {total_learned} new samples across {len(by_tf)} timeframes")
    return total_learned


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train_all()
