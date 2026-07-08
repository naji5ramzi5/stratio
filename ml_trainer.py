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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
TRAINING_BUFFER = os.path.join(BASE_DIR, "training_buffer.json")
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── IMPORTS ──────────────────────────────────────────────────────
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

import requests
BINANCE_BASE = "https://api.binance.com"

def _req(endpoint, params=None):
    try:
        r = requests.get(BINANCE_BASE + endpoint, params=params, timeout=8)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def fetch_klines(symbol, interval="1h", limit=500):
    data = _req("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    return data if isinstance(data, list) else []


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
            hidden_layer_sizes=(64, 32),
            activation="relu", solver="adam", alpha=0.01,
            batch_size=128, learning_rate="adaptive",
            max_iter=100, early_stopping=True,
            validation_fraction=0.1, random_state=random_state
        ),
        "rf": RandomForestRegressor(
            n_estimators=100, max_depth=8,
            min_samples_leaf=10, n_jobs=2,
            random_state=random_state
        ),
        "gbr": GradientBoostingRegressor(
            n_estimators=80, max_depth=4,
            learning_rate=0.1, subsample=0.8,
            random_state=random_state
        ),
        "lr": LinearRegression(),
    }
    if HAS_XGB:
        models["xgb"] = XGBRegressor(
            n_estimators=80, max_depth=4,
            learning_rate=0.1, subsample=0.8,
            random_state=random_state, verbosity=0
        )
    return models


# ═══════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════

def prepare_training_data(symbols, interval="1h", lookback=1000, min_samples=300):
    all_X, all_y = [], []
    global FEATURE_NAMES

    btc_returns = None
    btc_raw = fetch_klines("BTCUSDT", interval, lookback)
    if btc_raw and len(btc_raw) > min_samples:
        btc_df = pd.DataFrame(btc_raw, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_buy_vol", "taker_buy_quote", "ignore"
        ])
        for c in ["open", "high", "low", "close", "volume"]:
            btc_df[c] = btc_df[c].astype(float)
        btc_returns = btc_df["close"].pct_change().values

    for sym in symbols:
        raw = fetch_klines(sym, interval, lookback)
        if not raw or len(raw) < min_samples:
            continue
        cols = ["timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_vol", "trades", "taker_buy_vol", "taker_buy_quote", "ignore"]
        df = pd.DataFrame(raw, columns=cols)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        df = compute_features(df)
        df = df.dropna().reset_index(drop=True)
        if len(df) < 60:
            continue

        # BTC correlation feature
        if btc_returns is not None:
            btc_aligned = btc_returns[-len(df):]
            df["btc_corr_10"] = pd.Series(btc_aligned).rolling(10).apply(
                lambda x: np.corrcoef(x, df["ret_1"].values[-len(x):])[0, 1] if len(x) > 5 else 0
            )
            df["btc_ret_1"] = btc_aligned

        # Map interval to the correct tf_hours
        interval_map_rev = {"5m": 2, "15m": 6, "30m": 8, "1h": 24}
        target_tf = interval_map_rev.get(interval, 24)
        n_fwd = {2: 24, 6: 24, 8: 16, 24: 24}[target_tf]
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


def train_models(X, y, tf_hours=24):
    models = create_models()
    # Drop NaN — ensure numeric
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    if len(X) < 100:
        logger.warning(f"  [{tf_hours}h] Too few samples after NaN removal: {len(X)}")
        return {"models": {}, "scaler": StandardScaler(), "features": FEATURE_NAMES, "feature_count": 0, "weights": {}, "performance": {}, "tf_hours": tf_hours, "trained_at": datetime.now().isoformat()}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}
    for name, model in models.items():
        try:
            model.fit(X_scaled, y)
            score = model.score(X_scaled, y)
            results[name] = {"model": model, "score": score}
            logger.info(f"  [{tf_hours}h] {name}: score={score:.4f}")
        except Exception as e:
            logger.warning(f"  [{tf_hours}h] {name} failed: {e}")

    return {
        "models": results,
        "scaler": scaler,
        "features": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES) if FEATURE_NAMES else 0,
        "weights": {name: 1.0 for name in results},
        "performance": {name: {"correct": 0, "total": 0} for name in results},
        "tf_hours": tf_hours,
        "trained_at": datetime.now().isoformat()
    }


def train_all():
    logger.info("=" * 50)
    logger.info("ML TRAINER v2.0 – Training multi-model ensemble")
    logger.info("=" * 50)

    from advanced_bot import get_qualified_symbols
    symbols = get_qualified_symbols()[:50]
    logger.info(f"Symbols: {len(symbols)} | lookback: 1000")

    all_models = {}
    for tf_hours, interval in [(2, "5m"), (6, "15m"), (8, "30m"), (24, "1h")]:
        logger.info(f"\n─── Training {tf_hours}h models ({interval}) ───")
        X, y = prepare_training_data(symbols, interval, min_samples=200)
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
    cols = ["timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_buy_vol", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df = compute_features(df)

    # Add BTC features (cached)
    btc_key = ("BTCUSDT", interval)
    if btc_key not in _feature_cache:
        btc_raw = fetch_klines("BTCUSDT", interval, lookback)
        if btc_raw and len(btc_raw) > 20:
            btc_df = pd.DataFrame(btc_raw, columns=cols)
            for c_ in ["open", "high", "low", "close", "volume"]:
                btc_df[c_] = btc_df[c_].astype(float)
            _feature_cache[btc_key] = btc_df
        else:
            _feature_cache[btc_key] = None
    btc_df = _feature_cache[btc_key]
    if btc_df is not None and len(btc_df) >= len(df):
        btc_ret = btc_df["close"].pct_change().values[-len(df):]
        df["btc_ret_1"] = btc_ret
        df["btc_corr_10"] = df["ret_1"].rolling(10).corr(pd.Series(btc_ret))

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
        interval, lookback = "5m", 100
    elif tf_hours <= 6:
        interval, lookback = "15m", 100
    elif tf_hours <= 8:
        interval, lookback = "30m", 100
    else:
        interval, lookback = "1h", 200

    df = get_cached_features(symbol, interval, lookback)
    if df is None or len(df) < 2:
        return None, None

    feats = ensemble["features"]
    available = [f for f in feats if f in df.columns]
    if len(available) < len(feats) * 0.3:
        return None, None

    # Build feature vector matching training order
    X = np.zeros((1, len(feats)))
    for i, f in enumerate(feats):
        if f in df.columns:
            try:
                X[0, i] = float(df[f].values[-1])
            except (ValueError, TypeError):
                X[0, i] = 0.0

    scaler = ensemble["scaler"]
    X_scaled = scaler.transform(X)

    predictions = []
    weights = ensemble.get("weights", {})
    for name, md in ensemble["models"].items():
        try:
            pred = float(md["model"].predict(X_scaled)[0])
            w = weights.get(name, 1.0)
            predictions.append((pred, w, name))
        except Exception:
            continue

    if not predictions:
        return None, None

    # Weighted ensemble
    total_w = sum(w for _, w, _ in predictions)
    ensemble_pred = sum(p * w / total_w for p, w, _ in predictions) if total_w > 0 else 0

    # Confidence based on model agreement
    directions = [p > 0 for p, _, _ in predictions]
    agreement = sum(directions) / len(directions)
    agreement_conf = max(agreement, 1 - agreement) * 100

    return {
        "predicted_change_pct": round(ensemble_pred, 2),
        "confidence": round(agreement_conf, 1),
        "model_predictions": {name: round(p, 4) for p, _, name in predictions},
        "num_models": len(predictions),
        "feature_vector": X[0].tolist()
    }, X[0].tolist()


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

        # Handle feature count mismatch
        expected = ensemble["feature_count"]
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
