"""
calibrate_predictor.py – Train the meta-labeler + confidence calibration from
HONEST out-of-sample predictions.

Walk-forward protocol (no look-ahead):
  * per symbol: train the ensemble family on the FIRST 80% of bars,
    predict the LAST 20% (never seen).
  * for each test bar, predictor-style features are computed causally from the
    bars up to that bar only (rsi, atr, volume ratio, regime, support/resistance
    from rolling windows -- no future data).
  * y = 1 when the ensemble direction matched the realized forward return.

Outputs:
  * models/meta_labeler.pkl      -> meta-labeler (filters unreliable signals)
  * models/calibration.json      -> confidence-bin -> realized accuracy mapping

Run: python calibrate_predictor.py [--symbols BTCUSDT,ETHUSDT,SOLUSDT] [--total 4000]
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from ml_trainer import (compute_features, create_models, fetch_klines,
                        fetch_btc_returns, merge_btc_features,
                        KLINES_COLUMNS, NUMERIC_COLS)
from meta_labeling import MetaLabeler, META_MODEL_PATH
from settings import (MAX_FEATURES, HORIZON_BARS, INTERVAL_TF,
                      RANDOM_STATE, DEFAULT_SYMBOLS)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression

CALIBRATION_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "models", "calibration.json")


def _grade(pct):
    if pct >= 5.0: return "STRONG UP"
    if pct >= 3.0: return "MODERATE UP"
    if pct >= 1.5: return "SLIGHT UP"
    if pct >= -1.5: return "NEUTRAL"
    if pct >= -3.0: return "SLIGHT DOWN"
    if pct >= -5.0: return "MODERATE DOWN"
    return "STRONG DOWN"


def _regime(close, k):
    if k >= 50:
        sma20 = np.mean(close[k - 20:k])
        sma50 = np.mean(close[k - 50:k])
        if sma20 > sma50 * 1.05: return "bullish"
        if sma20 < sma50 * 0.95: return "bearish"
    return "neutral"


def build_series(symbol, interval, total, n_fwd):
    raw = fetch_klines(symbol, interval, limit=total)
    if not raw or len(raw) < 600:
        return None
    df = pd.DataFrame(raw, columns=KLINES_COLUMNS)
    for c in NUMERIC_COLS:
        df[c] = df[c].astype(float)
    df = compute_features(df)
    df = merge_btc_features(df, fetch_btc_returns(interval, total, min_samples=100))
    df["target"] = df["close"].shift(-n_fwd) / df["close"] - 1
    return df.dropna().reset_index(drop=True)


def run(symbols, total=4000, n_fwd=24, interval="1h", train_frac=0.8):
    ml = MetaLabeler()
    metaX, metaY, conf_pairs = [], [], []
    rows = 0

    for sym in symbols:
        df = build_series(sym, interval, total, n_fwd)
        if df is None:
            print(f"{sym}@{interval}: not enough data, skipped")
            continue
        exclude = set(KLINES_COLUMNS + ["target"] + [c for c in df.columns if c.startswith("target_")])
        feats = [c for c in df.columns if c not in exclude]
        X = df[feats].astype(float).values
        y = df["target"].values
        n = len(y)
        cut = int(n * train_frac)
        X_tr, y_tr, X_te, y_te = X[:cut], y[:cut], X[cut:], y[cut:]

        if X_tr.shape[1] > MAX_FEATURES:
            sel = SelectKBest(score_func=mutual_info_regression,
                              k=min(MAX_FEATURES, X_tr.shape[1] - 1))
            X_tr = sel.fit_transform(X_tr, y_tr)
            X_te = sel.transform(X_te)
        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        models = create_models(random_state=RANDOM_STATE)
        fitted = {}
        for name, model in models.items():
            try:
                model.fit(X_tr_s, y_tr)
                fitted[name] = model
            except Exception:
                continue
        models = fitted
        if not models:
            continue

        close = df["close"].values
        highs = df["high"].values
        lows = df["low"].values
        rsi = df["rsi_14"].values
        vratio = df["v_ratio_5_20"].values
        atrp = df["atr_14_pct"].values

        # vectorized causal helpers (batch predict once, per-row is cheap)
        P = np.stack([m.predict(X_te_s) for m in models.values()])  # (M, n_test)
        sign_up = (P > 0).mean(axis=0)
        sign_dn = (P < 0).mean(axis=0)
        pred = P.mean(axis=0)
        agree = np.maximum(sign_up, sign_dn) * 100.0
        # rolling max/min of last 30 bars up to and including row i (causal)
        roll_hi = pd.Series(highs).rolling(30, min_periods=5).max().to_numpy()
        roll_lo = pd.Series(lows).rolling(30, min_periods=5).min().to_numpy()
        sma20 = pd.Series(close).rolling(20).mean().to_numpy()
        sma50 = pd.Series(close).rolling(50).mean().to_numpy()

        for k in range(len(y_te)):
            predv = float(pred[k])
            actual = float(y_te[k])
            if not (np.isfinite(predv) and np.isfinite(actual)):
                continue
            correct = 1 if (predv > 0 and actual > 0) or (predv < 0 and actual < 0) else 0
            conf_pairs.append((float(agree[k]), correct))

            i = cut + k
            price = close[i]
            res = roll_hi[i] if roll_hi[i] > price * 0.98 else 0.0
            sup = roll_lo[i] if roll_lo[i] < price * 1.02 else 0.0
            if i >= 50:
                regime = ("bullish" if sma20[i] > sma50[i] * 1.05
                          else "bearish" if sma20[i] < sma50[i] * 0.95
                          else "neutral")
            else:
                regime = "neutral"
            dummy = {
                "confidence": float(agree[k]),
                "filters_passed": 2.0,
                "rsi": float(rsi[i]),
                "predicted_change_pct": predv * 100,
                "volume_ratio": float(vratio[i]),
                "atr_pct": float(atrp[i]),
                "divergence_score": 0.0,
                "buy_pressure_pct": 50.0,
                "nearest_support": float(sup),
                "nearest_resistance": float(res),
                "historical_accuracy_pct": 50.0,
                "market_regime": regime,
                "grade": _grade(predv * 100),
            }
            metaX.append(ml._build_features(dummy))
            metaY.append(correct)
            rows += 1

        print(f"{sym}@{interval}: OOS rows={len(y_te)} selected={X_tr_s.shape[1]}")

    return metaX, metaY, conf_pairs


def run_configs(symbols, configs):
    """Run several (interval, n_fwd, total) walk-forward configs and pool
    every out-of-sample prediction into one honest sample."""
    allX, allY, allConf = [], [], []
    for interval, n_fwd, total in configs:
        x, y, c = run(symbols, total=total, n_fwd=n_fwd, interval=interval)
        allX += x
        allY += y
        allConf += c
        print(f"  pooled: {len(allY)} samples")
    return allX, allY, allConf


def build():
    ml = MetaLabeler()
    rows = 0
    metaX, metaY, conf_pairs = [], [], []
    x, y, c = run_configs(list(DEFAULT_SYMBOLS), [
        ("1h", HORIZON_BARS[INTERVAL_TF["1h"]], 8000),
        ("15m", HORIZON_BARS[INTERVAL_TF["15m"]], 6000),
    ])
    metaX, metaY, conf_pairs = x, y, c
    rows = len(metaY)

    if rows < 50:
        print(f"too few samples ({rows}); nothing saved")
        return False

    # (ب) meta-labeler
    Xm = np.array(metaX)
    ym = np.array(metaY)
    print(f"\nmeta-labeler samples={len(ym)} hit_rate={ym.mean():.1%}")
    if ml.train(X=Xm, y=ym):
        ml.save()
    else:
        print("meta-labeler training failed")

    # (ج) confidence calibration -- realized accuracy per confidence bin
    bins = [(50, 60), (60, 70), (70, 80), (80, 90), (90, 100)]
    cal = {"overall": round(float(np.mean([ok for _, ok in conf_pairs])), 3),
           "n": rows}
    for lo, hi in bins:
        sel = [ok for c, ok in conf_pairs if lo <= c < hi]
        if len(sel) >= 20:
            cal[f"{lo}-{hi}"] = round(float(np.mean(sel)), 3)
    for lo, hi in bins:
        csel = [ok for c, ok in conf_pairs if lo <= c < hi]
        acc = cal.get(f"{lo}-{hi}")
        print(f"  conf[{lo:02d}-{hi:02d}) n={len(csel):4d} realized_acc={acc if acc else 'n/a'}")
    try:
        os.makedirs(os.path.dirname(CALIBRATION_PATH), exist_ok=True)
        with open(CALIBRATION_PATH, "w") as f:
            json.dump(cal, f, indent=2)
        print(f"calibration saved: {CALIBRATION_PATH}")
    except Exception as e:
        print(f"calibration save failed: {e}")
    return True


if __name__ == "__main__":
    ok = build()
    sys.exit(0 if ok else 1)
