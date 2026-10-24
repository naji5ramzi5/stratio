"""
validate_ml.py – Honest out-of-sample accuracy of the prediction-bot ML ensemble.

Takes a symbol's history, trains the same ensemble family used by ml_trainer on
the FIRST 80% of bars, and evaluates DIRECTION ACCURACY on the LAST 20% (bars
the model never saw).  Reports accuracy vs:
  * random guess (50%)
  * "always long" base rate (fraction of up-bars) -- the honest baseline in an
    uptrending market

Run: python validate_ml.py [--symbols BTCUSDT,ETHUSDT] [--total 2600] [--n_fwd 24]
"""
import argparse
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
from settings import MAX_FEATURES, DEFAULT_SYMBOLS, RANDOM_STATE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit


def build_frame(symbol, interval, total):
    raw = fetch_klines(symbol, interval, limit=total)
    if not raw or len(raw) < 500:
        return None
    df = pd.DataFrame(raw, columns=KLINES_COLUMNS)
    for c in NUMERIC_COLS:
        df[c] = df[c].astype(float)
    df = compute_features(df)
    return merge_btc_features(df, fetch_btc_returns(interval, total, min_samples=100))


def split_symbol(symbol, interval, total, n_fwd, train_frac=0.8):
    df = build_frame(symbol, interval, total)
    if df is None:
        return None
    exclude = set(KLINES_COLUMNS + [c for c in df.columns if c.startswith("target_")])
    feats = [c for c in df.columns if c not in exclude]
    df["target"] = df["close"].shift(-n_fwd) / df["close"] - 1
    df = df.dropna().reset_index(drop=True)
    if len(df) < 400:
        return None
    X = df[feats].astype(float).values
    y = df["target"].values
    n = len(y)
    cut = int(n * train_frac)
    return X[:cut], y[:cut], X[cut:], y[cut:], feats


def accuracy(pred, actual, direction=True):
    a = np.asarray(actual)
    p = np.asarray(pred)
    if direction:
        pos = (p > 0) == (a > 0)
        return float(pos.mean())
    return float(np.mean(np.abs(p - a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--total", type=int, default=2600)
    ap.add_argument("--n_fwd", type=int, default=24)
    ap.add_argument("--train_frac", type=float, default=0.8)
    args = ap.parse_args()

    base_rate, total_acc, n_rows = 0.0, 0.0, 0
    total_wacc = 0.0
    per_model = {}

    for sym in args.symbols.split(","):
        sp = split_symbol(sym, "1h", args.total, args.n_fwd, args.train_frac)
        if sp is None:
            print(f"{sym}: not enough data, skipped")
            continue
        X_tr, y_tr, X_te, y_te, feats = sp
        print(f"{sym}: train={len(y_tr)} test={len(y_te)} rows, {len(feats)} features")

        # same feature selection the production ensemble uses
        if X_tr.shape[1] > MAX_FEATURES:
            sel = SelectKBest(score_func=mutual_info_regression,
                              k=min(MAX_FEATURES, X_tr.shape[1] - 1))
            X_tr = sel.fit_transform(X_tr, y_tr)
            X_te = sel.transform(X_te)
            print(f"   selected {X_tr.shape[1]} features")

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        br = float((y_te > 0).mean())
        base_rate += br * len(y_te)
        n_rows += len(y_te)

        models = create_models(random_state=RANDOM_STATE)
        fitted = {}
        for name, model in models.items():
            try:
                model.fit(X_tr_s, y_tr)
                fitted[name] = model
            except Exception as e:
                print(f"   {name} failed: {e}")
        models = fitted

        accs = {}
        for name, model in models.items():
            try:
                pred = model.predict(X_te_s)
                a = accuracy(pred, y_te)
                accs[name] = a
                per_model.setdefault(name, []).append(a)
            except Exception as e:
                print(f"   {name} predict failed: {e}")
        ens_pred = np.mean([m.predict(X_te_s) for m in models.values()], axis=0)
        accs["ensemble"] = accuracy(ens_pred, y_te)
        per_model.setdefault("ensemble", []).append(accs["ensemble"])

        # walk-forward DIRECTION accuracy on the train split -> production-style
        # weights (the honest way to weight models without test-set peeking)
        tscv = TimeSeriesSplit(n_splits=3)
        wacc = {}
        for name, model in models.items():
            fold_acc = []
            for tr_idx, va_idx in tscv.split(X_tr_s):
                model.fit(X_tr_s[tr_idx], y_tr[tr_idx])
                fold_acc.append(accuracy(model.predict(X_tr_s[va_idx]), y_tr[va_idx]))
            wacc[name] = np.mean(fold_acc)
        w = np.array([max(wacc[n], 0.51) - 0.5 for n in models])
        w = w / w.sum()
        wpred = np.zeros(len(y_te))
        for (name, model), wi in zip(models.items(), w):
            wpred += wi * model.predict(X_te_s)
        accs["weighted_ens"] = accuracy(wpred, y_te)
        per_model.setdefault("weighted_ens", []).append(accs["weighted_ens"])
        total_acc += accs["ensemble"] * len(y_te)
        total_wacc += accs["weighted_ens"] * len(y_te)

        print(f"   base_rate(always-long)={br:.1%}")
        for k, v in sorted(accs.items()):
            print(f"   {k:10s} direction_accuracy={v:.1%}  excess_vs_base={v - br:+.1%}")

    if n_rows == 0:
        print("no data validated")
        return 1

    br = base_rate / n_rows
    print("\n" + "=" * 60)
    print(f"AGGREGATE (rows={n_rows})")
    print(f"  always-long base rate : {br:.1%}")
    print(f"  equal-weight ensemble : {total_acc / n_rows:.1%}  excess={total_acc / n_rows - br:+.1%}")
    print(f"  weighted   ensemble   : {total_wacc / n_rows:.1%}  excess={total_wacc / n_rows - br:+.1%}")
    print(f"  excess vs 50%         : {total_acc / n_rows - 0.5:+.1%}")
    for name, vals in sorted(per_model.items()):
        mean = np.mean(vals)
        print(f"  {name:10s} acc={mean:.1%}  excess_vs_base={mean - br:+.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
