"""
gate_ab_test.py - Honest A/B of the regime gate + confidence filter.

Replays the ML ensemble on real historical bars with the SAME training recipe
as production (train on the first 70% only, never touching the test bars), then
compares direction accuracy across:
   * ALL signals                 (baseline: does the ensemble have any edge?)
   * confidence >= 60            (filter A)
   * regime gate agrees          (filter B, EMA24 vs EMA96 + flat fallback)
   * confidence AND gate         (filter C)

Gate thresholds are FIXED constants (regime_gate.TREND_FRACTION,
CONDITIONAL_MIN_CONF) - they are never tuned on the test set, so the lift is a
fair estimate of what the live gate can deliver.

Run: python gate_ab_test.py [--symbols BTCUSDT,ETHUSDT] [--total 2600] [--train_frac 0.7]
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
                        KLINES_COLUMNS, NUMERIC_COLS, _calibration_bin)
from settings import MAX_FEATURES, DEFAULT_SYMBOLS, RANDOM_STATE, HORIZON_BARS
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression

from regime_gate import TREND_FRACTION, CONDITIONAL_MIN_CONF

TF_INTERVAL = {2: "5m", 6: "15m", 8: "30m", 24: "1h"}


def build_frame(symbol, interval, total):
    raw = fetch_klines(symbol, interval, limit=total)
    if not raw or len(raw) < 500:
        return None
    df = pd.DataFrame(raw, columns=KLINES_COLUMNS)
    for c in NUMERIC_COLS:
        df[c] = df[c].astype(float)
    df = compute_features(df)
    return merge_btc_features(df, fetch_btc_returns(interval, total, min_samples=100))


def regime_series(df):
    """Causal per-bar trend series: +1 UP, -1 DOWN, 0 FLAT (EMA24 vs EMA96)."""
    close = df["close"]
    ema24 = close.ewm(span=24).mean()
    ema96 = close.ewm(span=96).mean()
    rel = (ema24 - ema96) / ema96.abs()
    trend = pd.Series(0, index=df.index)
    trend[rel > TREND_FRACTION] = 1
    trend[rel < -TREND_FRACTION] = -1
    return trend


def run_tf(symbol, tf_hours, total, train_frac):
    interval = TF_INTERVAL[tf_hours]
    n_fwd = HORIZON_BARS[tf_hours]
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

    X_tr, X_te = X[:cut], X[cut:]
    y_tr, y_te = y[:cut], y[cut:]

    if X_tr.shape[1] > MAX_FEATURES:
        sel = SelectKBest(score_func=mutual_info_regression, k=min(MAX_FEATURES, X_tr.shape[1] - 1))
        sel.fit(X_tr, y_tr)  # selector fit on TRAIN only (no test peek)
        X_tr = sel.transform(X_tr)
        X_te = sel.transform(X_te)
    sc = StandardScaler()
    sc.fit(X_tr)  # scaler fit on TRAIN only
    X_tr_s = sc.transform(X_tr)
    X_te_s = sc.transform(X_te)

    models = {name: m for name, m in create_models(random_state=RANDOM_STATE).items()}
    for name, m in models.items():
        try:
            m.fit(X_tr_s, y_tr)
        except Exception as exc:
            print(f"   {symbol} {tf_hours}h: {name} failed {exc}")
            del models[name]
    if not models:
        return None

    preds = {name: m.predict(X_te_s) for name, m in models.items()}
    ens = np.mean(list(preds.values()), axis=0)

    agreement = np.zeros(len(y_te))
    for i in range(len(y_te)):
        signs = [1 if preds[k][i] > 0 else -1 for k in preds]
        pos = sum(1 for s in signs if s > 0)
        agreement[i] = max(pos, len(signs) - pos) / len(signs)

    conf = np.array([(_calibration_bin(a * 100) or 0.5) * 100 for a in agreement])

    trend = regime_series(df).values[cut:]
    hit = (ens > 0) == (y_te > 0)
    direction = np.sign(ens)

    gate_ok = np.zeros(len(y_te), dtype=bool)
    for i in range(len(y_te)):
        if direction[i] == 0:
            gate_ok[i] = False
            continue
        if trend[i] == direction[i]:
            gate_ok[i] = True
        elif trend[i] == 0 and conf[i] >= CONDITIONAL_MIN_CONF:
            gate_ok[i] = True
        else:
            gate_ok[i] = False

    conf_ok = conf >= 60.0
    both_ok = gate_ok & conf_ok
    base_rate = float((y_te > 0).mean())

    def report(mask):
        if mask.sum() < 10:
            return None
        return {
            "n": int(mask.sum()),
            "acc": float(hit[mask].mean()),
            "excess_vs_base": float(hit[mask].mean()) - base_rate,
            "mean_fwd_pct": float(y_te[mask].mean() * 100),
            "mean_fwd_long_only_pct": float(y_te[mask & (ens > 0)].mean() * 100) if (mask & (ens > 0)).any() else 0.0,
        }

    return {
        "symbol": symbol, "tf": tf_hours, "interval": interval, "n": len(y_te),
        "base_rate": base_rate,
        "all": report(np.ones(len(y_te), dtype=bool)),
        "conf_only": report(conf_ok),
        "gate_only": report(gate_ok),
        "gate_and_conf": report(both_ok),
    }


def fmt(tag, r, base_rate):
    if r is None:
        return f"  {tag:14s} n=    -   skipped (n<10)"
    return (f"  {tag:14s} n={r['n']:5d}  acc={r['acc']:5.1%}  "
            f"excess={r['excess_vs_base']:+5.1%}  mean_fwd={r['mean_fwd_pct']:+.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--total", type=int, default=2600)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--tfs", default="2,6,8,24", help="comma list of tf_hours")
    args = ap.parse_args()

    agg = {}
    for sym in args.symbols.split(","):
        for tf in [int(x) for x in args.tfs.split(",") if x.strip()]:
            r = run_tf(sym, tf, args.total, args.train_frac)
            if r is None:
                print(f"{sym} {tf}h: insufficient data")
                continue
            print(f"\n{sym} {tf}h ({r['interval']}, test n={r['n']}, base_rate={r['base_rate']:.1%}):")
            print(fmt("ALL", r["all"], r["base_rate"]))
            print(fmt("CONF>=60", r["conf_only"], r["base_rate"]))
            print(fmt("REGIME_GATE", r["gate_only"], r["base_rate"]))
            print(fmt("GATE+CONF", r["gate_and_conf"], r["base_rate"]))
            key = (sym, tf)
            agg[key] = r

    if not agg:
        print("nothing validated")
        return 1

    print("\n" + "=" * 66)
    print("AGGREGATE (weighted by test n)")
    labels = [("all", "ALL"), ("conf_only", "CONF>=60"),
              ("gate_only", "REGIME_GATE"), ("gate_and_conf", "GATE+CONF")]
    for key, label in labels:
        rows = [r[key] for r in agg.values() if r[key]]
        if not rows:
            continue
        n = sum(x["n"] for x in rows)
        acc = sum(x["acc"] * x["n"] for x in rows) / n
        excess = sum(x["excess_vs_base"] * x["n"] for x in rows) / n
        print(f"  {label:12s} n={n:6d}  acc={acc:5.1%}  excess_vs_base={excess:+5.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
