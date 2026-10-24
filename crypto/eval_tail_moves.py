"""Honest OOS evaluation: can we predict 24h tail moves (>3% / >5% / >12%)?

Time-ordered split (first 70% train / last 30% test) — never shuffled, so the
test window is genuinely unseen. All features are known at prediction time.
Reports: AUC, top-decile/quartile precision + lift, a calibration table, and
how often we could honestly say ">=70% confidence" (coverage + precision).

This measurement comes BEFORE any integration: if there is no OOS lift, we
say so instead of shipping a fake number.
"""
import json
import numpy as np

NUM_FEATURES = [
    "predicted_change_pct", "expected_vol_pct", "expected_vol_raw_pct",
    "prob_up_raw", "prob_down_raw", "atr_pct", "volume_ratio", "rsi",
    "buy_pressure_pct", "divergence_score", "ml_prediction",
    "foundation_prediction", "funding_rate_pct", "long_short_ratio",
    "open_interest_change_1h_pct", "volatility_annualized_pct",
    "regime_trend_pct", "historical_accuracy_pct", "meta_labeling_prob",
    "heuristic_confidence", "filters_passed", "exchanges_count",
]
CAT_FEATURES = [
    "market_regime", "detailed_regime", "volatility_regime", "relative_bias",
    "macd_trend", "long_short_sentiment", "onchain_signal", "market_sentiment",
    "risk", "grade",
]

with open("prediction_history.json", encoding="utf-8") as f:
    hist = json.load(f)
rows = [x for x in hist if x.get("_verified")
        and x.get("_actual_change_pct") is not None]
rows.sort(key=lambda x: x.get("_created_at") or "")
n = len(rows)
print(f"verified rows: {n}")

X = []
for x in rows:
    feats = {}
    for k in NUM_FEATURES:
        try:
            v = float(x.get(k))
            if not np.isfinite(v):
                v = np.nan
        except Exception:
            v = np.nan
        feats[k] = v
    X.append(feats)

medians = {}
for k in NUM_FEATURES:
    vals = [f[k] for f in X if np.isfinite(f[k])]
    medians[k] = float(np.median(vals)) if vals else 0.0

cat_values = {k: sorted({str(x.get(k)) for x in rows if x.get(k) is not None})
              for k in CAT_FEATURES}

def featurize(x):
    v = []
    for k in NUM_FEATURES:
        val = x.get(k)
        try:
            f = float(val)
            if not np.isfinite(f):
                f = medians[k]
        except Exception:
            f = medians[k]
        v.append(f)
    for k in CAT_FEATURES:
        s = str(x.get(k))
        for c in cat_values[k]:
            v.append(1.0 if s == c else 0.0)
    return v

Xm = np.array([featurize(x) for x in rows], dtype=float)

split = int(n * 0.7)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

for target in (3.0, 5.0, 12.0):
    y = np.array([1 if x["_actual_change_pct"] > target else 0 for x in rows])
    ytr, yte = y[:split], y[split:]
    Xtr, Xte = Xm[:split], Xm[split:]
    print(f"\n===== target: actual > +{target}% =====")
    print(f"base rate train={ytr.mean()*100:.1f}%  test={yte.mean()*100:.1f}%  "
          f"(train n={len(ytr)}, test n={len(yte)})")

    models = {}
    try:
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(Xtr), ytr)
        models["logreg"] = lr.predict_proba(sc.transform(Xte))[:, 1]
    except Exception as e:
        print("  logreg failed:", e)
    try:
        gb = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                        learning_rate=0.05, subsample=0.8,
                                        random_state=0).fit(Xtr, ytr)
        models["gbm"] = gb.predict_proba(Xte)[:, 1]
    except Exception as e:
        print("  gbm failed:", e)

    for name, p in models.items():
        try:
            auc = roc_auc_score(yte, p)
        except Exception:
            auc = float("nan")
        order = np.argsort(-p)
        line = f"  {name:7s} AUC={auc:.3f}"
        for frac, tag in ((0.10, "top10%"), (0.25, "top25%")):
            k = max(1, int(len(p) * frac))
            sel = order[:k]
            prec = yte[sel].mean()
            base = yte.mean()
            line += (f" | {tag}: n={k} prec={prec*100:.1f}% "
                     f"lift={prec/base:.2f}x")
        print(line)
        # honest >=70% claim check
        hi = p >= 0.70
        if hi.sum() > 0:
            print(f"          claimed P>=0.70: n={int(hi.sum())} "
                  f"({hi.sum()/len(p)*100:.1f}% of test) "
                  f"actual={yte[hi].mean()*100:.1f}%")
        else:
            print("          claimed P>=0.70: never (model never that sure)")
        # calibration deciles
        order2 = np.argsort(-p)
        dec = np.array_split(order2, 10)
        pairs = [(p[d].mean(), yte[d].mean(), len(d)) for d in dec if len(d)]
        cal = " ".join(f"{a*100:.0f}->{b*100:.0f}%" for a, b, _ in pairs)
        print(f"          calibration pred->actual by decile: {cal}")
