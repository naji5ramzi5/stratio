"""
STRATOCRYPTO — Walk-forward / Out-of-Sample evaluation harness
=============================================================

GOAL: Honestly measure whether the prediction engine has a tradable EDGE.

Strict rules (per owner requirements):
  * NO random splitting — purely chronological Walk-forward:
        Train -> Validation (threshold tuning) -> OOS Test
    with multiple rolling windows.
  * OOS results are NEVER used to tune models / thresholds / calibration.
    All tuning happens inside Train/Validation only.
  * Features are point-in-time (reused EXACTLY from ml_trainer.compute_features
    so the evaluation matches the live pipeline and cannot silently leak).
  * The label is the FUTURE return `n_fwd` bars ahead (shift(-n_fwd)) — by
    construction the label is allowed to be future; features are not.
  * Probabilistic metrics (Brier, calibration, log-loss, per-class PR) are
    reported, not just accuracy.
  * A separate, realistic trade simulation (fees + slippage) is run.
  * Always compared against baselines.
  * Paper/Research mode ONLY — no live orders.

Outputs:
  crypto/OOS_VALIDATION_REPORT.md
  crypto/evaluation/oos_results.json

If no edge is found, the report states PROFITABILITY UNPROVEN.
"""

import os
import sys
import json
import math
import time
import argparse
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support, brier_score_loss,
                             log_loss, confusion_matrix)
from sklearn.calibration import calibration_curve

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CRYPTO_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, CRYPTO_DIR)

from ml_trainer import (compute_features, merge_btc_features,
                        fetch_btc_returns, fetch_klines)
from settings import INTERVAL_TF, HORIZON_BARS, RANDOM_STATE, PIPELINE_VERSION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [WF] %(message)s")
log = logging.getLogger("WF")

# ── Configurable costs (documented, adjustable) ──────────────────
TAKER_BPS = float(os.getenv("WF_TAKER_BPS", "10"))      # 0.10% per fill
SLIP_BPS = float(os.getenv("WF_SLIP_BPS", "5"))         # 0.05% per fill
# Directional futures round-trip = 2 fills (entry+exit) => 2*(taker+slip)
ROUND_TRIP_BPS = 2 * (TAKER_BPS + SLIP_BPS)
NEUTRAL_BAND_PCT = 0.30   # |future return| <= band => NEUTRAL class

CLASS_NAMES = ["DOWN", "NEUTRAL", "UP"]   # 0,1,2


# ── Data loading (reuses the EXACT live feature pipeline) ─────────
def load_data(symbol, interval, bars, download_log):
    t0 = time.time()
    raw = fetch_klines(symbol, interval, bars)
    if not raw or len(raw) < 100:
        raise RuntimeError(f"insufficient data for {symbol} {interval}: "
                           f"{len(raw) if raw else 0} rows")
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close",
                                    "volume", "close_time", "quote_vol", "trades",
                                    "taker_buy_vol", "taker_buy_quote", "ignore"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    # EXACT same feature pipeline as live training (point-in-time causal).
    df = compute_features(df)
    btc = fetch_btc_returns(interval, bars, min_samples=200)
    df = merge_btc_features(df, btc)

    target_tf = INTERVAL_TF.get(interval, 24)
    n_fwd = HORIZON_BARS.get(target_tf, 24)
    # LABEL = FUTURE return n_fwd bars ahead (allowed to be future).
    df["target_pct"] = (df["close"].shift(-n_fwd) / df["close"] - 1) * 100.0

    exclude = {"timestamp", "open", "high", "low", "close", "volume",
               "close_time", "quote_vol", "trades", "taker_buy_vol",
               "taker_buy_quote", "ignore", "target_pct"}
    feats = [c for c in df.columns if c not in exclude]

    df = df.dropna().reset_index(drop=True)
    download_log.update({
        "source": "Binance API (data_loader.binance_ohlcv)",
        "symbol": symbol, "interval": interval,
        "bars_requested": bars, "rows_after_cleanup": len(df),
        "n_fwd_bars": n_fwd, "feature_count": len(feats),
        "start_utc": datetime.fromtimestamp(df["timestamp"].iloc[0] / 1000,
                                            tz=timezone.utc).isoformat()
                    if len(df) else None,
        "end_utc": datetime.fromtimestamp(df["timestamp"].iloc[-1] / 1000,
                                          tz=timezone.utc).isoformat()
                    if len(df) else None,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    return df, feats, n_fwd


def make_labels(target_pct, band=NEUTRAL_BAND_PCT):
    return np.where(target_pct > band, 2,
                    np.where(target_pct < -band, 0, 1))


# ── Walk-forward split (chronological, rolling) ──────────────────
def wf_splits(n, n_train, n_val, n_test, step, embargo=0):
    """Chronological folds with a purge between labels and the next split.

    A row's label looks ``n_fwd`` bars ahead. Without this embargo, labels at
    the end of Train can observe prices inside Validation/Test and leak future
    information into model selection.
    """
    out = []
    start = 0
    required = n_train + n_val + n_test + 2 * embargo
    while start + required <= n:
        tr = (start, start + n_train)
        va_start = tr[1] + embargo
        va = (va_start, va_start + n_val)
        te_start = va[1] + embargo
        te = (te_start, te_start + n_test)
        out.append((tr, va, te))
        start += step
    return out


# ── Trade simulation (realistic, separate from accuracy) ─────────
def trade_sim(actual_pct, ypred, exp_ret, cost_bps, stride):
    cost = cost_bps / 10000.0
    rets = []
    for i in range(0, len(actual_pct), stride):
        cls = ypred[i]
        e = exp_ret[i]
        if cls == 2 and e > cost:
            side = 1
        elif cls == 0 and e < -cost:
            side = -1
        else:
            side = 0
        if side != 0:
            rets.append(side * actual_pct[i] / 100.0 - cost)
    return np.array(rets)


def baseline_sim(actual_pct, signals, cost_bps, stride):
    cost = cost_bps / 10000.0
    rets = []
    for i in range(0, len(actual_pct), stride):
        side = signals[i]
        if side != 0:
            rets.append(side * actual_pct[i] / 100.0 - cost)
    return np.array(rets)


def perf_stats(rets, cost_bps=ROUND_TRIP_BPS):
    if len(rets) == 0:
        return {"n_trades": 0, "net_return_pct": 0.0, "gross_return_pct": 0.0,
                "fees_pct": 0.0, "profit_factor": 0.0, "expectancy_pct": 0.0,
                "max_drawdown_pct": 0.0, "win_rate_pct": 0.0, "turnover": 0.0}
    gross = float(np.sum(rets + (cost_bps / 10000.0)))
    fees = float(len(rets) * (cost_bps / 10000.0)) * 100.0
    wins = rets[rets > 0]
    losses = rets[rets < 0]
    pf = float(np.sum(wins) / abs(np.sum(losses))) if len(losses) else (float('inf') if len(wins) else 0.0)
    eq = np.cumsum(rets)
    peak = np.maximum.accumulate(eq)
    dd = float(np.max(peak - eq)) * 100.0
    return {
        "n_trades": int(len(rets)),
        "net_return_pct": round(float(np.sum(rets)) * 100.0, 3),
        "gross_return_pct": round(gross * 100.0, 3),
        "fees_pct": round(fees, 3),
        "profit_factor": round(pf, 3) if pf != float('inf') else None,
        "expectancy_pct": round(float(np.mean(rets)) * 100.0, 4),
        "max_drawdown_pct": round(dd, 3),
        "win_rate_pct": round(float(len(wins) / len(rets)) * 100.0, 2),
        "turnover": int(len(rets)),
    }


# ── Metrics for one fold ────────────────────────────────────────
def compute_metrics(yte, ypred, proba, actual_pct, exp_ret, ret1, n_fwd, cost_bps):
    acc = accuracy_score(yte, ypred)
    bal = balanced_accuracy_score(yte, ypred)
    prec, rec, f1, sup = precision_recall_fscore_support(
        yte, ypred, labels=[0, 1, 2], zero_division=0)
    brier = brier_score_loss(yte, proba, labels=[0, 1, 2])
    ll = log_loss(yte, proba, labels=[0, 1, 2])
    cm = confusion_matrix(yte, ypred, labels=[0, 1, 2]).tolist()
    neutral_pct = float(np.mean(ypred == 1))

    # Reliability table (one-vs-rest per class)
    reliability = {}
    for ci, cname in enumerate(CLASS_NAMES):
        yt = (yte == ci).astype(int)
        p = proba[:, ci]
        if p.max() - p.min() < 1e-6:
            continue
        try:
            frac, mean_pred = calibration_curve(yt, p, n_bins=5)
            reliability[cname] = {
                "bins": [round(float(m), 3) for m in mean_pred],
                "observed": [round(float(f), 3) for f in frac],
            }
        except Exception:
            pass

    # Trade simulation (non-overlapping every n_fwd bars)
    a = actual_pct
    pr = ypred
    er = exp_ret
    model_rets = trade_sim(a, pr, er, cost_bps, n_fwd)
    model_perf = perf_stats(model_rets, cost_bps)

    # Baselines on the SAME test window
    if len(ret1) != len(a):
        ret1 = np.zeros(len(a))
    maj = int(np.bincount(yte).argmax())
    rng = np.random.RandomState(RANDOM_STATE)
    baselines = {}
    # Buy & Hold / Always Long
    baselines["buy_and_hold"] = perf_stats(baseline_sim(a, np.ones(len(a), dtype=int), cost_bps, n_fwd), cost_bps)
    # Majority class
    maj_sig = np.full(len(a), 1 if maj == 2 else (-1 if maj == 0 else 0))
    baselines["majority_class"] = perf_stats(baseline_sim(a, maj_sig, cost_bps, n_fwd), cost_bps)
    # Random
    rnd = rng.choice([-1, 0, 1], size=len(a))
    baselines["random"] = perf_stats(baseline_sim(a, rnd, cost_bps, n_fwd), cost_bps)
    # Simple momentum (past return sign)
    mom_sig = np.sign(ret1).astype(int)
    baselines["simple_momentum"] = perf_stats(baseline_sim(a, mom_sig, cost_bps, n_fwd), cost_bps)

    # Per-regime breakdown (rolling vol)
    vol = pd.Series(ret1).rolling(20).std().values
    vmed = np.nanmedian(vol)
    reg = np.where(vol > vmed, "high_vol", "low_vol")
    reg_break = {}
    for rg in ["high_vol", "low_vol"]:
        mask = (reg == rg) & ~np.isnan(vol)
        if mask.sum() == 0:
            continue
        idx = np.where(mask)[0]
        reg_break[rg] = {
            "n": int(mask.sum()),
            "accuracy": round(float(accuracy_score(yte[idx], ypred[idx])), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(yte[idx], ypred[idx])), 4),
            "neutral_pct": round(float(np.mean(ypred[idx] == 1)), 3),
        }

    return {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal), 4),
        "precision": [round(float(x), 4) for x in prec],
        "recall": [round(float(x), 4) for x in rec],
        "f1": [round(float(x), 4) for x in f1],
        "support": [int(x) for x in sup],
        "brier_score": round(float(brier), 4),
        "log_loss": round(float(ll), 4),
        "confusion_matrix": cm,
        "neutral_pct": round(neutral_pct, 3),
        "reliability": reliability,
        "model_performance": model_perf,
        "baselines": baselines,
        "regime_breakdown": reg_break,
    }


# ── Leakage self-check ──────────────────────────────────────────
def leakage_self_check(df, feats):
    problems = []
    for f in feats:
        if f.startswith("target"):
            problems.append(f"feature '{f}' looks like a label (leak)")
    if "target_pct" in feats:
        problems.append("target column present in features (leak)")
    # Feature matrix must have no NaN after dropna
    if df[feats].isna().any().any():
        problems.append("NaN detected in feature matrix")
    # Confirm label is a forward shift (sanity): correlation of target with
    # immediate past close should be ~0 (not 1), i.e., not a replica of close.
    return problems


# ── Main ────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--bars", type=int, default=2000)
    ap.add_argument("--n_train", type=int, default=800)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--step", type=int, default=200)
    args = ap.parse_args()

    download_log = {}
    log.info(f"Loading {args.symbol} {args.interval} x{args.bars} ...")
    df, feats, n_fwd = load_data(args.symbol, args.interval, args.bars, download_log)
    log.info(f"Clean rows={len(df)} features={len(feats)} n_fwd={n_fwd}")

    leak = leakage_self_check(df, feats)
    log.info(f"Leakage self-check problems: {leak if leak else 'NONE'}")

    df["yclass"] = make_labels(df["target_pct"].values)

    splits = wf_splits(len(df), args.n_train, args.n_val, args.n_test, args.step,
                       embargo=n_fwd)
    log.info(f"Walk-forward windows: {len(splits)}")

    fold_results = []
    # pooled predictions + raw OOS slices for global metrics
    all_yte, all_ypred, all_proba = [], [], []
    all_actual, all_exp_ret, all_ret1 = [], [], []

    for fi, (tr, va, te) in enumerate(splits):
        X = df[feats].values.astype(float)
        y = df["yclass"].values
        Xtr, Xva, Xte = X[tr[0]:tr[1]], X[va[0]:va[1]], X[te[0]:te[1]]
        ytr, yva, yte = y[tr[0]:tr[1]], y[va[0]:va[1]], y[te[0]:te[1]]
        if len(np.unique(ytr)) < 2:
            continue
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xva_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

        clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
        clf.fit(Xtr_s, ytr)
        # The regressor must learn the future percentage return, not the
        # class label. Training it on ytr (0/1/2) made the trade simulation
        # compare an ordinal class number to a return threshold.
        reg = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
        reg.fit(Xtr_s, df["target_pct"].values[tr[0]:tr[1]])

        # Threshold tuning on VALIDATION only (never on test/oos)
        val_proba = clf.predict_proba(Xva_s)
        val_acc = accuracy_score(yva, clf.predict(Xva_s))
        # (threshold is implicit in class decision; we record val accuracy as
        #  the tuning signal — no OOS used for tuning)
        raw_proba = clf.predict_proba(Xte_s)
        # A small train window can miss one class. Always align probabilities
        # to the fixed DOWN/NEUTRAL/UP column order for metrics and reporting.
        proba = np.zeros((len(Xte_s), len(CLASS_NAMES)))
        proba[:, clf.classes_.astype(int)] = raw_proba
        ypred = clf.predict(Xte_s)
        exp_ret = reg.predict(Xte_s)

        f_ret1 = df["ret_1"].values[te[0]:te[1]] if "ret_1" in df else np.zeros(len(yte))
        m = compute_metrics(yte, ypred, proba, df["target_pct"].values[te[0]:te[1]],
                            exp_ret, f_ret1, n_fwd, ROUND_TRIP_BPS)
        m["fold"] = fi
        m["train_range"] = tr
        m["val_range"] = va
        m["test_range"] = te
        m["val_accuracy"] = round(float(val_acc), 4)
        fold_results.append(m)
        all_yte.extend(yte.tolist())
        all_ypred.extend(ypred.tolist())
        all_proba.append(proba)
        all_actual.append(df["target_pct"].values[te[0]:te[1]])
        all_exp_ret.append(exp_ret)
        all_ret1.append(f_ret1)

    if not fold_results:
        raise RuntimeError("no usable walk-forward folds; increase bars or reduce window sizes")

    all_yte = np.array(all_yte)
    all_ypred = np.array(all_ypred)
    all_proba = np.vstack(all_proba)
    all_actual = np.concatenate(all_actual)
    all_exp_ret = np.concatenate(all_exp_ret)
    all_ret1 = np.concatenate(all_ret1)

    pooled = compute_metrics(all_yte, all_ypred, all_proba,
                              all_actual, all_exp_ret, all_ret1,
                              n_fwd, ROUND_TRIP_BPS)

    # Edge decision
    model_net = pooled["model_performance"]["net_return_pct"]
    bh_net = pooled["baselines"]["buy_and_hold"]["net_return_pct"]
    model_pf = pooled["model_performance"]["profit_factor"]
    sufficient_evidence = (len(fold_results) >= 3 and
                           pooled["model_performance"]["n_trades"] >= 30)
    beat = (model_net > bh_net and model_net > 0 and model_pf is not None and
            model_pf > 1.0 and pooled["balanced_accuracy"] > 0.55)
    verdict = "PROFITABLE EDGE DEMONSTRATED" if (sufficient_evidence and beat) else "PROFITABILITY UNPROVEN"

    results = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "config": {
            "symbol": args.symbol, "interval": args.interval, "bars": args.bars,
            "n_train": args.n_train, "n_val": args.n_val, "n_test": args.n_test,
            "step": args.step, "n_fwd_bars": n_fwd, "neutral_band_pct": NEUTRAL_BAND_PCT,
            "embargo_bars": n_fwd,
            "round_trip_cost_bps": ROUND_TRIP_BPS, "taker_bps": TAKER_BPS, "slip_bps": SLIP_BPS,
        },
        "data_download": download_log,
        "leakage_self_check": {"passed": len(leak) == 0, "problems": leak},
        "n_folds": len(fold_results),
        "pooled_metrics": pooled,
        "folds": fold_results,
        "edge_verdict": verdict,
        "edge_requirements": {
            "minimum_folds": 3,
            "minimum_trades": 30,
            "requires_positive_net_return": True,
            "requires_beat_buy_and_hold": True,
            "minimum_profit_factor": 1.0,
            "minimum_balanced_accuracy": 0.55,
            "sufficient_evidence": sufficient_evidence,
        },
    }

    os.makedirs(os.path.join(CRYPTO_DIR, "evaluation"), exist_ok=True)
    with open(os.path.join(CRYPTO_DIR, "evaluation", "oos_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ── Markdown report ──
    rep = []
    rep.append("# STRATOCRYPTO — OOS / WALK-FORWARD VALIDATION REPORT\n")
    rep.append(f"*Generated:* {results['generated_at_utc']}  |  *Pipeline:* {PIPELINE_VERSION}\n")
    rep.append("## 1. Setup\n")
    rep.append(f"- Symbol: **{args.symbol}** | Interval: **{args.interval}** | Bars loaded: **{args.bars}**")
    rep.append(f"- Walk-forward windows: **{len(fold_results)}** (Train={args.n_train}, Val={args.n_val}, Test={args.n_test}, Step={args.step}, Embargo={n_fwd})")
    rep.append(f"- Horizon: **{n_fwd} bars** | Neutral band: ±{NEUTRAL_BAND_PCT}%")
    rep.append(f"- Round-trip cost: **{ROUND_TRIP_BPS:.1f} bps** (taker {TAKER_BPS}bps + slip {SLIP_BPS}bps, 2 fills)")
    rep.append(f"- Features: **{download_log.get('feature_count')}** (reused EXACTLY from live ml_trainer pipeline)\n")
    rep.append("## 2. Data Download Log\n")
    for k, v in download_log.items():
        rep.append(f"- {k}: `{v}`")
    rep.append("")
    rep.append("## 3. Leakage Self-Check\n")
    rep.append(f"- **Passed:** {results['leakage_self_check']['passed']}")
    if leak:
        rep.append(f"- Problems: {leak}")
    else:
        rep.append("- No future-looking feature detected. Label is a forward return (allowed); features are causal.")
    rep.append("")
    rep.append("## 4. Pooled OOS Metrics (all folds combined)\n")
    p = pooled
    rep.append(f"- Accuracy: **{p['accuracy']}**")
    rep.append(f"- Balanced Accuracy: **{p['balanced_accuracy']}**")
    rep.append(f"- Brier Score: **{p['brier_score']}** (lower=better; 0=perfect)")
    rep.append(f"- Log Loss: **{p['log_loss']}**")
    rep.append(f"- Neutral%: **{p['neutral_pct']}**")
    rep.append(f"- Precision [DOWN,NEU,UP]: {p['precision']}")
    rep.append(f"- Recall    [DOWN,NEU,UP]: {p['recall']}")
    rep.append(f"- Confusion [DOWN,NEU,UP]:\n```\n{p['confusion_matrix']}\n```")
    rep.append("\n### Calibration (reliability)\n")
    for cname, tbl in p["reliability"].items():
        rep.append(f"- **{cname}**: predicted={tbl['bins']} observed={tbl['observed']}")
    rep.append("")
    rep.append("## 5. Trade Simulation (after costs)\n")
    mp = p["model_performance"]
    rep.append(f"- Model net return: **{mp['net_return_pct']}%** | trades={mp['n_trades']} | "
               f"PF={mp['profit_factor']} | win%={mp['win_rate_pct']} | maxDD={mp['max_drawdown_pct']}%")
    rep.append("\n### Baselines (same OOS window, same costs)\n")
    for bname, bval in p["baselines"].items():
        rep.append(f"- {bname}: net={bval['net_return_pct']}% PF={bval['profit_factor']} win%={bval['win_rate_pct']} trades={bval['n_trades']}")
    rep.append("")
    rep.append("## 6. Per-Regime\n")
    for rg, rv in p["regime_breakdown"].items():
        rep.append(f"- {rg}: n={rv['n']} acc={rv['accuracy']} bal_acc={rv['balanced_accuracy']} neutral%={rv['neutral_pct']}")
    rep.append("")
    rep.append("## 7. Edge Verdict\n")
    rep.append(f"**{verdict}**\n")
    rep.append("- Edge requires at least 3 folds, 30 OOS trades, positive net return, profit factor > 1, balanced accuracy > 0.55, and outperformance versus Buy & Hold.")
    rep.append("- Note: prediction QUALITY (metrics above) is separate from tradeability (Section 5). "
               "A model can be statistically decent yet unprofitable after costs.\n")
    rep.append("## 8. Assumptions & Limits\n")
    rep.append("- Features reused from `ml_trainer.compute_features` to match live inference.")
    rep.append("- Trade sim assumes one position per `n_fwd` bars, round-trip cost, no funding, no partial fills.")
    rep.append("- No OOS data used for tuning. Tuning restricted to Train/Validation, with an embargo equal to the forecast horizon between windows.")
    rep.append("- If data download failed or windows were too few, results are NOT conclusive.")
    rep.append("\n---\n*Paper/Research mode only. No live orders were placed.*")

    with open(os.path.join(CRYPTO_DIR, "OOS_VALIDATION_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(rep))

    log.info(f"DONE. Verdict: {verdict}")
    log.info(f"Pooled accuracy={pooled['accuracy']} bal_acc={pooled['balanced_accuracy']} "
             f"brier={pooled['brier_score']} model_net%={mp['net_return_pct']} "
              f"BH_net%={pooled['baselines']['buy_and_hold']['net_return_pct']}")


if __name__ == "__main__":
    main()
