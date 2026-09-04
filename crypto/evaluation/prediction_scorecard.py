"""
prediction_scorecard.py – Honest, time-resolved evaluation of past predictions.

Source of truth: ``prediction_accuracy.json`` (prediction_tracker), which is
now actually verified by the live loop. It also reads the legacy
``prediction_history.json`` (advanced_bot's own verifier) so historical data
is not wasted.

Outputs:
  * a readable console report: overall / per-symbol / per-timeframe accuracy,
    the realized-accuracy-per-confidence-bin calibration table, and a recent
    window vs. lifetime comparison (drift check);
  * a JSON report written next to the state files (evaluation/report.json).

Run:  python evaluation/prediction_scorecard.py
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.state_store import load_json, save_json, safe_float

BASE_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, "..")))
TRACKER_FILE = os.path.join(BASE_DIR, "prediction_accuracy.json")
HISTORY_FILE = os.path.join(BASE_DIR, "prediction_history.json")
REPORT_FILE = os.path.join(BASE_DIR, "evaluation", "report.json")

CONF_BINS = [(50, 60), (60, 70), (70, 80), (80, 90), (90, 101)]


def _is_correct(rec, sym_key="symbol"):
    p = safe_float(rec.get("predicted_change_pct"))
    a = rec.get("actual_change_pct")
    if a is None and rec.get("_actual_change_pct") is not None:
        a = rec["_actual_change_pct"]
    a = safe_float(a)
    if p == 0 or a == 0:
        return None
    return 1 if (p > 0 and a > 0) or (p < 0 and a < 0) else 0


def _verified(records):
    """Filter to records that carry a usable outcome.

    Tracker records use ``actual_change_pct``; legacy history records use
    ``_actual_change_pct`` together with ``_verified``.
    """
    out = []
    for r in records:
        a = r.get("actual_change_pct")
        if a is None:
            a = r.get("_actual_change_pct")
        if a is None:
            continue
        if r.get("actual_change_pct") is None and r.get("_verified") is not True:
            continue
        out.append(r)
    return out


def _bucket(conf):
    c = safe_float(conf)
    for lo, hi in CONF_BINS:
        if lo <= c < hi:
            return f"{lo}-{hi if hi <= 100 else 100}"
    return None


def analyze(records, label):
    recs = _verified(records)
    n = len(recs)
    if n == 0:
        return {"source": label, "n": 0}

    correct = [c for c in (_is_correct(r) for r in recs) if c is not None]
    acc = sum(correct) / len(correct) if correct else None

    # per symbol / per timeframe
    per_symbol = defaultdict(list)
    per_tf = defaultdict(list)
    for r, c in zip(recs, correct):
        per_symbol[r.get("symbol", "?")].append(c)
        per_tf[r.get("timeframe_hours", "?")].append(c)

    by_symbol = {s: {"n": len(cs), "acc": round(sum(cs) / len(cs), 3)}
                 for s, cs in per_symbol.items() if len(cs) >= 5}
    by_tf = {str(t): {"n": len(cs), "acc": round(sum(cs) / len(cs), 3)}
             for t, cs in per_tf.items() if len(cs) >= 5}

    # calibration: realized accuracy per confidence bin
    cal = {}
    for lo, hi in CONF_BINS:
        key = f"{lo}-{hi if hi <= 100 else 100}"
        cs = [c for r, c in zip(recs, correct)
              if _bucket(r.get("confidence")) == key]
        if len(cs) >= 10:
            cal[key] = {"n": len(cs), "realized_acc": round(sum(cs) / len(cs), 3)}

    # drift: recent (last 30% by timestamp) vs lifetime
    dated = [(r.get("timestamp") or r.get("_created_at") or "", c)
             for r, c in zip(recs, correct)]
    dated = [d for d in dated if d[0]]
    recent = dated[-max(1, len(dated) // 3):] if dated else []
    drift = None
    if len(recent) >= 10:
        drift = {
            "recent_n": len(recent),
            "recent_acc": round(sum(c for _, c in recent) / len(recent), 3),
            "lifetime_acc": round(acc, 3) if acc is not None else None,
            "delta": round(sum(c for _, c in recent) / len(recent) - acc, 3) if acc else None,
        }

    mae = None
    errs = [abs(safe_float(r.get("predicted_change_pct"))
               - safe_float(r.get("actual_change_pct") or r.get("_actual_change_pct")))
            for r in recs]
    if errs:
        mae = round(sum(errs) / len(errs), 3)

    return {
        "source": label,
        "n": n,
        "direction_accuracy": round(acc, 3) if acc is not None else None,
        "mae_pct": mae,
        "by_symbol": by_symbol,
        "by_timeframe": by_tf,
        "calibration": cal,
        "drift": drift,
    }


def _fmt_acc(v):
    return f"{v*100:.1f}%" if v is not None else "n/a"


def print_report(reports):
    print("=" * 62)
    print("PREDICTION SCORECARD")
    print("=" * 62)
    for rep in reports:
        if rep["n"] == 0:
            print(f"\n[{rep['source']}] no verified predictions yet")
            continue
        print(f"\n[{rep['source']}]  n={rep['n']}")
        print(f"  direction accuracy : {_fmt_acc(rep['direction_accuracy'])}   "
              f"(coin flip = 50.0%)")
        print(f"  MAE (|pred-actual|): {rep['mae_pct']} pct points")
        if rep.get("by_symbol"):
            print("  by symbol:")
            for s in sorted(rep["by_symbol"], key=lambda x: -rep["by_symbol"][x]["acc"]):
                d = rep["by_symbol"][s]
                print(f"    {s:12s} n={d['n']:4d} acc={_fmt_acc(d['acc'])}")
        if rep.get("by_timeframe"):
            print("  by timeframe (h):")
            for t in sorted(rep["by_timeframe"]):
                d = rep["by_timeframe"][t]
                print(f"    {t:>4s}      n={d['n']:4d} acc={_fmt_acc(d['acc'])}")
        if rep.get("calibration"):
            print("  calibration (realized acc per confidence bin):")
            for k in sorted(rep["calibration"]):
                d = rep["calibration"][k]
                print(f"    conf[{k}) n={d['n']:4d} realized={_fmt_acc(d['realized_acc'])}")
        if rep.get("drift"):
            d = rep["drift"]
            print(f"  drift: recent({d['recent_n']})={_fmt_acc(d['recent_acc'])} "
                  f"vs lifetime={_fmt_acc(d['lifetime_acc'])} "
                  f"delta={d['delta']:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", default=TRACKER_FILE)
    ap.add_argument("--history", default=HISTORY_FILE)
    args = ap.parse_args()

    reports = []
    tracker = load_json(args.tracker, {})
    reports.append(analyze(tracker.get("predictions", []), "prediction_accuracy.json (tracker)"))

    history = load_json(args.history, [])
    reports.append(analyze(history if isinstance(history, list) else [], "prediction_history.json (legacy)"))

    print_report(reports)

    summary = {"generated_at": __import__("datetime").datetime.now().isoformat(),
               "reports": reports}
    try:
        save_json(REPORT_FILE, summary)
        print(f"\nreport saved: {REPORT_FILE}")
    except Exception as exc:
        print(f"\nreport save failed: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
