"""Honest accuracy report for the stored predictions.

Reads crypto/prediction_history.json (read-only) and prints what the bot has
actually achieved — no cherry-picking, no leakage:

  * directional hit-rate, overall and per timeframe
  * for calls that predicted a rise >= 3%: how many really rose >= 3%,
    compared with the base rate of "any verified prediction rising >= 3%"
    (this base rate is the benchmark the model must beat)
  * calibration of prob_up (does "60% up" really mean 60%?)
  * market-neutral excess return, if computed

Run:  python accuracy_report.py
"""
import json
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(HERE, "prediction_history.json")


def pct(a, b):
    return f"{(100.0 * a / b):5.1f}%" if b else "  n/a"


def main():
    if not os.path.exists(PATH):
        print("prediction_history.json not found in", HERE)
        return 1
    hist = json.load(open(PATH, encoding="utf-8"))
    print(f"records: {len(hist):,}")

    ver = [p for p in hist
           if p.get("_verified") and p.get("_actual_change_pct") is not None]
    print(f"verified: {len(ver):,}  ({len(ver)/max(len(hist),1)*100:.1f}% of records)")

    if not ver:
        print("\nNo verified predictions yet — nothing to measure. "
              "Let Bot 3 run for a few hours, then re-run this script.")
        return 0

    # ── 1. Directional hit-rate ─────────────────────────────────────────────
    correct = sum(1 for p in ver if p.get("_correct"))
    print("\n=== 1. DIRECTIONAL HIT-RATE (up/down only) ===")
    print(f"  overall: {pct(correct, len(ver))}  ({correct:,}/{len(ver):,})")
    print("  >>> 50% is a coin flip. This is the number the bot prints as WR.")

    by_tf = defaultdict(list)
    for p in ver:
        by_tf[p.get("timeframe_hours", 0)].append(bool(p.get("_correct")))
    print("  per timeframe:")
    for tf in sorted(by_tf):
        hits = sum(by_tf[tf])
        print(f"    {tf:>3}h: {pct(hits, len(by_tf[tf]))}  (n={len(by_tf[tf]):,})")

    # ── 2. The ">= 3% rise" claim ───────────────────────────────────────────
    print("\n=== 2. 'PREDICTED RISE >= 3%' — DID IT ACTUALLY RISE 3%? ===")
    calls = [p for p in ver if (p.get("predicted_change_pct") or 0) >= 3.0]
    if not calls:
        print("  no verified calls with predicted_change_pct >= 3%")
    else:
        rose = sum(1 for p in calls if p["_actual_change_pct"] > 0)
        hit3 = sum(1 for p in calls if p["_actual_change_pct"] >= 3.0)
        actuals = [p["_actual_change_pct"] for p in calls]
        base3 = sum(1 for p in ver if p["_actual_change_pct"] >= 3.0)
        print(f"  calls: {len(calls):,}")
        print(f"  any rise (>0%)      : {pct(rose, len(calls))}")
        print(f"  real rise >= 3%     : {pct(hit3, len(calls))}   <-- the honest hit-rate")
        print(f"  mean actual move    : {statistics.mean(actuals):+.2f}%")
        print(f"  median actual move  : {statistics.median(actuals):+.2f}%")
        print(f"  benchmark (ALL verified predictions rising >= 3%): {pct(base3, len(ver))}")
        edge = (hit3 / len(calls)) - (base3 / len(ver))
        print(f"  lift over benchmark : {edge*100:+.1f} points")
        print("  >>> If the lift is near zero, the '>3%' filter adds no real edge.")

    # ── 3. Calibration of prob_up ───────────────────────────────────────────
    print("\n=== 3. CALIBRATION (prob_up vs reality) ===")
    buckets = defaultdict(list)
    for p in ver:
        pu = p.get("prob_up")
        if pu is None:
            continue
        b = min(int(float(pu) * 10), 9)
        buckets[b].append(p["_actual_change_pct"] > 0)
    if not buckets:
        print("  no prob_up stored on verified records")
    else:
        print("  prob_up bucket |    n | actually rose")
        for b in sorted(buckets):
            vals = buckets[b]
            print(f"    {b/10:.1f}-{(b+1)/10:.1f}     | {len(vals):4d} | {pct(sum(vals), len(vals))}")
        print("  >>> A calibrated model has 'actually rose' close to the bucket value.")

    # ── 4. Market-neutral excess ────────────────────────────────────────────
    print("\n=== 4. MARKET-NEUTRAL EXCESS RETURN ===")
    ex = [p["_excess_pct"] for p in ver if p.get("_excess_pct") is not None]
    if len(ex) < 30:
        print(f"  not enough data yet (n={len(ex)}; needs 30+)")
    else:
        pos = sum(1 for e in ex if e > 0)
        print(f"  n={len(ex):,}  mean={statistics.mean(ex):+.3f}%  "
              f"median={statistics.median(ex):+.3f}%  positive={pct(pos, len(ex))}")
        print("  >>> This removes market beta: it is the edge that survives "
              "in a bull or bear week.")

    print("\n=== VERDICT ===")
    print("  Numbers above are the only defensible ones: they come from stored")
    print("  forecasts verified against later prices, with no look-ahead.")
    print("  Any claim of 80-91% accuracy must be reproducible here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
