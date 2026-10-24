"""Regime-neutral skill test.

Directional accuracy is contaminated by market beta: in a bull week every
'UP' call scores, in a bear week every 'DOWN' call scores. The honest test
of skill is CROSS-SECTIONAL: within the same timestamp, did the model rank
the coins correctly?  That metric is regime-independent by construction.

Two tests:
  1. Bucket-relative: does 'UP' beat the median coin at the same timestamp?
  2. Within-timestamp rank correlation between predicted and realised change.
"""
import json
import math
import os
import statistics
from collections import defaultdict

HIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_history.json")


def load():
    with open(HIST, encoding="utf-8") as f:
        rows = json.load(f)
    rows = [p for p in rows if p.get("_verified") and p.get("_actual_change_pct") is not None]
    return rows


def main():
    rows = load()
    # group by scan timestamp (minute granularity)
    by_ts = defaultdict(list)
    for p in rows:
        ts = (p.get("_created_at") or p.get("created_at") or "")[:16]
        by_ts[ts].append(p)
    groups = {k: v for k, v in by_ts.items() if len(v) >= 5}
    print("rows=%d  groups(min,>=5 coins)=%d" % (len(rows), len(groups)))

    # ---------- 1) cross-sectional: UP vs same-timestamp median ----------
    print()
    print("=" * 92)
    print("TEST 1: does predicted UP beat the median coin at the SAME timestamp?")
    print("(regime-neutral: the market move cancels out)")
    print("=" * 92)
    beats, total, loses = 0, 0, 0
    dev_beats, dev_total = 0, 0
    tst_beats, tst_total = 0, 0
    ts_sorted = sorted(groups)
    half = ts_sorted[len(ts_sorted) // 2]
    diffs_up, diffs_dn = [], []
    for ts, g in groups.items():
        med = statistics.median(p["_actual_change_pct"] for p in g)
        for p in g:
            rel = p["_actual_change_pct"] - med          # vs median, not vs 0
            if p["predicted_change_pct"] > 0:
                total += 1
                if rel > 0:
                    beats += 1
                elif rel < 0:
                    loses += 1
                diffs_up.append(rel)
                if ts < half:
                    dev_beats += 1
                else:
                    tst_beats += 1
                if ts < half:
                    dev_total += 1
                else:
                    tst_total += 1
            else:
                diffs_dn.append(rel)
    n_up = beats + (total - beats - loses)
    print("predicted UP  vs same-ts median: beat=%d lose=%d flat=%d -> win=%.1f%% (n=%d)"
          % (beats, loses, total - beats - loses, 100.0 * beats / max(beats + loses, 1), total))
    print("  DEV half: %.1f%% (n=%d) | TEST half: %.1f%% (n=%d)"
          % (100.0 * dev_beats / max(dev_total, 1), dev_total,
             100.0 * tst_beats / max(tst_total, 1), tst_total))
    print("  mean excess of UP calls: %+.3f%%  | DOWN calls: %+.3f%%"
          % (statistics.mean(diffs_up) if diffs_up else 0,
             statistics.mean(diffs_dn) if diffs_dn else 0))
    # naive baseline: always pick the highest-volume coin etc -> use random = 50%
    print("  (random coin picker = 50%)")

    # ---------- 2) rank correlation per timestamp ----------
    print()
    print("=" * 92)
    print("TEST 2: within-timestamp Spearman rank corr(predicted, realised)")
    print("=" * 92)

    def spearman(x, y):
        n = len(x)
        if n < 5:
            return None
        def ranks(v):
            order = sorted(range(n), key=lambda i: v[i])
            r = [0.0] * n
            i = 0
            while i < n:
                j = i
                while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                    j += 1
                avg = (i + j) / 2.0 + 1
                for k in range(i, j + 1):
                    r[order[k]] = avg
                i = j + 1
            return r
        rx, ry = ranks(x), ranks(y)
        mx, my = statistics.mean(rx), statistics.mean(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
        return num / den if den else None

    dev_r, tst_r = [], []
    for ts, g in groups.items():
        r = spearman([p["predicted_change_pct"] for p in g],
                     [p["_actual_change_pct"] for p in g])
        if r is not None:
            (dev_r if ts < half else tst_r).append(r)

    def summ(vals, label):
        if not vals:
            print("%s: no groups" % label)
            return
        m = statistics.mean(vals)
        se = statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0
        print("%s: mean rho=%+.4f  +- %.4f  (groups=%d)  t=%+.1f  %s"
              % (label, m, se, len(vals), m / se if se else 0,
                 "SIGNIFICANT" if abs(m / se) > 2 and m > 0 else ("sig NEG" if abs(m / se) > 2 else "noise")))

    summ(dev_r, "DEV ")
    summ(tst_r, "TEST")
    summ(dev_r + tst_r, "ALL ")

    # ---------- 3) top-vs-bottom decile spread ----------
    print()
    print("=" * 92)
    print("TEST 3: top-2 predicted vs bottom-2 predicted, same timestamp")
    print("=" * 92)
    dev_spread = []
    tst_spread = []
    for ts, g in groups.items():
        if len(g) < 6:
            continue
        s = sorted(g, key=lambda p: p["predicted_change_pct"])
        top = statistics.mean(p["_actual_change_pct"] for p in s[-2:])
        bot = statistics.mean(p["_actual_change_pct"] for p in s[:2])
        (tst_spread if ts >= half else dev_spread).append(top - bot)
    for vals, lab in ((dev_spread, "DEV "), (tst_spread, "TEST"), (dev_spread + tst_spread, "ALL ")):
        if not vals:
            continue
        m = statistics.mean(vals)
        se = statistics.stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0
        pos = sum(1 for v in vals if v > 0)
        print("%s: mean spread=%+.3f%% +- %.3f%%  groups=%d  positive=%.0f%%  t=%+.1f"
              % (lab, m, se, len(vals), 100.0 * pos / len(vals), m / se if se else 0))

    # ---------- 4) does the gate add cross-sectional value? ----------
    print()
    print("=" * 92)
    print("TEST 4: quality-gated predictions vs ungated (cross-sectional)")
    print("=" * 92)
    gated, ungated = [], []
    for ts, g in groups.items():
        med = statistics.median(p["_actual_change_pct"] for p in g)
        for p in g:
            if p["predicted_change_pct"] <= 0:
                continue
            rel = p["_actual_change_pct"] - med
            if p.get("quality_gate_passed"):
                gated.append(rel)
            else:
                ungated.append(rel)
    for vals, lab in ((gated, "gated UP  "), (ungated, "ungated UP")):
        if not vals:
            print("%s: empty" % lab)
            continue
        w = sum(1 for v in vals if v > 0)
        print("%s: n=%4d  win_vs_median=%.1f%%  mean_excess=%+.3f%%"
              % (lab, len(vals), 100.0 * w / len(vals), statistics.mean(vals)))


if __name__ == "__main__":
    main()
