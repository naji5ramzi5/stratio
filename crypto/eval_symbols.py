"""Per-symbol stability test.

If the model has ANY real skill it should concentrate somewhere. A symbol
whose directional accuracy is above baseline in BOTH time halves (with
adequate n) is a candidate for a symbol whitelist. Anything that flips is
regime noise.
"""
import json
import math
import os
from collections import defaultdict

HIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_history.json")


def load():
    with open(HIST, encoding="utf-8") as f:
        rows = json.load(f)
    rows = [p for p in rows if p.get("_verified") and p.get("_actual_change_pct") is not None]
    rows.sort(key=lambda p: p.get("_created_at") or p.get("created_at") or "")
    return rows


def correct(p):
    return (p["_actual_change_pct"] > 0) == (p["predicted_change_pct"] > 0)


def acc(rows):
    if not rows:
        return None, 0
    return 100.0 * sum(1 for p in rows if correct(p)) / len(rows), len(rows)


def main():
    rows = load()
    n = len(rows)
    dev, tst = rows[:n // 2], rows[n // 2:]

    by_sym_dev = defaultdict(list)
    by_sym_tst = defaultdict(list)
    for p in dev:
        by_sym_dev[p["symbol"]].append(p)
    for p in tst:
        by_sym_tst[p["symbol"]].append(p)

    base_d, _ = acc(dev)
    base_t, _ = acc(tst)
    print("DEV baseline=%.1f%%  TEST baseline=%.1f%%  (n=%d/%d)" % (base_d, base_t, len(dev), len(tst)))
    print()

    results = []
    for sym in sorted(set(by_sym_dev) | set(by_sym_tst)):
        a1, n1 = acc(by_sym_dev.get(sym, []))
        a2, n2 = acc(by_sym_tst.get(sym, []))
        if not a1 or not a2 or n1 < 40 or n2 < 40:
            continue
        results.append((sym, a1, n1, a2, n2))

    # stable = above own-half baseline in both halves
    stable = []
    for sym, a1, n1, a2, n2 in results:
        lift1 = a1 - base_d
        lift2 = a2 - base_t
        if lift1 > 0 and lift2 > 0:
            stable.append((sym, a1, n1, a2, n2, lift1, lift2))

    results.sort(key=lambda r: -(min(r[1] - base_d, r[3] - base_t)))
    print("%-14s %-26s %-26s %s" % ("symbol", "DEV", "TEST", "min-lift"))
    print("%-14s %-26s %-26s %s" % ("", "acc     n", "acc     n", ""))
    for sym, a1, n1, a2, n2 in results:
        l1, l2 = a1 - base_d, a2 - base_t
        mark = "STABLE" if l1 > 0 and l2 > 0 else ("flip" if l1 * l2 < 0 else "weak")
        print("%-14s %5.1f%%  %-4d   %5.1f%%  %-4d   %+5.1f/%+5.1f  %s"
              % (sym, a1, n1, a2, n2, l1, l2, mark))

    print()
    print("=" * 80)
    print("STABLE (positive lift vs baseline in BOTH halves): %d" % len(stable))
    for sym, a1, n1, a2, n2, l1, l2 in stable:
        print("  %-14s DEV %5.1f%% (n=%d)  TEST %5.1f%% (n=%d)" % (sym, a1, n1, a2, n2))
    if stable:
        # pooled performance of the stable whitelist, held out on TEST
        sel = [p for p in tst if p["symbol"] in {s[0] for s in stable}]
        a, c = acc(sel)
        print()
        print("WHITELIST on TEST: acc=%.1f%% (n=%d) vs TEST baseline %.1f%%" % (a, c, base_t))
        # worst-case: does whitelist beat baseline by more than 1 SE?
        if c:
            se = math.sqrt(max(a / 100 * (1 - a / 100), 1e-9) / c) * 100
            print("  margin=%+.1f%%  SE=%.1f%%  -> %s"
                  % (a - base_t, se, "REAL" if a - base_t > 2 * se else "within noise"))

    # how many symbols were even testable
    print()
    print("symbols with n>=40 in both halves: %d of %d" % (len(results), len(set(by_sym_dev) | set(by_sym_tst))))


if __name__ == "__main__":
    main()
