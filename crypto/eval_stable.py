"""Find STABLE (both-halves) signals. Focus on the two features whose
correctness-correlation kept the same sign in early and late halves:
funding_rate_pct (negative) and atr_pct (positive).

Also tests whether market_regime predicts MARKET direction (the only path
to 70% if the model itself has a long bias).
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


def stat(rows):
    if not rows:
        return 0.0, 0, 0.0
    ok = sum(1 for p in rows if correct(p))
    rets = [p["_actual_change_pct"] * (1 if p["predicted_change_pct"] > 0 else -1) for p in rows]
    return 100.0 * ok / len(rows), len(rows), sum(rets) / len(rets)


def pr(label, rows):
    a, n, r = stat(rows)
    se = math.sqrt(max(a / 100 * (1 - a / 100), 1e-9) / max(n, 1)) * 100
    print("%-46s n=%4d acc=%5.1f%% +-%4.1f ret=%+6.2f%%" % (label, n, a, se, r))


def main():
    rows = load()
    k = int(len(rows) * 0.6)
    halves = {"early": rows[:k], "late": rows[k:]}
    print("n=%d  early=%d late=%d" % (len(rows), k, len(rows) - k))
    base = stat(rows)
    print("BASELINE acc=%.1f%% ret=%+.2f%%" % (base[0], base[2]))
    print()

    # ---- 1. funding_rate sign test ----
    print("=" * 96)
    print("1) FUNDING RATE (stable r<0 in both halves)")
    print("=" * 96)
    for name, part in halves.items():
        have = [p for p in part if p.get("funding_rate_pct") is not None]
        for lo, hi, lab in [(-99, -0.01, "funding<0 (shorts pay)"),
                            (0.01, 99, "funding>0 (longs pay)"),
                            (0.05, 99, "funding>0.05 crowded long"),
                            (0.1, 99, "funding>0.10 very crowded")]:
            sel = [p for p in have if lo <= p["funding_rate_pct"] <= hi]
            if len(sel) >= 20:
                pr("%s | %s" % (name, lab), sel)
        print()

    # ---- 2. ATR test ----
    print("=" * 96)
    print("2) ATR (stable r>0 in both halves)")
    print("=" * 96)
    for name, part in halves.items():
        for t in (2, 3, 4, 5):
            sel = [p for p in part if p.get("atr_pct", 0) >= t]
            if len(sel) >= 20:
                pr("%s | atr>=%.d" % (name, t), sel)
        print()

    # ---- 3. market_regime predicting MARKET direction ----
    print("=" * 96)
    print("3) DOES market_regime PREDICT MARKET DIRECTION?")
    print("=" * 96)
    for name, part in halves.items():
        mkt_up = 100.0 * sum(1 for p in part if p["_actual_change_pct"] > 0) / len(part)
        print("%s: market up-rate = %.1f%%" % (name, mkt_up))
        byreg = defaultdict(list)
        for p in part:
            byreg[p.get("market_regime", "neutral")].append(p)
        for reg, sel in sorted(byreg.items()):
            if len(sel) < 20:
                continue
            up = 100.0 * sum(1 for p in sel if p["_actual_change_pct"] > 0) / len(sel)
            pr("   regime=%-16s -> market UP %.1f%%" % (reg, up), sel)
        print()

    # ---- 4. market_regime as gate: only trade WITH the regime ----
    print("=" * 96)
    print("4) TRADE WITH REGIME (UP only if bullish, DOWN only if bearish)")
    print("=" * 96)
    for name, part in halves.items():
        def with_regime(p):
            r = p.get("market_regime", "neutral")
            if p["predicted_change_pct"] > 0:
                return r in ("bullish", "strong_bullish")
            return r in ("bearish", "strong_bearish")
        sel = [p for p in part if with_regime(p)]
        pr("%s | with-regime only" % name, sel)
        sel2 = [p for p in part if not with_regime(p)]
        pr("%s | against-regime" % name, sel2)
        print()

    # ---- 5. combine the two STABLE features ----
    print("=" * 96)
    print("5) COMBINE STABLE FEATURES: funding sign matches prediction + atr")
    print("=" * 96)
    for name, part in halves.items():
        def fund_agrees(p):
            fr = p.get("funding_rate_pct")
            if fr is None:
                return False
            # crowded longs (fr>0) should mean DOWN, crowded shorts mean UP
            if p["predicted_change_pct"] > 0:
                return fr < 0
            return fr > 0
        sel = [p for p in part if fund_agrees(p)]
        pr("%s | funding opposes crowd" % name, sel)
        sel = [p for p in part if fund_agrees(p) and p.get("atr_pct", 0) >= 3]
        pr("%s | funding opposes crowd + atr>=3" % name, sel)
        sel = [p for p in part if fund_agrees(p) and p.get("atr_pct", 0) >= 4]
        pr("%s | funding opposes crowd + atr>=4" % name, sel)
        print()

    # ---- 6. CONTRARIAN test: is betting AGAINST the model better? ----
    print("=" * 96)
    print("6) CONTRARIAN: flip the model's call")
    print("=" * 96)
    for name, part in halves.items():
        a, n, r = stat(part)
        # flipped accuracy
        flipped = [p for p in part]
        ok = sum(1 for p in flipped if not correct(p))
        print("%s: model=%.1f%%  flipped=%.1f%%  n=%d" % (name, a, 100.0 * ok / len(part), len(part)))
        rets = [-p["_actual_change_pct"] * (1 if p["predicted_change_pct"] > 0 else -1) for p in part]
        print("      model ret=%+.2f%%  flipped ret=%+.2f%%" % (r, sum(rets) / len(rets)))


if __name__ == "__main__":
    main()
