"""Validate the two candidate STABLE signals with 4-way time splits and
direction controls, so we cannot be fooled by a bear-market artifact.

Candidates found so far:
  A) atr_pct >= 4/5   -> 60-77% early, 66-68% late (both > 50)
  B) regime=bullish   -> CONTRARIAN: market went DOWN after 'bullish' label
                         in BOTH halves (45% / 15% up-rate)
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
    if n == 0:
        print("%-52s empty" % label)
        return
    se = math.sqrt(max(a / 100 * (1 - a / 100), 1e-9) / n) * 100
    sig = "***" if a - 50 > 2 * se else ("*" if a - 50 > se else "")
    print("%-52s n=%4d acc=%5.1f%% +-%4.1f ret=%+6.2f%% %s" % (label, n, a, se, r, sig))


def main():
    rows = load()
    n = len(rows)
    q = [rows[i * n // 4:(i + 1) * n // 4] for i in range(4)]
    labels = ["Q1 (oldest)", "Q2", "Q3", "Q4 (newest)"]
    print("n=%d split into 4 time-quartiles" % n)
    for lab, part in zip(labels, q):
        mkt = 100.0 * sum(1 for p in part if p["_actual_change_pct"] > 0) / len(part)
        print("  %s: n=%d  dates %s..%s  market_up=%.0f%%" % (
            lab, len(part), (part[0].get('_created_at') or '')[:10],
            (part[-1].get('_created_at') or '')[:10], mkt))
    print()

    # ============ A) ATR candidate, controlled for direction ============
    print("=" * 100)
    print("A) ATR>=4  -- is it real, or just 'high vol means dump'?  (direction-controlled)")
    print("=" * 100)
    for lab, part in zip(labels, q):
        sel = [p for p in part if p.get("atr_pct", 0) >= 4]
        pr("%s | atr>=4 ALL" % lab, sel)
        pr("%s | atr>=4 & predicted UP" % lab, [p for p in sel if p["predicted_change_pct"] > 0])
        pr("%s | atr>=4 & predicted DOWN" % lab, [p for p in sel if p["predicted_change_pct"] < 0])
        if sel:
            pu = 100.0 * sum(1 for p in sel if p["predicted_change_pct"] > 0) / len(sel)
            mu = 100.0 * sum(1 for p in sel if p["_actual_change_pct"] > 0) / len(sel)
            print("      -> predUP=%.0f%%  mktUP=%.0f%%" % (pu, mu))
        print()

    print("=" * 100)
    print("A2) ATR thresholds sweep -- which cut is stable across all 4 quartiles?")
    print("=" * 100)
    print("%-10s %s" % ("atr>=", "  ".join("%-16s" % l for l in labels)))
    for t in (3, 4, 5, 6):
        cells = []
        for part in q:
            sel = [p for p in part if p.get("atr_pct", 0) >= t]
            a, cnt, r = stat(sel)
            cells.append("%5.1f%% n=%-4d" % (a, cnt) if cnt else "     --     ")
        print("%-10s %s" % (t, "  ".join("%-16s" % c for c in cells)))
    print()

    # ============ B) regime contrarian candidate ============
    print("=" * 100)
    print("B) regime label as CONTRARIAN signal (label bullish -> expect DOWN)")
    print("=" * 100)
    print("%-22s %s" % ("regime", "  ".join("%-18s" % l for l in labels)))
    for reg in ("bullish", "strong_bullish", "neutral", "bearish"):
        cells = []
        for part in q:
            sel = [p for p in part if p.get("market_regime") == reg]
            if len(sel) < 15:
                cells.append("n/a")
                continue
            up = 100.0 * sum(1 for p in sel if p["_actual_change_pct"] > 0) / len(sel)
            cells.append("mktUP=%4.1f%% n=%-3d" % (up, len(sel)))
        print("%-22s %s" % (reg, "  ".join("%-18s" % c for c in cells)))
    print()

    # Trade the contrarian rule: regime in (bullish,strong_bullish) -> must predict DOWN
    print("--- rule: regime BULLISH => only accept predicted DOWN (contrarian) ---")
    for lab, part in zip(labels, q):
        sel = [p for p in part
               if p.get("market_regime") in ("bullish", "strong_bullish")
               and p["predicted_change_pct"] < 0]
        pr("%s | bullish-label & DOWN" % lab, sel)
    print()
    print("--- inverse check: regime BEARISH => only accept predicted UP ---")
    for lab, part in zip(labels, q):
        sel = [p for p in part
               if p.get("market_regime") in ("bearish", "strong_bearish")
               and p["predicted_change_pct"] > 0]
        pr("%s | bearish-label & UP" % lab, sel)
    print()

    # ============ C) combination ============
    print("=" * 100)
    print("C) COMBINED: atr>=4  AND  direction consistent with contrarian regime")
    print("=" * 100)

    def combo(p):
        if p.get("atr_pct", 0) < 4:
            return False
        reg = p.get("market_regime", "neutral")
        if reg in ("bullish", "strong_bullish"):
            return p["predicted_change_pct"] < 0
        return True

    for lab, part in zip(labels, q):
        pr("%s | COMBO" % lab, [p for p in part if combo(p)])
    print()
    pr("ALL | COMBO", [p for p in rows if combo(p)])
    print()

    # ============ D) simple: does anything beat 50% reliably? ============
    print("=" * 100)
    print("D) SCORECARD -- every candidate, all 4 quartiles")
    print("=" * 100)
    cands = {
        "atr>=4": lambda p: p.get("atr_pct", 0) >= 4,
        "atr>=5": lambda p: p.get("atr_pct", 0) >= 5,
        "regime bull & DOWN": lambda p: p.get("market_regime") in ("bullish", "strong_bullish") and p["predicted_change_pct"] < 0,
        "atr>=4 & bull->DOWN": combo,
        "pred UP": lambda p: p["predicted_change_pct"] > 0,
        "pred DOWN": lambda p: p["predicted_change_pct"] < 0,
        "mag>=3": lambda p: abs(p["predicted_change_pct"]) >= 3,
        "funding<0": lambda p: p.get("funding_rate_pct") is not None and p["funding_rate_pct"] < 0,
    }
    print("%-24s %s  VERDICT" % ("rule", "  ".join("%-14s" % l for l in labels)))
    for name, fn in cands.items():
        cells = []
        accs = []
        for part in q:
            sel = [p for p in part if fn(p)]
            a, cnt, r = stat(sel)
            if cnt:
                cells.append("%.1f%%(%d)" % (a, cnt))
                accs.append(a)
            else:
                cells.append("--")
        if len(accs) == 4:
            verdict = "STABLE+" if all(a > 52 for a in accs) else (
                "stable" if all(a > 50 for a in accs) else "FLIP")
        else:
            verdict = "?"
        print("%-24s %s  %s" % (name, "  ".join("%-14s" % c for c in cells), verdict))


if __name__ == "__main__":
    main()
