"""Regime-neutral feature test: does a feature predict EXCESS return
(coin return minus same-timestamp median) in BOTH time halves?

Absolute accuracy is contaminated by market beta. Excess return is not.
"""
import json
import os
import statistics
from collections import defaultdict

HIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_history.json")


def load():
    with open(HIST, encoding="utf-8") as f:
        rows = json.load(f)
    rows = [p for p in rows if p.get("_verified") and p.get("_actual_change_pct") is not None]
    rows.sort(key=lambda p: p.get("_created_at") or p.get("created_at") or "")
    return rows


def main():
    rows = load()
    n = len(rows)
    halves = {"DEV": rows[:n // 2], "TEST": rows[n // 2:]}

    # excess return vs same-timestamp median
    for part in halves.values():
        by_ts = defaultdict(list)
        for p in part:
            by_ts[(p.get("_created_at") or "")[:16]].append(p)
        for g in by_ts.values():
            med = statistics.median(x["_actual_change_pct"] for x in g)
            for x in g:
                x["_excess"] = x["_actual_change_pct"] - med

    feats = {
        "regime bull/strong_bull": lambda p: p.get("market_regime") in ("bullish", "strong_bullish"),
        "regime neutral": lambda p: p.get("market_regime") == "neutral",
        "regime bearish": lambda p: p.get("market_regime") in ("bearish", "strong_bearish"),
        "funding<0": lambda p: p.get("funding_rate_pct") is not None and p["funding_rate_pct"] < 0,
        "funding>0": lambda p: p.get("funding_rate_pct") is not None and p["funding_rate_pct"] > 0,
        "atr>=4": lambda p: p.get("atr_pct", 0) >= 4,
        "atr>=5": lambda p: p.get("atr_pct", 0) >= 5,
        "vol_ratio>=1.2": lambda p: p.get("volume_ratio", 0) >= 1.2,
        "pred UP": lambda p: p["predicted_change_pct"] > 0,
        "pred DOWN": lambda p: p["predicted_change_pct"] < 0,
    }

    def stat(sel):
        if len(sel) < 20:
            return None
        ex = [p["_excess"] for p in sel]
        m = statistics.mean(ex)
        se = statistics.stdev(ex) / math.sqrt(len(ex)) if len(ex) > 1 else 0
        pos = sum(1 for v in ex if v > 0)
        return m, se, len(ex), 100.0 * pos / len(ex)

    import math
    print("%-26s %-34s %-34s %s" % ("feature", "DEV excess vs median", "TEST excess vs median", "VERDICT"))
    for name, fn in feats.items():
        cells = []
        ms = []
        for h in ("DEV", "TEST"):
            sel = [p for p in halves[h] if fn(p)]
            s = stat(sel)
            if s is None:
                cells.append("n/a")
                ms.append(None)
                continue
            m, se, cnt, pos = s
            ms.append(m)
            cells.append("%+.3f%% +-%.3f n=%-4d win=%.0f%%" % (m, se, cnt, pos))
        if None not in ms:
            verdict = "STABLE-" if ms[0] < -0.05 and ms[1] < -0.05 else (
                "STABLE+" if ms[0] > 0.05 and ms[1] > 0.05 else "flip/noise")
        else:
            verdict = "?"
        print("%-26s %-34s %-34s %s" % (name, cells[0], cells[1], verdict))

    # the interesting one, in detail: bull label fade, with sizing
    print()
    print("=" * 90)
    print("FADE THE BULL LABEL: regime bull => coin underperforms median?")
    print("=" * 90)
    for h in ("DEV", "TEST"):
        sel = [p for p in halves[h] if p.get("market_regime") in ("bullish", "strong_bullish")]
        s = stat(sel)
        if s:
            m, se, cnt, pos = s
            print("%s: excess=%+.3f%% +-%.3f n=%d below-median=%.0f%%  t=%+.1f"
                  % (h, m, se, cnt, 100 - pos, m / se if se else 0))
    # combined with predicted direction (must predict DOWN to act)
    print()
    print("  + only when model predicts DOWN (actionable):")
    for h in ("DEV", "TEST"):
        sel = [p for p in halves[h]
               if p.get("market_regime") in ("bullish", "strong_bullish")
               and p["predicted_change_pct"] < 0]
        s = stat(sel)
        if s:
            m, se, cnt, pos = s
            print("  %s: excess=%+.3f%% +-%.3f n=%d win=%.0f%% t=%+.1f" % (h, m, se, cnt, pos, m / se if se else 0))


if __name__ == "__main__":
    main()
