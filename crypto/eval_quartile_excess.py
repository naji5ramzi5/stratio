"""Validate the regime-neutral relative signals across 4 time quartiles.

The signal metric is EXCESS return vs same-timestamp median, which removes
market beta. A signal must be same-signed in all 4 quartiles to ship.
"""
import json
import math
import os
import statistics
from collections import defaultdict
from itertools import combinations

HIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_history.json")


def load():
    with open(HIST, encoding="utf-8") as f:
        rows = json.load(f)
    rows = [p for p in rows if p.get("_verified") and p.get("_actual_change_pct") is not None]
    rows.sort(key=lambda p: p.get("_created_at") or p.get("created_at") or "")
    # excess vs same-timestamp median over the WHOLE series
    by_ts = defaultdict(list)
    for p in rows:
        by_ts[(p.get("_created_at") or "")[:16]].append(p)
    for g in by_ts.values():
        med = statistics.median(x["_actual_change_pct"] for x in g)
        for x in g:
            x["_excess"] = x["_actual_change_pct"] - med
    return rows


def stat(sel):
    if len(sel) < 15:
        return None
    ex = [p["_excess"] for p in sel]
    m = statistics.mean(ex)
    se = statistics.stdev(ex) / math.sqrt(len(ex)) if len(ex) > 1 else 0
    return m, se, len(ex)


def main():
    rows = load()
    n = len(rows)
    q = [rows[i * n // 4:(i + 1) * n // 4] for i in range(4)]
    labs = ["Q1", "Q2", "Q3", "Q4"]

    feats = {
        "regime bull": lambda p: p.get("market_regime") in ("bullish", "strong_bullish"),
        "funding<0": lambda p: p.get("funding_rate_pct") is not None and p["funding_rate_pct"] < 0,
        "atr>=4": lambda p: p.get("atr_pct", 0) >= 4,
        "pred UP": lambda p: p["predicted_change_pct"] > 0,
        "neutral regime": lambda p: p.get("market_regime") == "neutral",
    }
    combos = {
        "bull & pred DOWN": lambda p: feats["regime bull"](p) and not feats["pred UP"](p),
        "bull & atr>=4": lambda p: feats["regime bull"](p) and feats["atr>=4"](p),
        "bull & funding<0": lambda p: feats["regime bull"](p) and feats["funding<0"](p),
        "funding<0 & atr>=4": lambda p: feats["funding<0"](p) and feats["atr>=4"](p),
        "NOT bull & pred UP": lambda p: not feats["regime bull"](p) and feats["pred UP"](p),
        "NOT bull & pred UP & funding>0": lambda p: (not feats["regime bull"](p)
                                                     and feats["pred UP"](p)
                                                     and p.get("funding_rate_pct") is not None
                                                     and p["funding_rate_pct"] > 0),
    }

    def row(label, fn):
        cells = []
        ms = []
        for part in q:
            sel = [p for p in part if fn(p)]
            s = stat(sel)
            if s is None:
                cells.append("n/a")
                ms.append(None)
                continue
            m, se, cnt = s
            ms.append(m)
            cells.append("%+.2f%%(%d)%s" % (m, cnt, "*" if se and m / se > 2 else ""))
        if None not in ms:
            verdict = "STABLE-" if all(m < -0.05 for m in ms) else (
                "STABLE+" if all(m > 0.05 for m in ms) else "flip")
        else:
            verdict = "?"
        print("%-30s %s  %s" % (label, "  ".join("%-14s" % c for c in cells), verdict))
        return ms

    print("EXCESS return vs same-timestamp median, per quartile  (* = t>2)")
    print("%-30s %s" % ("rule", "  ".join("%-14s" % l for l in labs)))
    print("-" * 100)
    all_ms = {}
    for k, v in feats.items():
        all_ms[k] = row(k, v)
    print()
    for k, v in combos.items():
        all_ms[k] = row(k, v)

    # overlap: how much do the stable fades co-occur?
    print()
    print("=" * 90)
    print("OVERLAP of the three fades (bull / funding<0 / atr>=4):")
    print("=" * 90)
    names = ["regime bull", "funding<0", "atr>=4"]
    for h, part in (("ALL", rows),):
        tot = len(part)
        for a, b in combinations(names, 2):
            both = sum(1 for p in part if feats[a](p) and feats[b](p))
            print("%-14s & %-14s n=%d (%.0f%% of all)" % (a, b, both, 100.0 * both / tot))

    # simple equal-weight fade basket: short all coins matching >=1 fade rule
    print()
    print("=" * 90)
    print("FADE BASKET: coin matches ANY of {bull, funding<0, atr>=4} -> excess")
    print("=" * 90)

    def fade(p):
        return feats["regime bull"](p) or feats["funding<0"](p) or feats["atr>=4"](p)

    for lab, part in zip(labs + ["ALL"], q + [rows]):
        sel = [p for p in part if fade(p)]
        s = stat(sel)
        if s:
            m, se, cnt = s
            print("%-5s excess=%+.3f%% +-%.3f n=%-4d t=%+.1f" % (lab, m, se, cnt, m / se if se else 0))

    # inverse basket
    print()
    print("CLEAN BASKET: matches NONE of the fades -> excess")
    for lab, part in zip(labs + ["ALL"], q + [rows]):
        sel = [p for p in part if not fade(p)]
        s = stat(sel)
        if s:
            m, se, cnt = s
            print("%-5s excess=%+.3f%% +-%.3f n=%-4d t=%+.1f" % (lab, m, se, cnt, m / se if se else 0))


if __name__ == "__main__":
    main()
