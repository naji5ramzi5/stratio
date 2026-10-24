"""Strict forward validation of the two surviving candidates.

Rule is DESIGNED on Q1+Q2 only, then locked and tested on Q3+Q4.
Direction is always controlled so a bear-market artifact cannot masquerade
as skill.
"""
import json
import math
import os

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
        print("%-50s empty" % label)
        return a, n, r
    se = math.sqrt(max(a / 100 * (1 - a / 100), 1e-9) / n) * 100
    print("%-50s n=%4d acc=%5.1f%% +-%4.1f ret=%+6.2f%%" % (label, n, a, se, r))
    return a, n, r


def main():
    rows = load()
    n = len(rows)
    dev = rows[:n // 2]        # Q1+Q2  -- design only
    tst = rows[n // 2:]        # Q3+Q4  -- locked test
    print("DEV n=%d (%s..%s)   TEST n=%d (%s..%s)" % (
        len(dev), (dev[0].get('_created_at') or '')[:10], (dev[-1].get('_created_at') or '')[:10],
        len(tst), (tst[0].get('_created_at') or '')[:10], (tst[-1].get('_created_at') or '')[:10]))
    mkt = lambda s: 100.0 * sum(1 for p in s if p["_actual_change_pct"] > 0) / len(s)
    print("market up-rate: DEV=%.0f%%  TEST=%.0f%%" % (mkt(dev), mkt(tst)))
    print()
    pr("DEV  baseline", dev)
    pr("TEST baseline", tst)
    print()

    # ---------- Candidate 1: atr >= 5 ----------
    print("=" * 92)
    print("CANDIDATE 1: atr_pct >= 5")
    print("=" * 92)
    for tag, part in (("DEV ", dev), ("TEST", tst)):
        sel = [p for p in part if p.get("atr_pct", 0) >= 5]
        pr("%s | atr>=5 ALL" % tag, sel)
        pr("%s | atr>=5 & UP" % tag, [p for p in sel if p["predicted_change_pct"] > 0])
        pr("%s | atr>=5 & DOWN" % tag, [p for p in sel if p["predicted_change_pct"] < 0])
        if sel:
            pu = 100.0 * sum(1 for p in sel if p["predicted_change_pct"] > 0) / len(sel)
            mu = 100.0 * sum(1 for p in sel if p["_actual_change_pct"] > 0) / len(sel)
            print("        predUP=%.0f%% mktUP=%.0f%%" % (pu, mu))
    # always-DOWN baseline on same rows (the confound check)
    for tag, part in (("DEV ", dev), ("TEST", tst)):
        sel = [p for p in part if p.get("atr_pct", 0) >= 5]
        if sel:
            dn = [p for p in sel if p["predicted_change_pct"] < 0]
            if dn:
                a, _, _ = stat(dn)
                print("        [confound] naive always-DOWN on same rows = %.1f%%" % a)
    print()

    # ---------- Candidate 2: contrarian regime ----------
    print("=" * 92)
    print("CANDIDATE 2: regime labelled bullish/strong_bullish => require DOWN")
    print("=" * 92)
    for tag, part in (("DEV ", dev), ("TEST", tst)):
        sel = [p for p in part
               if p.get("market_regime") in ("bullish", "strong_bullish")
               and p["predicted_change_pct"] < 0]
        pr("%s | bull-label & DOWN" % tag, sel)
        # the confound: is the label just a lagging pump detector?
        bull_rows = [p for p in part if p.get("market_regime") in ("bullish", "strong_bullish")]
        if bull_rows:
            print("        [label base] bull-label rows: n=%d mktUP=%.0f%% (label alone = bet DOWN)"
                  % (len(bull_rows), mkt(bull_rows)))
            naive = [p for p in bull_rows if p["_actual_change_pct"] < 0]
            ok = len(naive)
            print("        [confound] naive 'always DOWN after bull label' = %.1f%% (n=%d)"
                  % (100.0 * ok / len(bull_rows), len(bull_rows)))
    print()

    # ---------- Combined ----------
    print("=" * 92)
    print("COMBINED: atr>=5 AND (not bull-labelled OR predicted DOWN)")
    print("=" * 92)

    def combo(p):
        if p.get("atr_pct", 0) < 5:
            return False
        if p.get("market_regime") in ("bullish", "strong_bullish"):
            return p["predicted_change_pct"] < 0
        return True

    for tag, part in (("DEV ", dev), ("TEST", tst)):
        pr("%s | COMBO" % tag, [p for p in part if combo(p)])
    print()

    # ---------- Simpler alternative: does the MODEL's own call add value
    #             over the naive label-only rule? ----------
    print("=" * 92)
    print("VALUE-ADD TEST: model call vs naive label rule (candidate 2)")
    print("=" * 92)
    for tag, part in (("DEV ", dev), ("TEST", tst)):
        bull = [p for p in part if p.get("market_regime") in ("bullish", "strong_bullish")]
        if not bull:
            continue
        naive_acc = 100.0 * sum(1 for p in bull if p["_actual_change_pct"] < 0) / len(bull)
        model = [p for p in bull if p["predicted_change_pct"] < 0]
        m_acc, _, _ = stat(model)
        up_rows = [p for p in bull if p["predicted_change_pct"] > 0]
        u_acc, _, _ = stat(up_rows)
        print("%s: naive(all DOWN bet)=%.1f%%(n=%d) | model-says-DOWN=%.1f%%(n=%d) | model-says-UP=%.1f%%(n=%d)"
              % (tag, naive_acc, len(bull), m_acc, len(model), u_acc, len(up_rows)))
    print()

    # ---------- Final scorecard on TEST only ----------
    print("=" * 92)
    print("FINAL TEST-SET SCORECARD (locked, never used for design)")
    print("=" * 92)
    cands = {
        "atr>=5": lambda p: p.get("atr_pct", 0) >= 5,
        "atr>=5 & DOWN": lambda p: p.get("atr_pct", 0) >= 5 and p["predicted_change_pct"] < 0,
        "bull-label & DOWN": lambda p: p.get("market_regime") in ("bullish", "strong_bullish") and p["predicted_change_pct"] < 0,
        "COMBO": combo,
        "pred UP": lambda p: p["predicted_change_pct"] > 0,
        "pred DOWN": lambda p: p["predicted_change_pct"] < 0,
    }
    for name, fn in cands.items():
        pr("TEST | " + name, [p for p in tst if fn(p)])
    pr("TEST | baseline (all)", tst)


if __name__ == "__main__":
    main()
