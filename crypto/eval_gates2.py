"""Test the testable gate components for OOS lift (DEV vs TEST halves).

heuristic_confidence is NOT persisted historically, so the confidence gate
cannot be evaluated retroactively -- the others can:
  vol>=1.2, meta_reliable, meta_prob>=0.55, foundation-agrees
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


def pr(label, rows):
    if not rows:
        print("%-44s empty" % label)
        return
    n = len(rows)
    a = 100.0 * sum(1 for p in rows if correct(p)) / n
    ret = sum(p["_actual_change_pct"] * (1 if p["predicted_change_pct"] > 0 else -1) for p in rows) / n
    se = math.sqrt(max(a / 100 * (1 - a / 100), 1e-9) / n) * 100
    print("%-44s n=%4d acc=%5.1f%% +-%4.1f ret=%+6.2f%%" % (label, n, a, se, ret))


def main():
    rows = load()
    n = len(rows)
    dev, tst = rows[:n // 2], rows[n // 2:]

    tests = {
        "vol>=1.2": lambda p: p.get("volume_ratio", 9) >= 1.2,
        "vol>=0.8": lambda p: p.get("volume_ratio", 9) >= 0.8,
        "meta_reliable": lambda p: p.get("meta_labeling_reliable") is True,
        "meta_reliable & prob>=0.55": lambda p: (p.get("meta_labeling_reliable") is True
                                                 and (p.get("meta_labeling_prob") or 0) >= 0.55),
        "meta_reliable & prob<0.55": lambda p: (p.get("meta_labeling_reliable") is True
                                                and (p.get("meta_labeling_prob") or 9) < 0.55),
        "foundation agrees": lambda p: p.get("foundation_prediction") is None
                                       or (p["foundation_prediction"] > 0) == (p["predicted_change_pct"] > 0)
                                       or abs(p["foundation_prediction"]) < 1.0,
        "filters_passed>=4": lambda p: (p.get("filters_passed") or 0) >= 4,
        "filters_passed>=5": lambda p: (p.get("filters_passed") or 0) >= 5,
    }

    print("DEV n=%d baseline | TEST n=%d baseline" % (len(dev), len(tst)))
    pr("DEV  baseline", dev)
    pr("TEST baseline", tst)
    print()
    for name, fn in tests.items():
        d = [p for p in dev if fn(p)]
        t = [p for p in tst if fn(p)]
        pr("DEV  | " + name, d)
        pr("TEST | " + name, t)
        if d and t:
            a1 = 100.0 * sum(1 for p in d if correct(p)) / len(d)
            a2 = 100.0 * sum(1 for p in t if correct(p)) / len(t)
            b1 = 100.0 * sum(1 for p in dev if correct(p)) / len(dev)
            b2 = 100.0 * sum(1 for p in tst if correct(p)) / len(tst)
            verdict = "STABLE" if (a1 - b1 > 0 and a2 - b2 > 0) else "flip"
            print("      lift: DEV %+.1f  TEST %+.1f  -> %s" % (a1 - b1, a2 - b2, verdict))
        print()

    # intersection: volume + meta
    print("--- combo: vol>=1.2 AND meta_reliable AND foundation agrees ---")
    for tag, part in (("DEV ", dev), ("TEST", tst)):
        sel = [p for p in part if tests["vol>=1.2"](p)
               and tests["meta_reliable"](p) and tests["foundation agrees"](p)]
        pr(tag, sel)


if __name__ == "__main__":
    main()
