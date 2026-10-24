"""Walk-forward gate evaluator.

Any candidate gate must pass BOTH time-halves of the verified history with the
same sign of edge, otherwise it is regime-fit noise and must not ship.

Usage:
    python eval_gates.py
"""
import json
import math
import os
import sys

HIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_history.json")
OLD_HIST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "prediction_accuracy_verified.json")


def load_verified():
    rows = []
    if os.path.exists(HIST):
        with open(HIST, encoding="utf-8") as f:
            data = json.load(f)
        rows.extend(p for p in data
                    if p.get("_verified") and p.get("_actual_change_pct") is not None
                    and p.get("volume_ratio") is not None)
    return rows


def correct(p):
    return (p["_actual_change_pct"] > 0) == (p["predicted_change_pct"] > 0)


def stat(rows):
    if not rows:
        return 0.0, 0, 0.0
    ok = sum(1 for p in rows if correct(p))
    acc = 100.0 * ok / len(rows)
    rets = [p["_actual_change_pct"] * (1 if p["predicted_change_pct"] > 0 else -1)
            for p in rows]
    return acc, len(rows), sum(rets) / len(rets)


def gate_fails(p, min_vol, min_conf, require_meta=True, require_fnd=True):
    fails = []
    if (p.get("volume_ratio") or 1) < min_vol:
        fails.append("low_volume")
    fp = p.get("filters_passed", 0)
    conf = 50 + fp * 7
    if abs(p.get("predicted_change_pct", 0)) < 1:
        conf -= 15
    if conf < min_conf:
        fails.append("low_confidence")
    if require_meta and not p.get("meta_labeling_reliable", True):
        fails.append("meta")
    fnd = p.get("foundation_prediction")
    if require_fnd and fnd is not None:
        pc = p.get("predicted_change_pct", 0)
        if (fnd > 0) != (pc > 0) and abs(fnd) >= 1.0:
            fails.append("fnd")
    return fails


def evaluate(name, rows, half, min_vol, min_conf, mag_min, meta_on, fnd_on):
    base_acc, base_n, base_ret = stat(rows)
    pas = [p for p in rows if not gate_fails(p, min_vol, min_conf, meta_on, fnd_on)
           and abs(p.get("predicted_change_pct", 0)) >= mag_min]
    rej = [p for p in rows if p not in pas]
    a_p, n_p, r_p = stat(pas)
    a_r, _, _ = stat(rej)
    return {
        "name": name, "half": half,
        "base_acc": base_acc, "base_n": base_n,
        "pass_acc": a_p, "pass_n": n_p, "pass_rate": 100.0 * n_p / max(base_n, 1),
        "pass_ret": r_p, "rej_acc": a_r,
        "lift": a_p - base_acc,
    }


def main():
    rows = load_verified()
    if len(rows) < 200:
        print("not enough verified rows:", len(rows))
        return
    rows.sort(key=lambda p: p.get("_created_at") or p.get("created_at") or "")
    k = int(len(rows) * 0.6)
    halves = {"early": rows[:k], "late": rows[k:]}

    print("=" * 100)
    print("WALK-FORWARD GATE EVALUATION  (n=%d verified)" % len(rows))
    print("A gate ships only if lift>0 in BOTH halves AND pass_rate>=8%.")
    print("=" * 100)

    combos = []
    for min_vol in (1.0, 1.1, 1.2):
        for min_conf in (50, 55, 60):
            for mag_min in (0, 3):
                combos.append((min_vol, min_conf, mag_min, True, True))

    results = []
    for min_vol, min_conf, mag_min, meta_on, fnd_on in combos:
        label = "vol>=%.1f conf>=%d mag>=%d meta=%s" % (
            min_vol, min_conf, mag_min, "on" if meta_on else "off")
        e = evaluate(label, halves["early"], "early", min_vol, min_conf, mag_min, meta_on, fnd_on)
        l = evaluate(label, halves["late"], "late", min_vol, min_conf, mag_min, meta_on, fnd_on)
        results.append((label, e, l))

    print("%-42s | %-30s | %-30s | %s" % (
        "gate", "EARLY", "LATE", "verdict"))
    print("-" * 130)
    shipped = []
    for label, e, l in results:
        stable = (e["lift"] > 0 and l["lift"] > 0
                  and e["pass_rate"] >= 8 and l["pass_rate"] >= 8)
        verdict = "SHIP" if stable else "reject"
        if stable:
            shipped.append((label, e, l))
        print("%-42s | acc=%5.1f n=%4d lift=%+5.1f | acc=%5.1f n=%4d lift=%+5.1f | %s" % (
            label, e["pass_acc"], e["pass_n"], e["lift"],
            l["pass_acc"], l["pass_n"], l["lift"], verdict))

    print()
    if shipped:
        print("SHIPPED GATES:")
        for label, e, l in shipped:
            print("  %s  -> combined acc=%.1f%% ret=%.2f%%" % (
                label, (e["pass_acc"] + l["pass_acc"]) / 2, (e["pass_ret"] + l["pass_ret"]) / 2))
    else:
        print("NO GATE SHIPS. Directional edge is not exploitable by these filters.")
        print("Baseline accuracy early=%.1f%% late=%.1f%%" % (
            halves['early'] and stat(halves['early'])[0], stat(halves['late'])[0]))

    print()
    print("=" * 100)
    print("DIRECTION SPLIT (the regime test)")
    print("=" * 100)
    for name, part in halves.items():
        up = [p for p in part if p["predicted_change_pct"] > 0]
        dn = [p for p in part if p["predicted_change_pct"] < 0]
        au, nu, ru = stat(up)
        ad, nd, rd = stat(dn)
        mk_up = 100.0 * sum(1 for p in part if p["_actual_change_pct"] > 0) / len(part)
        print("%s: market up=%.0f%% | UP n=%d acc=%.1f%% ret=%+.2f%% | DOWN n=%d acc=%.1f%% ret=%+.2f%%"
              % (name, mk_up, nu, au, ru, nd, ad, rd))


if __name__ == "__main__":
    main()
