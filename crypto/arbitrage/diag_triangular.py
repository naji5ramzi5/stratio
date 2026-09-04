"""Diagnostic: characterize the GROSS triangular cross-rate distribution.
Verifies engine sanity and reports how large raw deviations are, how often
they exceed given fee thresholds, and their persistence (half-life).
"""
import os
import sys
import json
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from arbitrage.storage import load_aggtrades  # noqa: E402
from arbitrage.replay import build_quote_series, merge_cycle_quotes  # noqa: E402
from arbitrage.triangular import cycle_rates  # noqa: E402
import numpy as np

START = date(2026, 7, 25)
END = date(2026, 8, 9)
DAYS = [(START + timedelta(days=i)).isoformat() for i in range((END - START).days + 1)]


def analyze(a: str, b: str):
    syms = [f"{a}USDT", f"{b}USDT", f"{b}{a}"]
    gross_above_0 = 0          # buckets where r > 1 (raw cross-rate edge, any size)
    thresholds = {5: 0, 10: 0, 15: 0, 20: 0, 30: 0, 50: 0}
    dist = []
    max_run = 0
    total_buckets = 0
    for day in DAYS:
        trades = {}
        ok = True
        for s in syms:
            try:
                trades[s] = load_aggtrades(s, day, "spot")
            except FileNotFoundError:
                ok = False
        if not ok:
            continue
        quotes = {s: build_quote_series(t, 100_000) for s, t in trades.items()}
        merged = merge_cycle_quotes(quotes, syms, 100_000)
        r1, r2 = cycle_rates(merged, a, b)
        for r in (r1, r2):
            e = (r - 1) * 1e4
            valid = np.isfinite(e)
            dist.append(e[valid])
            total_buckets += int(valid.sum())
            gross_above_0 += int((e > 0).sum())
            for t in thresholds:
                thresholds[t] += int((e > t).sum())
            # persistence: max consecutive buckets with e > 5bps
            m = e > 5
            d = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
            st = np.flatnonzero(d == 1); en = np.flatnonzero(d == -1)
            if len(st):
                max_run = max(max_run, int((en - st).max()))
    e_all = np.concatenate(dist)
    return {
        "cycle": f"{a}/{b}",
        "days_analyzed": total_buckets > 0,
        "buckets_total": total_buckets,
        "gross_buckets_gt_0": int(gross_above_0),
        "gross_pct_gt_0": round(100 * gross_above_0 / total_buckets, 4) if total_buckets else 0,
        "buckets_gt_thr_bps": {str(k): v for k, v in thresholds.items()},
        "pct_gt_thr_bps": {str(k): round(100 * v / total_buckets, 4) if total_buckets else 0
                           for k, v in thresholds.items()},
        "gross_bps_mean": round(float(e_all.mean()), 3),
        "gross_bps_std": round(float(e_all.std()), 3),
        "gross_bps_p01": round(float(np.percentile(e_all, 1)), 3),
        "gross_bps_p99": round(float(np.percentile(e_all, 99)), 3),
        "gross_bps_min": round(float(e_all.min()), 3),
        "gross_bps_max": round(float(e_all.max()), 3),
        "max_consecutive_buckets_gt_5bps": int(max_run),
        "max_consecutive_seconds": max_run * 0.1,
    }


def main():
    out = {"bucket_us": 100_000, "results": {}}
    for a, b in (("BTC", "ETH"), ("BTC", "BNB")):
        print(f"--- {a}/{b} ---")
        r = analyze(a, b)
        print(json.dumps(r, indent=1))
        out["results"][f"{a}_{b}"] = r
    path = os.path.join(os.path.dirname(__file__), "..", "ARBITRAGE_V1_DIAGNOSTIC.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
