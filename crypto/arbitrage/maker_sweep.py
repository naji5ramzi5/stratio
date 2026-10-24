"""ARBITRAGE_V4b: maker triangular on perps — threshold sweep + honest edge distribution.
The 17k 'candidates' from V4a were ~0.001bps noise. Here we report the actual
net-edge distribution and test whether ANY sustainable threshold exists.
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
DAYS = [(START + timedelta(days=i)) for i in range((END - START).days + 1)]
SYMBOLS = ["BTCUSDT", "ETHUSDT", "ETHBTC"]
MAKER_FEE_LEG_BPS = 2.0


def main():
    fee_net = MAKER_FEE_LEG_BPS * 3 / 1e4
    all_edges = []
    for day in DAYS:
        trades = {}
        for s in SYMBOLS:
            try:
                trades[s] = load_aggtrades(s, day.isoformat(), "futures_um")
            except FileNotFoundError:
                trades[s] = None
        if any(t is None for t in trades.values()):
            continue
        quotes = {s: build_quote_series(t, 100_000) for s, t in trades.items()}
        merged = merge_cycle_quotes(quotes, SYMBOLS, 100_000)
        r1, r2 = cycle_rates(merged, "BTC", "ETH")
        for r in (r1, r2):
            net = (r - 1) - fee_net
            all_edges.append(net[~np.isnan(net)])
    e = np.concatenate(all_edges) * 1e4  # net bps
    n = len(e)
    out = {
        "buckets": int(n),
        "fee_net_bps": fee_net * 1e4,
        "net_bps_mean": round(float(e.mean()), 4),
        "net_bps_std": round(float(e.std()), 4),
        "pct_positive": round(100 * float((e > 0).sum()) / n, 4),
    }
    print(f"buckets={n} mean={out['net_bps_mean']}bps std={out['net_bps_std']} "
          f"pct>0={out['pct_positive']}%")
    for thr in (2.0, 5.0, 8.0, 10.0, 15.0):
        cnt = int((e >= thr).sum())
        out[f"buckets_ge_{thr}bps"] = cnt
        out[f"pct_ge_{thr}bps"] = round(100 * cnt / n, 5)
        print(f"  >= {thr}bps: {cnt} buckets ({100*cnt/n:.5f}%)")
    out["p99"] = round(float(np.percentile(e, 99)), 3)
    out["max"] = round(float(e.max()), 3)
    path = os.path.join(os.path.dirname(__file__), "..", "ARBITRAGE_V4_MAKER_SWEEP.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
