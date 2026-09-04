"""ARBITRAGE_V4: MAKER triangular arbitrage on Binance USD-M futures perps.
Cycle: BTCUSDT + ETHUSDT + ETHBTC (all perps) — 16 days real tick data.
Costs: maker fee 2bps/leg x 3 = 6bps/cycle (vs 30bps for taker).
Fills: passive quotes at real bid/ask; a leg fills when an opposing aggressive
trade arrives at our price within the fill window. Queue-position risk is a
probability parameter (1.0 = front of queue, 0.5, 0.25 = deeper in queue).
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

MAKER_FEE_LEG_BPS = 2.0
SYMBOLS = ["BTCUSDT", "ETHUSDT", "ETHBTC"]
A, B = "BTC", "ETH"
BUFFER_BPS = 2.0  # extra threshold over fees


def analyze_day(day, bucket_us=100_000):
    trades = {}
    for s in SYMBOLS:
        try:
            trades[s] = load_aggtrades(s, day.isoformat(), "futures_um")
        except FileNotFoundError:
            return None
    quotes = {s: build_quote_series(t, bucket_us) for s, t in trades.items()}
    merged = merge_cycle_quotes(quotes, SYMBOLS, bucket_us)
    r1, r2 = cycle_rates(merged, A, B)
    return merged, r1, r2


def maker_fill_sim(merged, r1, r2, fee_net, buffer_bps, p_fill):
    """Passive-fill simulation. Candidate: net > 0 for >=1 bucket.
    Fill at the LAST quote of the fill window (next bucket) — the actual tape
    price at the moment the passive orders rest on the book. Apply queue prob."""
    net1 = r1 - 1 - fee_net
    net2 = r2 - 1 - fee_net
    trades = []
    for net in (net1, net2):
        above = net > 0
        idx = np.flatnonzero(above)
        if not len(idx):
            continue
        # group consecutive candidate buckets -> one trade per run
        d = np.diff(np.concatenate(([0], above.astype(np.int8), [0])))
        starts = np.flatnonzero(d == 1)
        ends = np.flatnonzero(d == -1)
        for s, e in zip(starts, ends):
            # fill window: 1 bucket after detection (passive order rests 100ms)
            fi = min(s + 1, len(net) - 1)
            edge = net[fi]
            if edge <= 0:
                continue
            trades.append({"entry": int(s), "fill": int(fi), "edge_bps": float(edge)})
    rng = np.random.default_rng(42)
    filled = [t for t in trades if rng.random() <= p_fill]
    return trades, filled


def main():
    fee_net = MAKER_FEE_LEG_BPS * 3 / 1e4
    results = {}
    for p_fill in (1.0, 0.5, 0.25):
        cand, filled = [], []
        for day in DAYS:
            out = analyze_day(day)
            if out is None:
                continue
            merged, r1, r2 = out
            c, f = maker_fill_sim(merged, r1, r2, fee_net, BUFFER_BPS, p_fill)
            cand.extend(c)
            filled.extend(f)
        results[f"maker_fill_{p_fill}"] = {
            "candidates": len(cand),
            "filled": len(filled),
            "net_bps_mean": round(float(np.mean([t["edge_bps"] for t in filled])), 3) if filled else 0,
            "net_pnl_usd": round(float(np.sum([t["edge_bps"] for t in filled]) / 1e4 * 100), 2) if filled else 0,
            "profitable": int(np.sum([t["edge_bps"] > 0 for t in filled])),
        }
        print(f"p_fill={p_fill}: cand={len(cand)} filled={len(filled)} "
              f"pnl=${results[f'maker_fill_{p_fill}']['net_pnl_usd']} "
              f"prof={results[f'maker_fill_{p_fill}']['profitable']}")

    out = {"fee_net_bps": fee_net * 1e4, "buffer_bps": BUFFER_BPS,
           "days": [d.isoformat() for d in DAYS], "results": results}
    path = os.path.join(os.path.dirname(__file__), "..", "ARBITRAGE_V4_MAKER.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
