"""ARBITRAGE_V4c: maker triangular on perps — correct execution simulation.
- Requires net edge >= thr at detection AND after the fill window (adverse selection).
- Fill price = real tape price at detection+latency (passive orders at touch).
- Queue stress: p_fill applied per completed cycle.
- Capital: $100/cycle, prefunded, no compounding.
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
NOTIONAL = 100.0


def load_all_days():
    series = {s: [] for s in SYMBOLS}
    for day in DAYS:
        for s in SYMBOLS:
            try:
                series[s].append(load_aggtrades(s, day.isoformat(), "futures_um"))
            except FileNotFoundError:
                pass
    return series


def simulate(thr_bps: float, latency_buckets: int, p_fill: float,
             rng: np.random.Generator, series: dict) -> dict:
    fee_net = MAKER_FEE_LEG_BPS * 3 / 1e4
    trades = []
    candidates = 0
    for day_idx in range(len(series["BTCUSDT"])):
        trades_day = {s: series[s][day_idx] for s in SYMBOLS}
        quotes = {s: build_quote_series(t, 100_000) for s, t in trades_day.items()}
        merged = merge_cycle_quotes(quotes, SYMBOLS, 100_000)
        r1, r2 = cycle_rates(merged, "BTC", "ETH")
        for r in (r1, r2):
            net = (r - 1) - fee_net
            n = len(net)
            above = net * 1e4 >= thr_bps
            d = np.diff(np.concatenate(([0], above.astype(np.int8), [0])))
            starts = np.flatnonzero(d == 1)
            ends = np.flatnonzero(d == -1)
            for s_idx, e_idx in zip(starts, ends):
                candidates += 1
                fi = s_idx + latency_buckets
                if fi >= n:
                    continue
                edge_bps = net[fi] * 1e4
                if edge_bps < thr_bps:
                    continue  # opportunity gone when we arrive
                trades.append(edge_bps)
    if not trades:
        return {"candidates": candidates, "filled": 0, "net_pnl_usd": 0.0,
                "net_bps_mean": 0.0, "profitable": 0}
    edges = np.array(trades)
    if p_fill < 1.0:
        keep = rng.random(len(edges)) <= p_fill
        edges = edges[keep]
    return {
        "candidates": candidates,
        "filled": int(len(edges)),
        "net_bps_mean": round(float(edges.mean()), 3),
        "net_pnl_usd": round(float(edges.sum() / 1e4 * NOTIONAL), 2),
        "profitable": int((edges > 0).sum()),
    }


def main():
    series = load_all_days()
    rng = np.random.default_rng(7)
    out = {"fee_net_bps": MAKER_FEE_LEG_BPS * 3, "notional_usd": NOTIONAL, "results": {}}
    for thr in (5.0, 8.0, 10.0, 15.0):
        for lat in (0, 1, 5):
            for p in (1.0, 0.5):
                key = f"thr{thr}_lat{lat*100}ms_p{p}"
                r = simulate(thr, lat, p, rng, series)
                out["results"][key] = r
                print(f"{key:22s} cand={r['candidates']:7d} filled={r['filled']:6d} "
                      f"pnl=${r['net_pnl_usd']:9.2f} mean={r['net_bps_mean']:6.3f}bps "
                      f"prof={r['profitable']}")
    path = os.path.join(os.path.dirname(__file__), "..", "ARBITRAGE_V4C_MAKER_SIM.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
