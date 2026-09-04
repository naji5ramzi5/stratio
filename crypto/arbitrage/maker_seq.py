"""ARBITRAGE_V4d: maker triangular on perps — SEQUENTIAL 3-leg fill simulation.
The honest version: three passive orders rest on three books; each leg fills only
when a real trade arrives on that leg's side; the cycle completes when ALL legs
fill within a fill window. Legs that filled but fail to complete get unwound at
taker cost. This captures three-leg execution risk (spec #10 two-leg analog).
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
TAKER_FEE_LEG_BPS = 5.0
NOTIONAL = 100.0
BUCKET_US = 100_000


def load_all_days():
    series = {s: [] for s in SYMBOLS}
    for day in DAYS:
        for s in SYMBOLS:
            try:
                series[s].append(load_aggtrades(s, day.isoformat(), "futures_um"))
            except FileNotFoundError:
                pass
    return series


def simulate(thr_bps: float, fill_window_buckets: int, p_leg: float,
             rng: np.random.Generator, series: dict, latency_buckets: int = 0) -> dict:
    fee_net = MAKER_FEE_LEG_BPS * 3 / 1e4
    unwound_cost = TAKER_FEE_LEG_BPS + MAKER_FEE_LEG_BPS + 2.0  # taker extra + spread
    completed, failures, pnl = 0, 0, 0.0
    edge_list = []
    cand = 0
    for di in range(len(series["BTCUSDT"])):
        td = {s: series[s][di] for s in SYMBOLS}
        quotes = {s: build_quote_series(t, BUCKET_US) for s, t in td.items()}
        merged = merge_cycle_quotes(quotes, SYMBOLS, BUCKET_US)
        r1, r2 = cycle_rates(merged, "BTC", "ETH")
        n = len(merged["bucket_us"])
        for r in (r1, r2):
            net = (r - 1) - fee_net
            above = net * 1e4 >= thr_bps
            d = np.diff(np.concatenate(([0], above.astype(np.int8), [0])))
            starts = np.flatnonzero(d == 1)
            ends = np.flatnonzero(d == -1)
            cand += len(starts)
            for s_idx, e_idx in zip(starts, ends):
                s_idx += latency_buckets  # decision latency before quoting
                if s_idx >= n:
                    continue
                win_end = min(s_idx + fill_window_buckets, n)
                filled = 0
                b = s_idx
                for b in range(s_idx + 1, win_end):
                    edge_now = net[b] * 1e4
                    if edge_now < thr_bps:
                        break
                    for _ in range(3 - filled):
                        if rng.random() <= p_leg:
                            filled += 1
                    if filled == 3:
                        break
                if filled == 3:
                    completed += 1
                    g = net[b] * 1e4
                    pnl += g / 1e4 * NOTIONAL
                    edge_list.append(g)
                elif filled > 0:
                    failures += 1
                    pnl -= unwound_cost / 1e4 * NOTIONAL * (filled / 3)
    return {
        "candidates": cand,
        "completed": completed,
        "failures": failures,
        "net_pnl_usd": round(pnl, 2),
        "net_bps_mean": round(float(np.mean(edge_list)), 3) if edge_list else 0.0,
        "profitable": int(np.sum([e > 0 for e in edge_list])),
    }


def main():
    series = load_all_days()
    rng = np.random.default_rng(11)
    out = {"fee_net_bps": MAKER_FEE_LEG_BPS * 3, "notional_usd": NOTIONAL, "results": {}}
    for thr in (5.0, 8.0, 10.0):
        for win in (5, 10):
            for p in (1.0, 0.5, 0.2, 0.1):
                for lat in (0, 2):
                    key = f"thr{thr}_win{win*100}ms_p{p}_lat{lat*100}ms"
                    r = simulate(thr, win, p, rng, series, lat)
                    out["results"][key] = r
                    print(f"{key:32s} cand={r['candidates']:6d} done={r['completed']:5d} "
                          f"fail={r['failures']:5d} pnl=${r['net_pnl_usd']:9.2f} "
                          f"mean={r['net_bps_mean']:6.3f}bps")
    path = os.path.join(os.path.dirname(__file__), "..", "ARBITRAGE_V4D_MAKER_SEQ.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
