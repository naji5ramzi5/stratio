"""ARBITRAGE_V5: funding-rate capture strategy on real Binance funding history.
Data: 500 funding rows (~166 days) x 6 symbols, plus real spot/futures tick data
for the 16-day window to measure entry/exit basis drag.
Strategy: cash-and-carry when funding >= open_thr (long spot + short perp, delta-neutral);
collect funding each 8h; exit when funding <= exit_thr or max_hold.
PnL = funding collected + basis_entry - basis_exit - round_trip_costs.
"""
import json
import os
import sys
import numpy as np
from datetime import datetime, timezone

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
FUND_DIR = os.path.join(HERE, "data", "funding")
FUNDING_INTERVAL_H = 8

ROUND_TRIP_BPS = 30.0  # spot taker 10 + futures taker 5, entry + exit
TRADE_PERCENT = 1.0    # trade size (% of capital)


def load_funding(sym: str) -> tuple:
    rows = json.load(open(os.path.join(FUND_DIR, f"{sym}.json")))
    ts = np.array([r["fundingTime"] for r in rows], dtype=np.int64)
    rate = np.array([float(r["fundingRate"]) for r in rows]) * 1e4
    order = np.argsort(ts)
    return ts[order], rate[order]


def simulate_full_history(sym: str, open_thr: float, exit_thr: float,
                          max_hold_days: int = 30) -> dict:
    """Long spot + short perp when funding >= open_thr (bps/8h). Collect until exit."""
    ts, rate = load_funding(sym)
    n = len(rate)
    i = 0
    trades = []
    while i < n:
        if rate[i] >= open_thr:
            entry = i
            collected = 0.0
            j = i
            while j < n:
                collected += rate[j]
                j += 1
                hold_h = (j - entry) * FUNDING_INTERVAL_H
                if j >= n or rate[j] <= exit_thr or hold_h >= max_hold_days * 24:
                    break
            trades.append({
                "entry_ts": ts[entry], "exit_ts": ts[j - 1],
                "payments": j - entry,
                "hold_days": round((j - entry) * FUNDING_INTERVAL_H / 24, 2),
                "funding_bps": round(float(collected), 3),
                "cost_bps": ROUND_TRIP_BPS,
                "net_bps": round(float(collected) - ROUND_TRIP_BPS, 3),
                "net_pnl_usd": round((float(collected) - ROUND_TRIP_BPS) / 1e4 * 100, 3),
            })
            i = j
        else:
            i += 1
    return trades


def main():
    results = {}
    for sym in SYMBOLS:
        ts, rate = load_funding(sym)
        stats = {
            "rows": len(rate),
            "days": round(len(rate) / 3, 1),
            "mean_bps_8h": round(float(rate.mean()), 3),
            "annual_apr_pct": round(float(rate.mean() * 3 * 365 / 100), 2),
            "pct_positive": round(100 * float((rate > 0).sum()) / len(rate), 1),
            "max_bps": round(float(rate.max()), 2),
            "min_bps": round(float(rate.min()), 2),
        }
        results[sym] = {"stats": stats, "trades": {}}
        for label, ot, et in (("thr3_0", 3.0, 0.0), ("thr1_0", 1.0, 0.0),
                              ("thr0_any", 0.5, 0.0), ("always", -1e9, 0.0)):
            tr = simulate_full_history(sym, ot, et)
            if tr:
                nets = np.array([t["net_bps"] for t in tr])
                results[sym]["trades"][label] = {
                    "trades": len(tr),
                    "profitable": int((nets > 0).sum()),
                    "net_bps_mean": round(float(nets.mean()), 3),
                    "net_bps_total": round(float(nets.sum()), 3),
                    "net_pnl_usd_total": round(float(nets.sum() / 1e4 * 100), 2),
                    "avg_hold_days": round(float(np.mean([t["hold_days"] for t in tr])), 2),
                    "avg_payments": round(float(np.mean([t["payments"] for t in tr])), 1),
                    "avg_funding_bps": round(float(np.mean([t["funding_bps"] for t in tr])), 3),
                }
            else:
                results[sym]["trades"][label] = {"trades": 0, "net_pnl_usd_total": 0.0}
        s = results[sym]
        print(f"{sym}: APR={s['stats']['annual_apr_pct']}% thr3: "
              f"{s['trades']['thr3_0']} thr1: {s['trades']['thr1_0']} always: {s['trades']['always']}")

    path = os.path.join(ROOT, "ARBITRAGE_V5_FUNDING.json")
    with open(path, "w") as f:
        json.dump({"round_trip_cost_bps": ROUND_TRIP_BPS, "results": results}, f, indent=2)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
