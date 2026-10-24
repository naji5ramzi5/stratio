"""ARBITRAGE_V4E: DEV/OOS split for the maker triangular strategy.
DEV: 07-25..08-03 | OOS: 08-04..08-09 (untouched). Pre-registered params only.
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

SYMBOLS = ["BTCUSDT", "ETHUSDT", "ETHBTC"]
MAKER_FEE_LEG_BPS = 2.0
TAKER_FEE_LEG_BPS = 5.0
NOTIONAL = 100.0
BUCKET_US = 100_000


def load_days(day_list):
    series = {s: [] for s in SYMBOLS}
    for day in day_list:
        for s in SYMBOLS:
            try:
                series[s].append(load_aggtrades(s, day, "futures_um"))
            except FileNotFoundError:
                pass
    return series


def simulate(series, thr_bps, win, p_leg, rng):
    fee_net = MAKER_FEE_LEG_BPS * 3 / 1e4
    unwound_cost = TAKER_FEE_LEG_BPS + MAKER_FEE_LEG_BPS + 2.0
    completed = failures = 0
    pnl = 0.0
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
            for s_idx, e_idx in zip(starts, ends):
                win_end = min(s_idx + win, n)
                filled = 0
                b = s_idx
                for b in range(s_idx + 1, win_end):
                    if net[b] * 1e4 < thr_bps:
                        break
                    for _ in range(3 - filled):
                        if rng.random() <= p_leg:
                            filled += 1
                    if filled == 3:
                        break
                if filled == 3:
                    completed += 1
                    pnl += net[b] * NOTIONAL
                elif filled > 0:
                    failures += 1
                    pnl -= unwound_cost / 1e4 * NOTIONAL * (filled / 3)
    return {"completed": completed, "failures": failures,
            "net_pnl_usd": round(pnl, 2)}


def main():
    dev = [(date(2026, 7, 25) + timedelta(days=i)).isoformat() for i in range(10)]
    oos = [(date(2026, 8, 4) + timedelta(days=i)).isoformat() for i in range(6)]
    s_dev, s_oos = load_days(dev), load_days(oos)
    rng = np.random.default_rng(11)
    out = {"protocol": "DEV 07-25..08-03 | OOS 08-04..08-09 | pre-registered params", "results": {}}
    for thr in (5.0, 8.0, 10.0):
        for win in (5, 10):
            for p in (0.5, 0.2):
                key = f"thr{thr}_win{win*100}ms_p{p}"
                d = simulate(s_dev, thr, win, p, rng)
                o = simulate(s_oos, thr, win, p, rng)
                out["results"][key] = {"dev": d, "oos": o}
                print(f"{key:22s} DEV: done={d['completed']:4d} pnl=${d['net_pnl_usd']:7.2f}  "
                      f"OOS: done={o['completed']:4d} pnl=${o['net_pnl_usd']:7.2f}")
    path = os.path.join(os.path.dirname(__file__), "..", "ARBITRAGE_V4E_MAKER_DEVOOS.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
