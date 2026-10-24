"""ARBITRAGE_V3: DEV/OOS split on the triangular experiment.
DEV: 2026-07-25..2026-08-03 (10 days) — params fixed a priori (fees, latency), NOT tuned on dev.
OOS: 2026-08-04..2026-08-09 (6 days) — untouched during research.
Honest protocol: report both; no parameter selection based on OOS.
"""
import os
import sys
import json
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from arbitrage.storage import load_aggtrades  # noqa: E402
from arbitrage.execution import FeeConfig, ExecConfig, FeeEngine  # noqa: E402
from arbitrage.triangular import run_triangular_v1  # noqa: E402


def days_between(start: date, end: date) -> list:
    return [(start + timedelta(days=i)).isoformat()
            for i in range((end - start).days + 1)]


def run_split(cycle: tuple, day_list: list, fee_net: float, exec_cfg: ExecConfig):
    a, b = cycle
    syms = [f"{a}USDT", f"{b}USDT", f"{b}{a}"]
    trades = 0
    pnl = 0.0
    cand = 0
    for day in day_list:
        try:
            t = {s: load_aggtrades(s, day, "spot") for s in syms}
        except FileNotFoundError:
            continue
        res = run_triangular_v1(t, a, b, fee_net, exec_cfg)
        trades += res["combined"]["trades"]
        pnl += res["combined"]["net_pnl_usd"]
        cand += res["candidates_total"]
    return {"trades": trades, "net_pnl_usd": round(pnl, 2), "candidates": cand}


def main():
    fee_net = FeeEngine(FeeConfig(taker_bps=10.0)).spot_cycle_cost(3)
    dev = days_between(date(2026, 7, 25), date(2026, 8, 3))
    oos = days_between(date(2026, 8, 4), date(2026, 8, 9))
    out = {"protocol": "DEV 07-25..08-03 (10d) | OOS 08-04..08-09 (6d) | params FIXED a priori",
           "fee_net_bps": fee_net * 1e4, "results": {}}
    for cycle in (("BTC", "ETH"), ("BTC", "BNB")):
        for lat in (0, 50, 100, 250, 500):
            cfg = ExecConfig(latency_ms=lat)
            d = run_split(cycle, dev, fee_net, cfg)
            o = run_split(cycle, oos, fee_net, cfg)
            out["results"][f"{cycle[0]}_{cycle[1]}_lat{lat}"] = {"dev": d, "oos": o}
            print(f"{cycle[0]}/{cycle[1]} lat={lat:4d}ms  "
                  f"DEV: trades={d['trades']:3d} pnl=${d['net_pnl_usd']:8.2f} cand={d['candidates']:4d}  "
                  f"OOS: trades={o['trades']:3d} pnl=${o['net_pnl_usd']:8.2f} cand={o['candidates']:4d}")
    path = os.path.join(os.path.dirname(__file__), "..", "ARBITRAGE_V3_DEV_OOS.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
