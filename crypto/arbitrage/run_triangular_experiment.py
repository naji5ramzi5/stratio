"""ARBITRAGE_V1: triangular arbitrage on real Binance tick data.
Cycles: USDT->BTC->ETH->USDT (and reverse) ; USDT->BTC->BNB->USDT (and reverse)
16 days of aggTrades (2026-07-25 -> 2026-08-09). Point-in-time safe.
Base case: taker fees 10bps/leg, latency 50ms, notional $100/cycle, fill prob 1.0.
"""
import os
import sys
import json
from datetime import date, timedelta
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from arbitrage.storage import load_aggtrades, list_days  # noqa: E402
from arbitrage.execution import FeeConfig, ExecConfig, FeeEngine  # noqa: E402
from arbitrage.triangular import run_triangular_v1  # noqa: E402

START = date(2026, 7, 25)
END = date(2026, 8, 9)
DAYS = [(START + timedelta(days=i)).isoformat() for i in range((END - START).days + 1)]

FEE_CFG = FeeConfig(taker_bps=10.0)  # Binance spot standard taker


def run_cycle(cycle_symbols: dict, a: str, b: str, exec_cfg: ExecConfig, fee_net: float,
              days: list, verbose=False) -> dict:
    per_day = []
    for day in days:
        trades = {}
        for sym in cycle_symbols:
            try:
                trades[sym] = load_aggtrades(sym, day, "spot")
            except FileNotFoundError:
                trades[sym] = None
        if any(t is None for t in trades.values()):
            continue
        res = run_triangular_v1(trades, a, b, fee_net, exec_cfg, verbose)
        res["day"] = day
        per_day.append(res)
        if verbose:
            c = res["combined"]
            print(f"  {day}: trades={c['trades']:3d} net_pnl=${c['net_pnl_usd']:8.2f} "
                  f"cand={res['candidates_total']}")
    return per_day


def main():
    fee_net = FeeEngine(FEE_CFG).spot_cycle_cost(3)
    results = {}
    for name, a, b in (("BTC_ETH", "BTC", "ETH"), ("BTC_BNB", "BTC", "BNB")):
        symbols = [f"{a}USDT", f"{b}USDT", f"{b}{a}"]
        for lat in (0, 50, 100, 250, 500):
            exec_cfg = ExecConfig(latency_ms=lat)
            per_day = run_cycle(symbols, a, b, exec_cfg, fee_net, DAYS, verbose=(lat == 0))
            agg = {
                "trades": sum(d["combined"]["trades"] for d in per_day),
                "candidates": sum(d["candidates_total"] for d in per_day),
                "net_pnl_usd": round(sum(d["combined"]["net_pnl_usd"] for d in per_day), 2),
                "net_bps": 0.0,
            }
            n = agg["trades"]
            if n:
                tot_bps = sum(d["combined"]["net_bps"] * d["combined"]["trades"] for d in per_day)
                agg["net_bps"] = round(tot_bps / n, 3)
            results[f"{name}_lat{lat}"] = agg

    out = {"fee_net_bps": fee_net * 1e4, "days": DAYS, "notional_usd": 100.0, "results": results}
    path = os.path.join(os.path.dirname(__file__), "..", "ARBITRAGE_V1_TRIANGULAR.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== SUMMARY (fee=%.1fbps/cycle, $100 notional per trade) ===" % (fee_net * 1e4))
    for k, v in results.items():
        print(f"{k:14s} trades={v['trades']:5d} cand={v['candidates']:5d} "
              f"net_bps={v['net_bps']:8.3f} net_pnl=${v['net_pnl_usd']:10.2f}")
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
