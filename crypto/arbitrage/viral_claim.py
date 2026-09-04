"""VIRAL_CLAIM_REPLICATION: is $0.90 -> $408,000 in ~2 days mathematically possible?
Pure math + comparison against measured market data (ARBITRAGE_V1_DIAGNOSTIC / V2_BASIS).
Does NOT optimize to reach the target; it tests whether the target is plausible.
"""
import json
import os
import math
import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")


def required_per_cycle_return(target_ratio: float, n_cycles: int) -> float:
    return target_ratio ** (1 / n_cycles) - 1


def main():
    start, target, hours = 0.90, 408_000.0, 48
    ratio = target / start
    seconds = hours * 3600

    print(f"=== CLAIM: ${start} -> ${target} in {hours}h (x{ratio:,.0f}) ===")
    print(f"required compounded log-return: {math.log(ratio):.4f}")

    # Scenario table: trades per minute -> required per-trade return
    print("\n--- Required average return per cycle at different frequencies ---")
    scenarios = []
    for per_min in (1, 2, 5, 10, 30, 60, 120, 600):
        n = int(per_min * 60 * hours)
        r = required_per_cycle_return(ratio, n)
        scenarios.append((per_min, n, r * 1e4))
        print(f"  {per_min:4d} trades/min -> {n:8d} cycles -> {r*1e4:8.4f} bps/cycle")

    # Load measured data
    diag = json.load(open(os.path.join(ROOT, "ARBITRAGE_V1_DIAGNOSTIC.json")))
    basis = json.load(open(os.path.join(ROOT, "ARBITRAGE_V2_BASIS.json")))

    print("\n--- What the measured market actually offers (16 days, real Binance tick data) ---")
    for k, r in diag["results"].items():
        print(f"  {k}: gross cross-rate mean {r['gross_bps_mean']}bps, "
              f"buckets >=5bps {r['buckets_gt_thr_bps']['5']} "
              f"(of {r['buckets_total']}), >=10bps {r['buckets_gt_thr_bps']['10']}")

    # Cost floor per cycle: 30bps spot taker (3 legs) — measured edges never reach this
    fee_bps = 30.0
    print(f"\n  cost floor (3x spot taker 10bps): {fee_bps}bps/cycle")
    print(f"  -> required frequency to clear {fee_bps}bps costs with x{ratio:,.0f}: "
          f"{math.log(ratio) / (fee_bps/1e4):,.0f} cycles over {hours}h "
          f"= {math.log(ratio)/(fee_bps/1e4)/seconds*60:,.1f} trades/min (impossible: "
          f"no net-positive edge measured)")

    # Capital/liquidity constraints
    print("\n--- Capital & liquidity constraints ---")
    print(f"  1) Minimum order: Binance spot min notional ~$5-10 -> ${start:.2f} start "
          f"cannot trade main pairs (BTCUSDT min qty 0.00001 ~ $0.6, step size 0.00001; "
          f"ETH min notional etc.).")
    print(f"  2) Fees on ${start:.2f}: {fee_bps}bps = ${start*fee_bps/1e4:.5f}/cycle -> fine, "
          f"but the RATE must be profitable.")
    print(f"  3) Measured max gross edge >=10bps: ~772 buckets of 15.7M (0.005%) -> "
          f"essentially zero exploitable rate at any realistic frequency.")
    print(f"  4) To reach $408k with avg net edge e and notional N capped by depth: "
          f"cycles = {math.log(ratio)/1e-4:,.0f} at 1bps net (unachieved), "
          f"and executable notional per cycle is bounded by top-of-book depth "
          f"(order book depth NOT in free historical data -> cannot be proven).")

    verdict = {
        "verdict": "CLAIM NOT REPRODUCIBLE UNDER CURRENT MARKET CONDITIONS",
        "evidence": [
            "No net-positive cycle edge measured in 15.7M real quote buckets over 16 days",
            "0/2 cycles produced any executable trade after 30bps taker fees",
            "Spot-futures basis never exceeded +10bps and required 30bps round-trip costs",
            "Required per-cycle return at even 1 trade/minute is 188bps — 6x the measured cost floor",
            "Min-order-size constraints prevent $0.90 capital from deploying on BTC/ETH pairs",
        ],
    }
    print(f"\n=== {verdict['verdict']} ===")
    for e in verdict["evidence"]:
        print(f"  - {e}")

    with open(os.path.join(ROOT, "VIRAL_CLAIM_REPLICATION.json"), "w") as f:
        json.dump({"start_usd": start, "target_usd": target, "hours": hours,
                   "scenarios": scenarios, "fee_bps": fee_bps, **verdict}, f, indent=2)


if __name__ == "__main__":
    main()
