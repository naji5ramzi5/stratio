"""Transaction-cost model (Phase 2, STEP 11).

Uses the project's documented cost assumptions (arbitrage/execution.py):
- futures taker fee: 5 bps per leg/side
- spot taker fee: 10 bps per side (used only if a spot venue is studied)
- spread: measured per symbol-day from the trade-tape bid/ask proxy
  (arbitrage/replay.build_quote_series — PIT-safe), mean (ask-bid)/mid
- slippage: stress assumption 1.0 bps per side (not measured; documented
  as an assumption; no hidden tuning)

Round-trip directional trade: enter at T, exit at T+H
    round_trip_bps = 2 * (taker_bps + slippage_bps) + spread_bps_proxy
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostConfig:
    taker_bps: float = 5.0      # futures per side (project FeeConfig)
    slippage_bps: float = 1.0   # stress assumption per side
    spread_bps: float = 0.05    # proxy default; replaced by measured value

    def round_trip_bps(self) -> float:
        return 2.0 * (self.taker_bps + self.slippage_bps) + self.spread_bps


def measure_spread_bps(ts_us, price, buyer_maker, bucket_us=100_000) -> float:
    """Mean (ask-bid)/mid from the trade-tape proxy, in bps.

    Reuses the PIT-safe proxy from the arbitrage layer; NaN-guarded.
    Returns 0.0 if no spread can be measured (no trades of one side).
    """
    import sys, os

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from arbitrage.replay import build_quote_series

    q = build_quote_series(
        {"ts_us": np.asarray(ts_us), "price": np.asarray(price),
         "buyer_maker": np.asarray(buyer_maker)}, bucket_us
    )
    mid = (q["bid"] + q["ask"]) / 2.0
    spread = (q["ask"] - q["bid"]) / mid
    valid = np.isfinite(spread) & (spread > 0)
    if not valid.any():
        return 0.0
    return float(np.mean(spread[valid]) * 1e4)


def build_cost_config(symbol_day_arrays: dict) -> CostConfig:
    """CostConfig with the MEASURED spread for one symbol-day."""
    spread = measure_spread_bps(
        symbol_day_arrays["ts_us"], symbol_day_arrays["price"],
        symbol_day_arrays["buyer_maker"]
    )
    return CostConfig(spread_bps=spread)
