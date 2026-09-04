"""Bulk-download aggTrades data for the arbitrage research phase.
Spot: BTCUSDT, ETHUSDT, ETHBTC, BNBUSDT, BNBBTC  (triangular cycles)
Futures UM: BTCUSDT, ETHUSDT                        (spot/futures basis)
Period: 2026-07-25 -> 2026-08-09
"""
import sys
import os
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from arbitrage.vision import download_agg_trades  # noqa: E402

START = date(2026, 7, 25)
END = date(2026, 8, 9)

SPOT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "ETHBTC", "BNBUSDT", "BNBBTC"]
FUT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]

def run():
    total = 0
    d = START
    while d <= END:
        for s in SPOT_SYMBOLS:
            local = download_agg_trades(s, d, "spot")
            total += os.path.getsize(local) if os.path.exists(local) else 0
        for s in FUT_SYMBOLS:
            local = download_agg_trades(s, d, "futures_um")
            total += os.path.getsize(local) if os.path.exists(local) else 0
        print(f"  day {d.isoformat()} done, cumulative {round(total/1e6,1)} MB")
        d += timedelta(days=1)
    print(f"TOTAL: {round(total/1e6,1)} MB")

if __name__ == "__main__":
    run()
