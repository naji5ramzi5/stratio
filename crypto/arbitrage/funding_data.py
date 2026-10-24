"""Funding rate + market data acquisition via Binance REST (fapi/api).
Research-only. Downloads:
- fundingRate history (full, ~1000 rows = ~330 days per symbol)
- premiumIndex snapshots
- futures aggTrades for cross pairs (ETHBTC, BNBBTC) via Vision
"""
import json
import os
import sys
from datetime import date, timedelta
from urllib.request import urlopen, Request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from arbitrage.vision import download_agg_trades  # noqa: E402

FAPI = "https://fapi.binance.com"
HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "data", "funding")


def _get(url: str):
    req = Request(url, headers={"User-Agent": "stratocrypto-arb-research"})
    with urlopen(req, timeout=30) as r:
        return json.load(r)


def download_funding(symbols: list, force: bool = False) -> dict:
    os.makedirs(OUT, exist_ok=True)
    result = {}
    for sym in symbols:
        local = os.path.join(OUT, f"{sym}.json")
        if os.path.exists(local) and not force:
            result[sym] = json.load(open(local))
            print(f"{sym}: cached ({len(result[sym])} rows)")
            continue
        rows = []
        start = 0
        while True:
            batch = _get(f"{FAPI}/fapi/v1/fundingRate?symbol={sym}&startTime={start}&limit=1000")
            if not batch:
                break
            rows.extend(batch)
            last = batch[-1]["fundingTime"]
            if len(batch) < 1000:
                break
            start = last + 1
        json.dump(rows, open(local, "w"))
        result[sym] = rows
        if rows:
            print(f"{sym}: {len(rows)} funding rows "
                  f"({rows[0]['fundingTime']} .. {rows[-1]['fundingTime']})")
    return result


def download_futures_cross_pairs(day_list: list):
    for sym in ("ETHBTC", "BNBBTC"):
        for d in day_list:
            download_agg_trades(sym, d, "futures_um")


if __name__ == "__main__":
    SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
    download_funding(SYMBOLS)
    days = [date(2026, 7, 25) + timedelta(days=i) for i in range(16)]
    download_futures_cross_pairs(days)
    print("done")
