"""
Multi-Exchange Data Aggregator v1.0
يجمع بيانات من Binance + Kraken + Bybit ويحسب المتوسط
"""
import requests
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


def _fetch_binance(symbol, interval, limit):
    try:
        resp = requests.get(
            f"https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=8
        )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                return [{"exchange": "binance", "close": float(k[4]), "volume": float(k[5]),
                         "high": float(k[2]), "low": float(k[3]), "open": float(k[1]),
                         "time": k[0]} for k in data]
    except Exception as e:
        logger.debug(f"[MULTI] Binance {symbol}: {e}")
    return []


def _fetch_kraken(pair, interval, limit):
    try:
        interval_map = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
        kraken_interval = interval_map.get(interval, 60)
        resp = requests.get(
            f"https://api.kraken.com/0/public/OHLC",
            params={"pair": pair, "interval": kraken_interval},
            timeout=8
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("error") == [] and data.get("result"):
                for key in data["result"]:
                    if key != "last":
                        ohlc = data["result"][key][-limit:]
                        return [{"exchange": "kraken", "close": float(k[4]), "volume": float(k[6]),
                                 "high": float(k[2]), "low": float(k[3]), "open": float(k[1]),
                                 "time": k[0]} for k in ohlc]
    except Exception as e:
        logger.debug(f"[MULTI] Kraken {pair}: {e}")
    return []


def _fetch_bybit(symbol, interval, limit):
    try:
        resp = requests.get(
            f"https://api.bybit.com/v5/market/kline",
            params={"category": "spot", "symbol": symbol, "interval": interval, "limit": limit},
            timeout=8
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("retCode") == 0 and data.get("result"):
                klines = data["result"]["list"][:limit]
                return [{"exchange": "bybit", "close": float(k[4]), "volume": float(k[5]),
                         "high": float(k[2]), "low": float(k[3]), "open": float(k[1]),
                         "time": int(k[0])} for k in klines]
    except Exception as e:
        logger.debug(f"[MULTI] Bybit {symbol}: {e}")
    return []


def _convert_symbol(symbol, exchange):
    if exchange == "kraken":
        if symbol.endswith("USDT"):
            return symbol.replace("USDT", "USDT")
        if symbol.endswith("BTC"):
            return symbol.replace("BTC", "XBT")
        return symbol
    return symbol


def get_aggregated_klines(symbol, interval="1h", limit=100):
    binance = _fetch_binance(symbol, interval, limit)
    kraken = _fetch_kraken(_convert_symbol(symbol, "kraken"), interval, limit)
    bybit = _fetch_bybit(symbol, interval, limit)

    sources = []
    if binance:
        sources.append(("binance", binance))
    if kraken:
        sources.append(("kraken", kraken))
    if bybit:
        sources.append(("bybit", bybit))

    if not sources:
        return []
    if len(sources) < 2:
        return sources[0][1]

    min_len = min(len(k) for _, k in sources)
    if min_len < 10:
        return sources[0][1]

    result = []
    for i in range(min_len):
        closes = [k[i]["close"] for _, k in sources]
        highs = [k[i]["high"] for _, k in sources]
        lows = [k[i]["low"] for _, k in sources]
        result.append({
            "exchange": "aggregated",
            "close": round(float(np.mean(closes)), 8),
            "high": round(float(np.max(highs)), 8),
            "low": round(float(np.min(lows)), 8),
            "volume": round(sum(float(k[i].get("volume", 0)) for _, k in sources), 2),
            "time": sources[0][1][i].get("time", 0) if isinstance(sources[0][1][i], dict) else 0,
        })
    return result


def get_exchange_price(symbol):
    prices = {}
    try:
        b = _fetch_binance(symbol, "1m", 1)
        if b: prices["binance"] = b[0]["close"]
    except Exception:
        pass
    try:
        k = _fetch_kraken(_convert_symbol(symbol, "kraken"), 1, 1)
        if k: prices["kraken"] = k[0]["close"]
    except Exception:
        pass
    try:
        by = _fetch_bybit(symbol, "1", 1)
        if by: prices["bybit"] = by[0]["close"]
    except Exception:
        pass

    if not prices:
        return {"price": 0, "exchanges": {}}

    avg = sum(prices.values()) / len(prices)
    spreads = {ex: round(abs(p - avg) / avg * 10000, 1) for ex, p in prices.items()}

    return {
        "price": round(avg, 8),
        "exchanges": {k: round(v, 8) for k, v in prices.items()},
        "spread_bps": spreads,
        "num_exchanges": len(prices),
    }