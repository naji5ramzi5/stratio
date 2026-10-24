"""Data collection for QUANT-X.

Sources (all free, no API key):
  * Spot klines          — Binance public API (via data_loader.binance_ohlcv)
  * Funding rate         — Binance USDT-M futures (fapi/v1/fundingRate)
  * Open interest        — Binance USDT-M futures (fapi/v1/openInterest)
  * Long/Short ratio     — Binance futures data endpoint
  * Fear & Greed index   — alternative.me

Missing by design (paid / not available here) is reported per-step by the
report layer, never fabricated: on-chain (MVRV/NUPL/SOPR/flows/whales/ETF),
macro (DXY/yields/rates), options (gamma/max pain), liquidation heatmap.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import requests

from data_loader.binance_ohlcv import fetch_klines

logger = logging.getLogger(__name__)

FAPI = "https://fapi.binance.com"
FUT_DATA = "https://fapi.binance.com/futures/data"
FNG_URL = "https://api.alternative.me/fng/"

TIMEOUT = 10


def _get(url, **params):
    try:
        r = requests.get(url, params=params or None, timeout=TIMEOUT)
        if r.status_code != 200:
            logger.warning("HTTP %s for %s", r.status_code, url)
            return None
        return r.json()
    except Exception as exc:
        logger.warning("fetch failed %s: %s", url, exc)
        return None


def fetch_funding_rate(symbol, limit=30):
    """Latest funding rate history (USDT-M perpetual)."""
    data = _get(f"{FAPI}/fapi/v1/fundingRate", symbol=symbol, limit=limit)
    if not data:
        return None
    return [{
        "time": datetime.fromtimestamp(r["fundingTime"] / 1000, tz=timezone.utc).isoformat(),
        "rate_pct": float(r["fundingRate"]) * 100.0,
    } for r in data]


def fetch_open_interest(symbol):
    data = _get(f"{FAPI}/fapi/v1/openInterest", symbol=symbol)
    if not data:
        return None
    return {"oi": float(data.get("openInterest", 0.0)), "time": data.get("time")}


def fetch_long_short_ratio(symbol, limit=5):
    data = _get(f"{FUT_DATA}/globalLongShortAccountRatio",
                symbol=symbol, period="1h", limit=limit)
    if not data:
        return None
    return [{
        "time": datetime.fromtimestamp(r["timestamp"] / 1000, tz=timezone.utc).isoformat(),
        "long_account": float(r["longAccount"]),
        "short_account": float(r["shortAccount"]),
        "ratio": float(r["longShortRatio"]),
    } for r in data]


def fetch_fear_greed():
    data = _get(FNG_URL, limit=1)
    if not data or not data.get("data"):
        return None
    d = data["data"][0]
    try:
        return {"value": int(d.get("value", -1)), "classification": d.get("value_classification", "unknown")}
    except (TypeError, ValueError):
        return None


def collect_market_data(symbol, interval="1h", lookback=200):
    """Collect the available data bundle for ``symbol``.

    Returns a dict with per-source records plus a ``status`` field so the
    report layer can state exactly what is missing.
    """
    bundle = {"symbol": symbol.upper(), "interval": interval, "lookback": lookback}

    klines = fetch_klines(symbol, interval, limit=lookback)
    bundle["klines"] = klines
    bundle["klines_status"] = "available" if klines else "missing"

    if klines:
        from ml_trainer import KLINES_COLUMNS  # local import keeps module light
        bundle["columns"] = KLINES_COLUMNS

    # Futures & sentiment (independent of spot klines)
    bundle["funding_rate"] = fetch_funding_rate(symbol, limit=30)
    bundle["funding_status"] = "available" if bundle["funding_rate"] else "missing"
    bundle["open_interest"] = fetch_open_interest(symbol)
    bundle["oi_status"] = "available" if bundle["open_interest"] else "missing"
    bundle["long_short"] = fetch_long_short_ratio(symbol, limit=5)
    bundle["ls_status"] = "available" if bundle["long_short"] else "missing"
    bundle["fear_greed"] = fetch_fear_greed()
    bundle["fng_status"] = "available" if bundle["fear_greed"] else "missing"

    # Collected at analysis time (UTC)
    bundle["as_of_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return bundle
