"""Shared market data pipeline — single source of truth for klines/prices.

Every bot (advanced_bot, dashboard, kimi) currently fetches independently from
Binance and writes its own files, which causes:
  * redundant API calls (rate-limit pressure)
  * racing writers on shared files (prediction_accuracy.json corruption)

This module centralises the fetch behind a TTL cache with a threading lock so
concurrent readers see fresh-enough data and only one code path hits Binance.
The canonical network loader is data_loader.binance_ohlcv (session-reuse,
retry, pagination) — we wrap it, we do not duplicate it.
"""
import json
import logging
import os
import threading
import time

import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(BASE_DIR, "market_cache.json")
_LOCK = threading.Lock()

DEFAULT_TTL = {
    "price": 60,       # spot price: 1 minute
    "all_prices": 60,  # full price snapshot: 1 minute
    "klines_1h": 300,  # 1h klines: 5 minutes
    "klines_5m": 120,  # 5m klines: 2 minutes
    "klines_1d": 1800, # daily klines: 30 minutes
}


def _klines_ttl(interval):
    return DEFAULT_TTL.get(f"klines_{interval}", 300)


def _load_cache():
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, default=str)
    except Exception as e:
        logger.warning(f"[DATA-PIPELINE] cache write failed: {e}")


def _now_ts():
    return int(time.time())


def _age(entry):
    if not entry:
        return float("inf")
    return _now_ts() - entry.get("ts", 0)


def _prune(cache, max_entries=2000, max_age_sec=3600):
    if len(cache) <= max_entries:
        return cache
    cutoff = _now_ts() - max_age_sec
    return {k: v for k, v in cache.items() if v.get("ts", 0) > cutoff}


def get_klines(symbol, interval="1h", limit=500, ttl=None):
    """Cached klines. Returns list of OHLCV lists (Binance format) or []."""
    if ttl is None:
        ttl = _klines_ttl(interval)
    key = f"klines:{symbol.upper()}:{interval}:{limit}"
    with _LOCK:
        cache = _load_cache()
        entry = cache.get(key)
        if entry and _age(entry) < ttl and isinstance(entry.get("data"), list):
            return entry["data"]
    # cache miss / stale — fetch fresh via the canonical loader
    from data_loader.binance_ohlcv import fetch_klines
    data = fetch_klines(symbol, interval, limit)
    if isinstance(data, list) and data:
        with _LOCK:
            cache = _load_cache()
            cache[key] = {"ts": _now_ts(), "data": data}
            _save_cache(_prune(cache))
    return data if isinstance(data, list) else []


def get_price(symbol, ttl=None):
    """Cached spot price (float) or None."""
    if ttl is None:
        ttl = DEFAULT_TTL["price"]
    key = f"price:{symbol.upper()}"
    with _LOCK:
        cache = _load_cache()
        entry = cache.get(key)
        if entry and _age(entry) < ttl and isinstance(entry.get("data"), (int, float)):
            return float(entry["data"])
    # fetch fresh
    from data_loader.binance_ohlcv import get_session
    from settings import BINANCE_BASE
    try:
        resp = get_session().get(BINANCE_BASE + "/api/v3/ticker/price",
                                 params={"symbol": symbol.upper()}, timeout=8)
        price = None
        if resp.status_code == 200:
            price = float(resp.json().get("price", 0))
    except Exception:
        price = None
    if price and price > 0:
        with _LOCK:
            cache = _load_cache()
            cache[key] = {"ts": _now_ts(), "data": price}
            _save_cache(cache)
    return price


def get_all_prices(ttl=None):
    """Snapshot {SYMBOL: float} of every spot price (cached)."""
    if ttl is None:
        ttl = DEFAULT_TTL["all_prices"]
    key = "all_prices"
    with _LOCK:
        cache = _load_cache()
        entry = cache.get(key)
        if entry and _age(entry) < ttl and isinstance(entry.get("data"), dict):
            return entry["data"]
    from data_loader.binance_ohlcv import get_session
    from settings import BINANCE_BASE
    out = {}
    try:
        resp = get_session().get(BINANCE_BASE + "/api/v3/ticker/price", timeout=10)
        if resp.status_code == 200:
            for t in resp.json():
                try:
                    p = float(t.get("price", 0))
                except (TypeError, ValueError):
                    continue
                if p > 0 and t.get("symbol"):
                    out[t["symbol"]] = p
    except Exception:
        pass
    if out:
        with _LOCK:
            cache = _load_cache()
            cache[key] = {"ts": _now_ts(), "data": out}
            _save_cache(cache)
    return out


def get_close_array(symbol, interval="1h", limit=500):
    """Convenience: numpy array of closes for a symbol."""
    k = get_klines(symbol, interval, limit)
    if not k:
        return np.array([], dtype=float)
    return np.array([float(row[4]) for row in k], dtype=float)
