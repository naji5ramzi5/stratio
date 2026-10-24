"""
Canonical Binance OHLCV loader used by the ML/training/evaluation stack.

Consolidates the duplicated klines fetchers (advanced_bot, dashboard,
train, multi_exchange) into one paginated, session-reusing, retrying
implementation so every consumer gets identical raw data semantics.

Callers keep their own thin wrappers for compatibility, but the actual
network logic lives here only.
"""
from __future__ import annotations

import logging
import time

import requests

from settings import BINANCE_BASE

logger = logging.getLogger(__name__)

_session = None
_MAX_KLINE_LIMIT = 1000
_RETRIES = 3
_BACKOFF_SECONDS = 0.4

KLINES_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_vol", "trades", "taker_buy_vol",
    "taker_buy_quote", "ignore",
]


def get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": "stratocrypto/2.1"})
    return _session


def _request_json(endpoint: str, params: dict):
    last_exc = None
    for attempt in range(_RETRIES):
        try:
            resp = get_session().get(BINANCE_BASE + endpoint,
                                     params=params, timeout=10)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 418):
                # rate limited: back off then retry
                time.sleep(_BACKOFF_SECONDS * (attempt + 1) * 2)
                continue
            return None
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(_BACKOFF_SECONDS * (attempt + 1))
    if last_exc is not None:
        logger.debug(f"GET {endpoint} failed after {_RETRIES} tries: {last_exc}")
    return None


def fetch_klines(symbol: str, interval: str = "1h", limit: int = 500):
    """Fetch klines for ``symbol`` (oldest -> newest), paginating backwards.

    Binance caps a single klines request at 1000 bars; larger limits are
    fetched in reverse-time order and recombined oldest-first.
    """
    symbol = symbol.upper()
    limit = max(1, int(limit))
    if limit <= _MAX_KLINE_LIMIT:
        data = _request_json("/api/v3/klines",
                             {"symbol": symbol, "interval": interval, "limit": limit})
        return data if isinstance(data, list) else []
    out: list = []
    end_time = None
    while len(out) < limit:
        params = {"symbol": symbol, "interval": interval, "limit": _MAX_KLINE_LIMIT}
        if end_time is not None:
            params["endTime"] = end_time
        data = _request_json("/api/v3/klines", params)
        if not isinstance(data, list) or not data:
            break
        out = data + out
        if len(data) < _MAX_KLINE_LIMIT:
            break
        end_time = int(data[0][0]) - 1
    return out[:limit]


def fetch_close(symbol: str, interval: str = "1h", limit: int = 500):
    """Fetch only the close series (float list, oldest -> newest)."""
    raw = fetch_klines(symbol, interval, limit)
    if not raw:
        return []
    return [float(r[4]) for r in raw if len(r) > 4]
