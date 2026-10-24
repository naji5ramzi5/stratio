"""Binance Vision data downloader for the arbitrage research subsystem.
Downloads aggTrades, 1s/1m klines, USD-M bookTicker and fundingRate from
https://data.binance.vision (free S3 bucket). Verifies sha256 checksums.
Research-only. No live trading.
"""
import io
import json
import os
import shutil
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from urllib.request import urlopen, Request

BASE = "https://data.binance.vision/data"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data")


def _fetch(url: str, timeout: int = 120) -> bytes:
    req = Request(url, headers={"User-Agent": "stratocrypto-arb-research"})
    with urlopen(req, timeout=timeout) as r:
        return r.read()


def _checksum_url(url: str) -> str:
    return url + ".CHECKSUM"


def verify_checksum(url: str, data: bytes) -> bool:
    import hashlib
    try:
        csum = _fetch(_checksum_url(url), timeout=30).decode().strip().split()[0]
        return hashlib.sha256(data).hexdigest() == csum
    except Exception:
        return False


def _download_zip(url: str, local: str, retries: int = 3) -> bool:
    """Download zip to local path, verify checksum, return True if new/success."""
    if os.path.exists(local):
        return False
    os.makedirs(os.path.dirname(local), exist_ok=True)
    for attempt in range(retries):
        try:
            data = _fetch(url, timeout=120)
            if not verify_checksum(url, data):
                return False
            with open(local, "wb") as f:
                f.write(data)
            return True
        except Exception as e:
            if attempt == retries - 1:
                print(f"  !! {url.split('/')[-1]} failed after {retries} tries: {type(e).__name__}")
                return False
            time.sleep(2 * (attempt + 1))
    return False


def _read_zip_csv(local: str):
    with zipfile.ZipFile(local) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            return io.TextIOWrapper(f, encoding="utf-8")


def _days(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def download_agg_trades(symbol: str, day: date, market: str = "spot",
                        force: bool = False) -> str:
    """Daily aggTrades file. Columns: agg_id, price, qty, first_id, last_id, ts, buyer_maker, best_match"""
    if market == "spot":
        url = f"{BASE}/spot/daily/aggTrades/{symbol}/{symbol}-aggTrades-{day.isoformat()}.zip"
    elif market == "futures_um":
        url = f"{BASE}/futures/um/daily/aggTrades/{symbol}/{symbol}-aggTrades-{day.isoformat()}.zip"
    else:
        raise ValueError(market)
    local = os.path.join(CACHE_DIR, market, "aggTrades", symbol, f"{day.isoformat()}.zip")
    if force and os.path.exists(local):
        os.remove(local)
    if _download_zip(url, local):
        print(f"  downloaded {symbol} aggTrades {day.isoformat()}")
    return local


def download_klines(symbol: str, interval: str, day: date, market: str = "spot",
                    force: bool = False) -> str:
    if market == "spot":
        url = f"{BASE}/spot/daily/klines/{symbol}/{interval}/{symbol}-{interval}-klines-{day.isoformat()}.zip"
    elif market == "futures_um":
        url = f"{BASE}/futures/um/daily/klines/{symbol}/{interval}/{symbol}-{interval}-klines-{day.isoformat()}.zip"
    else:
        raise ValueError(market)
    local = os.path.join(CACHE_DIR, market, "klines", symbol, interval, f"{day.isoformat()}.zip")
    if force and os.path.exists(local):
        os.remove(local)
    if _download_zip(url, local):
        print(f"  downloaded {symbol} {interval} {day.isoformat()}")
    return local


def download_book_ticker(symbol: str, day: date, market: str = "futures_um",
                         force: bool = False) -> str:
    url = f"{BASE}/futures/um/daily/bookTicker/{symbol}/{symbol}-bookTicker-{day.isoformat()}.zip"
    local = os.path.join(CACHE_DIR, market, "bookTicker", symbol, f"{day.isoformat()}.zip")
    if force and os.path.exists(local):
        os.remove(local)
    if _download_zip(url, local):
        print(f"  downloaded {symbol} bookTicker {day.isoformat()}")
    return local


def download_funding(symbol: str, month: str, market: str = "futures_um",
                     force: bool = False) -> str:
    url = f"{BASE}/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{month}.zip"
    local = os.path.join(CACHE_DIR, market, "fundingRate", symbol, f"{month}.zip")
    if force and os.path.exists(local):
        os.remove(local)
    if _download_zip(url, local):
        print(f"  downloaded {symbol} fundingRate {month}")
    return local


def probe_available(symbol: str = "BTCUSDT", day: date | None = None) -> dict:
    """Verify which datasets exist for a given day (1 = reachable, 0 = missing)."""
    day = day or date.today() - timedelta(days=2)
    results = {}
    for market in ("spot", "futures_um"):
        for kind, iv in (("aggTrades", None), ("klines/1s", "1s"), ("klines/1m", "1m"), ("bookTicker", None)):
            if market == "spot" and kind.startswith("bookTicker"):
                continue
            if kind.startswith("klines"):
                url = f"{BASE}/{market}/daily/{kind}/{symbol}/{symbol}-{iv}-klines-{day.isoformat()}.zip"
            elif kind == "bookTicker":
                url = f"{BASE}/{market}/daily/bookTicker/{symbol}/{symbol}-bookTicker-{day.isoformat()}.zip"
            else:
                url = f"{BASE}/{market}/daily/aggTrades/{symbol}/{symbol}-aggTrades-{day.isoformat()}.zip"
            try:
                _fetch(url, timeout=20)
                results[f"{market}/{kind}"] = 1
            except Exception:
                results[f"{market}/{kind}"] = 0
    return results


if __name__ == "__main__":
    print(probe_available())
