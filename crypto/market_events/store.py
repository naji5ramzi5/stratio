"""Derived event store (trades).

Architecture decision (documented in PHASE1_DATA_MODEL.md): the store is a
**derived, validated cache** — never an authoritative store, never a
rewriter. Raw Vision zips and recorder JSONL remain the source of truth.

Coverage:
- TRADE streams: per (market, symbol, day) compact .npz cache + manifest.json
- ORDER_BOOK data: kept in recorder JSONL (already sequential, sorted,
  partitioned by day, timestamped) — read directly by the adapter; caching
  them adds nothing because the JSONL IS the cache.

Properties:
- sorted by event timestamp within a day; int64 µs timestamps
- atomic writes (tmp + replace); manifest records source zip sha256
- deterministic: identical zip → identical cache (rows, first/last ts,
  content checksum)
- range queries via searchsorted (no re-filtering from scratch per call —
  the exact inefficiency the old OHLCV layer had)
- reads never touch the source zip after ingestion
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from typing import Dict, Iterator, Optional, Tuple

import numpy as np

from .adapters import aggtrade_rows_from_zip
from .model import MarketEvent, make_trade
from .replay import ReplayEngine

DEFAULT_ROOT = os.path.join(os.path.dirname(__file__), "cache")

MARKETS = ("spot", "futures_um")
MARKET_EXCHANGE = {"spot": "binance_spot", "futures_um": "binance_futures_um"}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class EventStore:
    def __init__(self, root: str = DEFAULT_ROOT):
        self.root = root

    # -- paths --------------------------------------------------------------

    def _cache_path(self, market: str, symbol: str, day: str) -> str:
        return os.path.join(
            self.root, market, symbol, f"{day}.npz"
        )

    def _manifest_path(self, market: str, symbol: str, day: str) -> str:
        return os.path.join(
            self.root, market, symbol, f"{day}.manifest.json"
        )

    # -- ingestion ----------------------------------------------------------

    def ingest_trade_day(
        self,
        source_zip: str,
        market: str,
        symbol: str,
        day: str,
        receive_timestamp: Optional[int] = None,
        force: bool = False,
    ) -> dict:
        """Build the derived cache for one day of aggTrades.

        Returns the manifest entry. Idempotent: skips if manifest exists and
        checksum matches the source, unless force=True.
        """
        if market not in MARKETS:
            raise ValueError(f"unknown market {market!r}")
        cache_path = self._cache_path(market, symbol, day)
        manifest_path = self._manifest_path(market, symbol, day)
        checksum = _sha256_file(source_zip)

        if os.path.exists(manifest_path) and not force:
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("source_checksum") == checksum and os.path.exists(cache_path):
                return existing

        ts = []
        seq = []
        price = []
        qty = []
        bm = []
        exchange = MARKET_EXCHANGE[market]
        for row, has_header in aggtrade_rows_from_zip(
            source_zip, exchange, symbol, receive_timestamp
        ):
            if has_header:
                seq.append(int(row[0]))
                price.append(float(row[1]))
                qty.append(float(row[2]))
                ts.append(int(row[5]) * 1000)
                bm.append(row[6].strip().lower() in ("true", "1"))
            else:
                seq.append(int(row[0]))
                price.append(float(row[1]))
                qty.append(float(row[2]))
                ts.append(int(row[5]))
                bm.append(row[6].strip().lower() in ("true", "1"))

        order = np.argsort(ts, kind="stable")
        ts_a = np.asarray(ts, dtype=np.int64)[order]
        seq_a = np.asarray(seq, dtype=np.int64)[order]
        price_a = np.asarray(price, dtype=np.float64)[order]
        qty_a = np.asarray(qty, dtype=np.float64)[order]
        bm_a = np.asarray(bm, dtype=bool)[order]

        if len(ts_a) == 0:
            raise ValueError(f"no rows in {source_zip}")

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(cache_path), suffix=".npz")
        os.close(fd)
        try:
            np.savez_compressed(
                tmp,
                ts_us=ts_a,
                seq=seq_a,
                price=price_a,
                qty=qty_a,
                buyer_maker=bm_a,
            )
            shutil.move(tmp, cache_path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

        content_hash = hashlib.sha256(
            ts_a.tobytes() + price_a.tobytes()
        ).hexdigest()
        manifest = {
            "market": market,
            "symbol": symbol,
            "day": day,
            "source": os.path.basename(source_zip),
            "source_checksum": checksum,
            "content_checksum": content_hash,
            "rows": int(len(ts_a)),
            "first_ts_us": int(ts_a[0]),
            "last_ts_us": int(ts_a[-1]),
            "receive_ts_us": receive_timestamp,
            "created_at": __import__("time").time(),
        }
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(manifest_path), suffix=".json")
        os.close(fd)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        shutil.move(tmp, manifest_path)
        return manifest

    # -- reads --------------------------------------------------------------

    def has_day(self, market: str, symbol: str, day: str) -> bool:
        return os.path.exists(self._cache_path(market, symbol, day)) and os.path.exists(
            self._manifest_path(market, symbol, day)
        )

    def manifest(self, market: str, symbol: str, day: str) -> Optional[dict]:
        p = self._manifest_path(market, symbol, day)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_days(self, market: str, symbol: str) -> list:
        d = os.path.join(self.root, market, symbol)
        if not os.path.exists(d):
            return []
        return sorted(
            f[: -len(".npz")]
            for f in os.listdir(d)
            if f.endswith(".npz") and os.path.exists(
                os.path.join(d, f[: -len(".npz")] + ".manifest.json")
            )
        )

    def load_day(self, market: str, symbol: str, day: str) -> dict:
        """Load the full cached day as dict of numpy arrays."""
        if not self.has_day(market, symbol, day):
            raise FileNotFoundError(
                f"no cached day {market}/{symbol}/{day} — ingest first"
            )
        with np.load(self._cache_path(market, symbol, day)) as data:
            out = {
                "ts_us": data["ts_us"].copy(),
                "seq": data["seq"].copy(),
                "price": data["price"].copy(),
                "qty": data["qty"].copy(),
                "buyer_maker": data["buyer_maker"].copy(),
            }
        return out

    def range_query(
        self,
        market: str,
        symbol: str,
        day: str,
        start_us: Optional[int] = None,
        end_us: Optional[int] = None,
    ) -> dict:
        """Slice the day between [start_us, end_us) via searchsorted.

        Returns a dict of arrays; never re-reads the source zip.
        """
        data = self.load_day(market, symbol, day)
        ts = data["ts_us"]
        lo = 0 if start_us is None else int(np.searchsorted(ts, start_us, side="left"))
        hi = len(ts) if end_us is None else int(
            np.searchsorted(ts, end_us, side="left")
        )
        return {k: v[lo:hi] for k, v in data.items()}

    def replay(
        self,
        market: str,
        symbol: str,
        day: str,
        receive_timestamp: Optional[int] = None,
        build_book: bool = False,
    ) -> ReplayEngine:
        """Replay engine over the cached day (sorted by construction)."""
        data = self.load_day(market, symbol, day)
        exchange = MARKET_EXCHANGE[market]
        ts = data["ts_us"]
        seq = data["seq"]
        price = data["price"]
        qty = data["qty"]
        bm = data["buyer_maker"]

        def gen() -> Iterator[MarketEvent]:
            for i in range(len(ts)):
                yield make_trade(
                    exchange=exchange,
                    symbol=symbol,
                    event_timestamp=int(ts[i]),
                    receive_timestamp=receive_timestamp,
                    sequence_number=int(seq[i]),
                    trade_id=int(seq[i]),
                    price=float(price[i]),
                    quantity=float(qty[i]),
                    is_buyer_maker=bool(bm[i]),
                    source=f"event_store:{market}:{symbol}:{day}",
                )

        return ReplayEngine(gen(), build_book=build_book)

    def summary(self) -> dict:
        out = {}
        for market in MARKETS:
            d = os.path.join(self.root, market)
            if not os.path.exists(d):
                continue
            for symbol in sorted(os.listdir(d)):
                sym_dir = os.path.join(d, symbol)
                if not os.path.isdir(sym_dir):
                    continue
                out[f"{market}/{symbol}"] = self.list_days(market, symbol)
        return out
