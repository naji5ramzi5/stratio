"""Binance adapter: raw exchange rows → canonical MarketEvent.

Row formats handled (exact schemas from the Phase 1 audit):

1. aggTrades spot (Vision, no header):
   agg_id, price, qty, first_id, last_id, ts_us, buyer_maker, best_match
   → TradeEvent (event_timestamp = ts_us)

2. aggTrades futures (Vision, header):
   agg_trade_id, price, quantity, first_trade_id, last_trade_id,
   transact_time(ms), is_buyer_maker
   → TradeEvent (event_timestamp = ms × 1000)

3. Recorder depth JSONL:
   {"t": ms, "U": first_seq, "u": final_seq,
    "b": [[p,q],...], "a": [[p,q],...]}
   → ORDER_BOOK_SNAPSHOT (partial top-N book; sequence = U/u)

4. Futures bookTicker (fstream):
   {"u": update_id, "s": "BTCUSDT", "b": "best_bid", "B": size,
    "a": "best_ask", "A": size, "T": ms}
   → QUOTE

receive_timestamp is NEVER derived from wall clock here; callers pass it
explicitly (e.g. file download time) or it stays None and validation flags it.
"""

from __future__ import annotations

import csv
import io
import zipfile
from typing import Iterable, Iterator, List, Optional, Tuple

from ..model import (
    MarketEvent,
    PriceLevel,
    make_book_snapshot,
    make_quote,
    make_trade,
)

SPOT_EXCHANGE = "binance_spot"
FUTURES_EXCHANGE = "binance_futures_um"

# ---------------------------------------------------------------------------
# aggTrades rows
# ---------------------------------------------------------------------------


def aggtrade_row_to_event(
    row: List[str],
    exchange: str,
    symbol: str,
    receive_timestamp: Optional[int] = None,
    has_header: bool = False,
) -> MarketEvent:
    """Convert one raw aggTrades CSV row to a TradeEvent.

    Spot rows (no header) use ts_us directly; futures rows (header) use
    transact_time in ms, converted to µs.
    """
    if has_header:
        # futures: agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker
        agg_id = int(row[0])
        price = float(row[1])
        qty = float(row[2])
        ts_us = int(row[5]) * 1000
        buyer_maker = row[6].strip().lower() in ("true", "1")
    else:
        # spot: agg_id,price,qty,first_id,last_id,ts_us,buyer_maker,best_match
        agg_id = int(row[0])
        price = float(row[1])
        qty = float(row[2])
        ts_us = int(row[5])
        buyer_maker = row[6].strip().lower() in ("true", "1")
    return make_trade(
        exchange=exchange,
        symbol=symbol,
        event_timestamp=ts_us,
        receive_timestamp=receive_timestamp,
        sequence_number=agg_id,
        trade_id=agg_id,
        price=price,
        quantity=qty,
        is_buyer_maker=buyer_maker,
        source="binance_vision_aggtrades",
    )


def aggtrade_rows_from_zip(
    zip_path: str,
    exchange: str,
    symbol: str,
    receive_timestamp: Optional[int] = None,
) -> Iterator[Tuple[List[str], bool]]:
    """Yield (row, has_header) tuples read sequentially from a Vision zip.

    Streaming: reads the single member file line by line, never loads the
    whole day into memory. Auto-detects the futures header.
    """
    with zipfile.ZipFile(zip_path) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
            reader = csv.reader(text)
            has_header = False
            started = False
            for row in reader:
                if not row or not row[0].strip():
                    continue
                if not started:
                    started = True
                    has_header = row[0].strip().startswith("agg_trade_id") or row[0].strip().startswith("agg_id")
                    if has_header:
                        continue
                yield row, has_header


def aggtrade_events_from_zip(
    zip_path: str,
    exchange: str,
    symbol: str,
    receive_timestamp: Optional[int] = None,
) -> Iterator[MarketEvent]:
    for row, has_header in aggtrade_rows_from_zip(
        zip_path, exchange, symbol, receive_timestamp
    ):
        yield aggtrade_row_to_event(
            row, exchange, symbol, receive_timestamp, has_header
        )


# ---------------------------------------------------------------------------
# Recorder depth rows
# ---------------------------------------------------------------------------


def recorder_depth_row_to_event(
    row: dict,
    exchange: str,
    symbol: str,
    receive_timestamp: Optional[int] = None,
) -> MarketEvent:
    """Convert one recorder depth JSONL dict to a partial ORDER_BOOK_SNAPSHOT.

    Row schema: {"t": ms, "U": first_seq, "u": final_seq, "b": [[p,q]], "a": [[p,q]]}
    Prices/quantities arrive as strings; canonicalized to float here.
    """
    t_ms = int(row["t"])
    first_seq = int(row["U"])
    final_seq = int(row["u"])
    bids = [(float(p), float(q)) for p, q in row["b"]]
    asks = [(float(p), float(q)) for p, q in row["a"]]
    event = make_book_snapshot(
        exchange=exchange,
        symbol=symbol,
        event_timestamp=t_ms * 1000,
        receive_timestamp=receive_timestamp,
        sequence_number=first_seq,
        bids=bids,
        asks=asks,
        first_update_id=first_seq,
        final_update_id=final_seq,
        source="recorder_depth_jsonl",
        partial=True,
    )
    return event


# ---------------------------------------------------------------------------
# bookTicker → QUOTE
# ---------------------------------------------------------------------------


def book_ticker_row_to_quote(
    row: dict,
    exchange: str,
    symbol: Optional[str] = None,
    receive_timestamp: Optional[int] = None,
) -> MarketEvent:
    """Convert a bookTicker message to a QUOTE event.

    Row schema: {"u": update_id, "s": "BTCUSDT", "b": bid, "B": bid_size,
    "a": ask, "A": ask_size, "T": event ms}
    """
    sym = symbol or row.get("s")
    if not sym:
        raise ValueError("bookTicker row missing symbol")
    return make_quote(
        exchange=exchange,
        symbol=sym,
        event_timestamp=int(row["T"]) * 1000,
        receive_timestamp=receive_timestamp,
        sequence_number=int(row["u"]),
        quote_id=int(row["u"]),
        bid=float(row["b"]),
        ask=float(row["a"]),
        bid_size=float(row.get("B", 0.0)),
        ask_size=float(row.get("A", 0.0)),
        source="binance_bookticker",
    )


# ---------------------------------------------------------------------------
# Convenience: iterate a whole day of zips into events
# ---------------------------------------------------------------------------


def day_aggtrade_events(
    symbol: str,
    day: str,
    market: str = "spot",
    receive_timestamp: Optional[int] = None,
    data_dir: Optional[str] = None,
) -> Iterator[MarketEvent]:
    """Yield TradeEvents for one day from the local Vision cache.

    Path convention (arbitrage layer): <data_dir>/<market>/aggTrades/<SYM>/<day>.zip
    """
    import os

    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "arbitrage", "data")
    exchange = SPOT_EXCHANGE if market == "spot" else FUTURES_EXCHANGE
    path = os.path.join(data_dir, market, "aggTrades", symbol, f"{day}.zip")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    yield from aggtrade_events_from_zip(path, exchange, symbol, receive_timestamp)
