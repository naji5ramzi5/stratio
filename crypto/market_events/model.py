"""Canonical, exchange-neutral market event model.

Every event is a typed payload carrying a common envelope:

    exchange, symbol, event_timestamp, receive_timestamp, sequence_number

- event_timestamp / receive_timestamp are int64 microseconds (see timestamps.py).
- sequence_number is the exchange sequence (trade id, U/u pair) when the
  source provides one, otherwise None.
- source identifies the original data record for provenance, e.g.
  "binance_vision_spot_aggtrades" or "recorder_depth_jsonl".

Prices and quantities are always canonicalized to float (price) and float
(quantity) by the adapters; nothing Binance-specific leaks into these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

PriceLevel = Tuple[float, float]  # (price, quantity)


class EventType(Enum):
    TRADE = "TRADE"
    QUOTE = "QUOTE"
    ORDER_BOOK_SNAPSHOT = "ORDER_BOOK_SNAPSHOT"
    ORDER_BOOK_UPDATE = "ORDER_BOOK_UPDATE"


@dataclass(frozen=True, slots=True)
class MarketEvent:
    event_type: EventType
    exchange: str
    symbol: str
    event_timestamp: int
    receive_timestamp: Optional[int]
    sequence_number: Optional[int]
    source: str
    payload: object

    @property
    def time(self) -> int:
        """Convenience alias: canonical event timestamp (µs)."""
        return self.event_timestamp


@dataclass(frozen=True, slots=True)
class TradePayload:
    price: float
    quantity: float
    is_buyer_maker: bool
    trade_id: Optional[int] = None


@dataclass(frozen=True, slots=True)
class QuotePayload:
    bid: float
    ask: float
    bid_size: float = 0.0
    ask_size: float = 0.0
    quote_id: Optional[int] = None


@dataclass(frozen=True, slots=True)
class OrderBookSnapshotPayload:
    bids: List[PriceLevel]
    asks: List[PriceLevel]
    first_update_id: int = 0
    final_update_id: int = 0
    partial: bool = False


@dataclass(frozen=True, slots=True)
class OrderBookUpdatePayload:
    """Order-book diff: price-level changes on either side.

    Updates are list of PriceLevel. Levels with quantity == 0 mean
    "delete this level" (Binance diff semantics). For sources that send
    full top-N replacement books (e.g. the recorder), use
    OrderBookSnapshotPayload instead.
    """

    bid_updates: List[PriceLevel] = field(default_factory=list)
    ask_updates: List[PriceLevel] = field(default_factory=list)
    first_update_id: Optional[int] = None
    final_update_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Factory helpers: the only sanctioned way to construct events
# ---------------------------------------------------------------------------


def make_trade(
    exchange: str,
    symbol: str,
    event_timestamp: int,
    price: float,
    quantity: float,
    is_buyer_maker: bool,
    trade_id: Optional[int] = None,
    receive_timestamp: Optional[int] = None,
    sequence_number: Optional[int] = None,
    source: str = "adapter",
) -> MarketEvent:
    if sequence_number is None:
        sequence_number = trade_id
    return MarketEvent(
        event_type=EventType.TRADE,
        exchange=exchange,
        symbol=symbol,
        event_timestamp=event_timestamp,
        receive_timestamp=receive_timestamp,
        sequence_number=sequence_number,
        source=source,
        payload=TradePayload(price, quantity, is_buyer_maker, trade_id),
    )


def make_quote(
    exchange: str,
    symbol: str,
    event_timestamp: int,
    bid: float,
    ask: float,
    bid_size: float = 0.0,
    ask_size: float = 0.0,
    quote_id: Optional[int] = None,
    receive_timestamp: Optional[int] = None,
    sequence_number: Optional[int] = None,
    source: str = "adapter",
) -> MarketEvent:
    if sequence_number is None:
        sequence_number = quote_id
    return MarketEvent(
        event_type=EventType.QUOTE,
        exchange=exchange,
        symbol=symbol,
        event_timestamp=event_timestamp,
        receive_timestamp=receive_timestamp,
        sequence_number=sequence_number,
        source=source,
        payload=QuotePayload(bid, ask, bid_size, ask_size, quote_id),
    )


def make_book_snapshot(
    exchange: str,
    symbol: str,
    event_timestamp: int,
    bids: List[PriceLevel],
    asks: List[PriceLevel],
    first_update_id: int = 0,
    final_update_id: int = 0,
    receive_timestamp: Optional[int] = None,
    sequence_number: Optional[int] = None,
    source: str = "adapter",
    partial: bool = False,
) -> MarketEvent:
    if sequence_number is None:
        sequence_number = first_update_id
    return MarketEvent(
        event_type=EventType.ORDER_BOOK_SNAPSHOT,
        exchange=exchange,
        symbol=symbol,
        event_timestamp=event_timestamp,
        receive_timestamp=receive_timestamp,
        sequence_number=sequence_number,
        source=source,
        payload=OrderBookSnapshotPayload(
            bids, asks, first_update_id, final_update_id, partial
        ),
    )


def make_book_update(
    exchange: str,
    symbol: str,
    event_timestamp: int,
    bid_updates: List[PriceLevel],
    ask_updates: List[PriceLevel],
    first_update_id: Optional[int] = None,
    final_update_id: Optional[int] = None,
    receive_timestamp: Optional[int] = None,
    sequence_number: Optional[int] = None,
    source: str = "adapter",
) -> MarketEvent:
    if sequence_number is None:
        sequence_number = first_update_id
    return MarketEvent(
        event_type=EventType.ORDER_BOOK_UPDATE,
        exchange=exchange,
        symbol=symbol,
        event_timestamp=event_timestamp,
        receive_timestamp=receive_timestamp,
        sequence_number=sequence_number,
        source=source,
        payload=OrderBookUpdatePayload(
            bid_updates, ask_updates, first_update_id, final_update_id
        ),
    )


def event_type_name(event: MarketEvent) -> str:
    return event.event_type.value
