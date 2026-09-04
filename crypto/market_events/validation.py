"""Event validation.

Rules are strict by design: **never silently repair**. Every defect is
reported with a machine-readable reason. A validating consumer can either
reject the event or skip it, but can never pretend it did not happen.

Checks implemented:
- envelope: event_type, exchange, symbol present and typed correctly
- timestamps: event_timestamp is a positive int; receive_timestamp, if
  present, must be a positive int (missing receive is a defect, reported
  as "missing_receive_timestamp", never filled in)
- prices/quantities: price > 0, quantity >= 0, finite
- quote sanity: ask > bid (no crossed or empty quote)
- book sanity: all levels price > 0, quantity >= 0, bids sorted desc,
  asks sorted asc, no crossed books (best bid < best ask)
- sequence: sequence_number, when present, must be a non-negative int

Ordering / duplicate detection is NOT part of this module: it is the job
of the replay engine (monotonic clock + duplicate rejection).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from .model import (
    EventType,
    MarketEvent,
    OrderBookSnapshotPayload,
    OrderBookUpdatePayload,
    QuotePayload,
    TradePayload,
)

REASON_ENVELOPE = "envelope"
REASON_TIMESTAMP = "timestamp"
REASON_RECEIVE = "missing_receive_timestamp"
REASON_QUANTITY = "quantity"
REASON_PRICE = "price"
REASON_QUOTE_CROSSED = "quote_crossed"
REASON_BOOK_LEVELS = "book_levels"
REASON_BOOK_SORT = "book_sort"
REASON_BOOK_CROSSED = "book_crossed"
REASON_SEQUENCE = "sequence"
REASON_UNKNOWN_TYPE = "unknown_event_type"


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reasons: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid

    def __str__(self) -> str:
        if self.valid:
            return "valid"
        return "invalid: " + ", ".join(self.reasons)


def validate(event: MarketEvent) -> ValidationResult:
    reasons: List[str] = []

    if not isinstance(event, MarketEvent):
        return ValidationResult(False, [REASON_ENVELOPE])

    if not isinstance(event.event_type, EventType):
        reasons.append(REASON_UNKNOWN_TYPE)
    if not isinstance(event.exchange, str) or not event.exchange:
        reasons.append(REASON_ENVELOPE)
    if not isinstance(event.symbol, str) or not event.symbol:
        reasons.append(REASON_ENVELOPE)

    ets = event.event_timestamp
    if not isinstance(ets, int) or isinstance(ets, bool) or ets <= 0:
        reasons.append(REASON_TIMESTAMP)
    if event.receive_timestamp is None:
        reasons.append(REASON_RECEIVE)
    elif not isinstance(event.receive_timestamp, int) or event.receive_timestamp <= 0:
        reasons.append(REASON_TIMESTAMP)

    if event.sequence_number is not None and (
        not isinstance(event.sequence_number, int)
        or isinstance(event.sequence_number, bool)
        or event.sequence_number < 0
    ):
        reasons.append(REASON_SEQUENCE)

    payload = event.payload
    if isinstance(payload, TradePayload):
        _validate_price_quantity(payload.price, payload.quantity, reasons)
    elif isinstance(payload, QuotePayload):
        _validate_price_quantity(payload.bid, payload.bid_size, reasons)
        _validate_price_quantity(payload.ask, payload.ask_size, reasons)
        if payload.ask <= payload.bid:
            reasons.append(REASON_QUOTE_CROSSED)
    elif isinstance(payload, OrderBookSnapshotPayload):
        reasons += _validate_levels(payload.bids, "bid")
        reasons += _validate_levels(payload.asks, "ask")
        reasons += _validate_book(payload.bids, payload.asks)
    elif isinstance(payload, OrderBookUpdatePayload):
        reasons += _validate_levels(payload.bid_updates, "bid")
        reasons += _validate_levels(payload.ask_updates, "ask")
    else:
        reasons.append(REASON_UNKNOWN_TYPE)

    return ValidationResult(not reasons, reasons)


def _validate_price_quantity(
    price: float, quantity: float, reasons: List[str]
) -> None:
    if not isinstance(price, (int, float)) or math.isnan(price) or price <= 0:
        reasons.append(REASON_PRICE)
    if not isinstance(quantity, (int, float)) or math.isnan(quantity) or quantity < 0:
        reasons.append(REASON_QUANTITY)


def _validate_levels(levels, side: str) -> List[str]:
    reasons: List[str] = []
    for idx, (price, qty) in enumerate(levels):
        if (
            not isinstance(price, (int, float))
            or math.isnan(price)
            or price <= 0
        ):
            reasons.append(f"{REASON_PRICE}:{side}:{idx}")
        if not isinstance(qty, (int, float)) or math.isnan(qty) or qty < 0:
            reasons.append(f"{REASON_QUANTITY}:{side}:{idx}")
    return reasons


def _validate_book(bids, asks) -> List[str]:
    reasons: List[str] = []
    bid_prices = [p for p, _ in bids]
    ask_prices = [p for p, _ in asks]
    if any(bid_prices[i] < bid_prices[i + 1] for i in range(len(bid_prices) - 1)):
        reasons.append(f"{REASON_BOOK_SORT}:bids")
    if any(ask_prices[i] > ask_prices[i + 1] for i in range(len(ask_prices) - 1)):
        reasons.append(f"{REASON_BOOK_SORT}:asks")
    if bids and asks and bid_prices[0] >= ask_prices[0]:
        reasons.append(REASON_BOOK_CROSSED)
    return reasons
