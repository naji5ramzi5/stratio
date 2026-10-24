"""Order-book state engine.

Maintains a best-effort top-of-book and price-level state from canonical
events. Two consumption modes:

- SNAPSHOT events: replace levels. If ``partial``, replace only the levels
  present in the event (the rest of the state is kept); otherwise clear and
  rebuild the side entirely. Partial books (top-N recorder) only ever touch
  the visible top; all other levels are removed so the state is always an
  honest copy of what we have seen — never a fabricated full book.
- UPDATE events (diffs): apply level changes. quantity == 0 removes the
  level (Binance semantics).

Invariants enforced:
- events must arrive in sequence-number order (when sequence numbers exist);
  a gap or rewind raises OrderBookSequenceError
- price > 0, quantity >= 0 (re-validated here — adapters are not trusted)
- never crossed: a bid >= best ask raises OrderBookCrossedError
- a full snapshot event clears stale levels before applying

Determinism guarantee: applying the same event sequence to two fresh
OrderBook instances yields identical state (pure function of the events).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .model import (
    EventType,
    MarketEvent,
    OrderBookSnapshotPayload,
    OrderBookUpdatePayload,
)

PriceLevel = Tuple[float, float]


class OrderBookError(Exception):
    pass


class OrderBookSequenceError(OrderBookError):
    pass


class OrderBookCrossedError(OrderBookError):
    pass


class OrderBookValidationError(OrderBookError):
    pass


class OrderBook:
    """Level-price-keyed order book with lazy best-level access."""

    def __init__(self, exchange: str, symbol: str, max_levels: int = 100_000):
        self.exchange = exchange
        self.symbol = symbol
        self.max_levels = max_levels
        self._bids: Dict[float, float] = {}  # price -> qty
        self._asks: Dict[float, float] = {}
        self._bid_prices: List[float] = []
        self._ask_prices: List[float] = []
        self._version = 0
        self._last_sequence: Optional[int] = None

    # -- state queries ------------------------------------------------------

    @property
    def best_bid(self) -> Optional[float]:
        if not self._bids:
            return None
        return self._bid_prices[0]

    @property
    def best_ask(self) -> Optional[float]:
        if not self._asks:
            return None
        return self._ask_prices[0]

    @property
    def mid(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    @property
    def spread(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return ba - bb

    @property
    def spread_bps(self) -> Optional[float]:
        bb, ba = self.best_bid, self.best_ask
        mid = self.mid
        if bb is None or ba is None or mid is None or mid <= 0:
            return None
        return (ba - bb) / mid * 10_000.0

    @property
    def version(self) -> int:
        return self._version

    def size_at(self, price: float, side: str) -> float:
        levels = self._bids if side == "bid" else self._asks
        return levels.get(price, 0.0)

    def top_n(self, side: str, n: int = 5) -> List[PriceLevel]:
        prices = self._bid_prices if side == "bid" else self._ask_prices
        levels = self._bids if side == "bid" else self._asks
        out = []
        for p in prices[:n]:
            q = levels[p]
            if q > 0:
                out.append((p, q))
        return out

    def depth_bps(self, side: str, quote: float, n_levels: int = 5) -> float:
        """Cumulative notional (price×qty) over top *n_levels*."""
        total = 0.0
        for p, q in self.top_n(side, n_levels):
            total += p * q
        return total / quote * 10_000.0 if quote > 0 else 0.0

    # -- state mutation -----------------------------------------------------

    def apply(self, event: MarketEvent) -> "OrderBook":
        self._check_event(event)
        payload = event.payload
        if isinstance(payload, OrderBookSnapshotPayload):
            self._apply_snapshot(payload)
        elif isinstance(payload, OrderBookUpdatePayload):
            self._apply_update(payload)
        else:
            raise OrderBookValidationError(
                f"OrderBook.apply requires a book event, got {event.event_type}"
            )
        self._version += 1
        self._last_sequence = event.sequence_number
        return self

    def _check_event(self, event: MarketEvent) -> None:
        if event.event_type not in (EventType.ORDER_BOOK_SNAPSHOT, EventType.ORDER_BOOK_UPDATE):
            raise OrderBookValidationError(f"not a book event: {event.event_type}")
        if event.exchange != self.exchange or event.symbol != self.symbol:
            raise OrderBookValidationError(
                f"book {self.exchange}/{self.symbol} got event {event.exchange}/{event.symbol}"
            )
        seq = event.sequence_number
        if self._last_sequence is not None and seq is not None:
            if seq < self._last_sequence:
                raise OrderBookSequenceError(
                    f"sequence rewind: had {self._last_sequence}, got {seq}"
                )
            if seq == self._last_sequence:
                raise OrderBookSequenceError(f"duplicate sequence: {seq}")

    def _apply_snapshot(self, payload: OrderBookSnapshotPayload) -> None:
        if payload.partial:
            for p, q in payload.bids:
                self._upsert_level(p, q, "bid")
            for p, q in payload.asks:
                self._upsert_level(p, q, "ask")
            self._prune(payload)
        else:
            self._bids = {}
            self._asks = {}
            self._bid_prices = []
            self._ask_prices = []
            for p, q in payload.bids:
                self._upsert_level(p, q, "bid")
            for p, q in payload.asks:
                self._upsert_level(p, q, "ask")
        self._check_crossed()

    def _apply_update(self, payload: OrderBookUpdatePayload) -> None:
        for p, q in payload.bid_updates:
            self._upsert_level(p, q, "bid")
        for p, q in payload.ask_updates:
            self._upsert_level(p, q, "ask")
        self._check_crossed()

    def _upsert_level(self, price: float, qty: float, side: str) -> None:
        if not isinstance(price, (int, float)) or price <= 0:
            raise OrderBookValidationError(f"invalid price {price!r}")
        if not isinstance(qty, (int, float)) or qty < 0:
            raise OrderBookValidationError(f"invalid quantity {qty!r}")
        levels = self._bids if side == "bid" else self._asks
        if qty == 0:
            levels.pop(price, None)
        else:
            levels[price] = qty
        if len(levels) > self.max_levels:
            raise OrderBookValidationError(
                f"book grew beyond max_levels={self.max_levels}"
            )

    def _prune(self, payload: OrderBookSnapshotPayload) -> None:
        """Partial snapshot: drop stale levels BETTER than the new visible
        top (they were not in the top-N the exchange just sent, so they can
        no longer exist). Levels worse than the visible top keep their
        previous state — the update tells us nothing about them."""
        bid_threshold = max((p for p, _ in payload.bids), default=None)
        ask_threshold = min((p for p, _ in payload.asks), default=None)
        if bid_threshold is not None:
            self._bids = {p: q for p, q in self._bids.items() if p <= bid_threshold}
        if ask_threshold is not None:
            self._asks = {p: q for p, q in self._asks.items() if p >= ask_threshold}

    @staticmethod
    def _best_of(levels: Dict[float, float]) -> Optional[float]:
        if not levels:
            return None
        return max(levels)

    def _check_crossed(self) -> None:
        self._reindex()  # level lists must be fresh before best-level reads
        bb, ba = self.best_bid, self.best_ask
        if bb is not None and ba is not None and bb >= ba:
            raise OrderBookCrossedError(
                f"crossed book: best bid {bb} >= best ask {ba}"
            )

    def _reindex(self) -> None:
        self._bid_prices = sorted(self._bids.keys(), reverse=True)
        self._ask_prices = sorted(self._asks.keys())

    def __len__(self) -> int:
        return len(self._bids) + len(self._asks)

    def snapshot(self) -> dict:
        return {
            "exchange": self.exchange,
            "symbol": self.symbol,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid": self.mid,
            "spread_bps": self.spread_bps,
            "bid_depth": len(self._bids),
            "ask_depth": len(self._asks),
            "version": self._version,
        }
