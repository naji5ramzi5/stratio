"""Event replay engine with point-in-time guarantees.

Design:

    ReplayEngine iterates a strictly time-ordered event stream. The ONLY way
    to observe market state is to advance the clock with advance_to();
    every advance consumes exactly the events with event_timestamp <= target,
    exposes them through current_market_state / current_event_time, and
    leaves future events untouched in the stream.

    Fail-loud guarantees:
    - advance_to() with a target earlier than current_event_time
      (rewinding) raises ReplayClockError
    - non-monotonic input streams raise ReplayClockError
    - duplicate trade sequence numbers raise ReplayPITError
    - order-book rejections (sequence gaps/rewinds, crossed books) raise
      ReplayPITError — the consumer learns the replay is corrupt instead of
      silently building a wrong book
    - there is no API that reads events without consuming them: peeking into
      the future is structurally impossible

Exposed state (spec Step 9):
    current_event_time      ts of last consumed event (µs), None before start
    current_receive_time    receive ts of last consumed event
    current_market_state    OrderBook fed by book events (or None)
    event_sequence          number of events consumed
"""

from __future__ import annotations

from typing import Iterable, Iterator, Optional

from .model import EventType, MarketEvent
from .orderbook import OrderBook, OrderBookError


class ReplayPITError(RuntimeError):
    pass


class ReplayClockError(RuntimeError):
    pass


class ReplayEngine:
    def __init__(
        self,
        events: Iterable[MarketEvent],
        require_monotonic: bool = True,
        build_book: bool = True,
    ):
        self._iter: Iterator[MarketEvent] = iter(events)
        self._require_monotonic = require_monotonic
        self._build_book = build_book
        self._current_event_time: Optional[int] = None
        self._current_receive_time: Optional[int] = None
        self._event_sequence: int = 0
        self._book: Optional[OrderBook] = None
        self._last_event: Optional[MarketEvent] = None
        self._trade_seqs_seen: set = set()
        self._pending: Optional[MarketEvent] = None  # lookahead buffer

    # -- observable state ---------------------------------------------------

    @property
    def current_event_time(self) -> Optional[int]:
        return self._current_event_time

    @property
    def current_receive_time(self) -> Optional[int]:
        return self._current_receive_time

    @property
    def event_sequence(self) -> int:
        return self._event_sequence

    @property
    def current_market_state(self) -> Optional[OrderBook]:
        return self._book

    @property
    def at_start(self) -> bool:
        return self._current_event_time is None

    @property
    def last_event(self) -> Optional[MarketEvent]:
        return self._last_event

    # -- clock control ------------------------------------------------------

    def advance_to(self, target_ts: int) -> Optional[MarketEvent]:
        """Consume all events with event_timestamp <= target_ts.

        Returns the last consumed event, or None if none qualified. Events
        strictly after *target_ts* stay in the stream for a later advance.
        Raises ReplayClockError if the clock cannot rewind to *target_ts*.
        """
        if not isinstance(target_ts, int) or target_ts <= 0:
            raise ReplayClockError(f"invalid target timestamp: {target_ts!r}")
        cur = self._current_event_time
        if cur is not None and target_ts < cur:
            raise ReplayClockError(
                f"clock rewind: cannot advance to {target_ts}, "
                f"clock already at {cur}"
            )
        last: Optional[MarketEvent] = None
        while True:
            event = self._peek()
            if event is None:
                break
            if event.event_timestamp > target_ts:
                break
            self._take()
            self._consume(event)
            last = event
        return last

    def advance_by(self, delta_us: int) -> Optional[MarketEvent]:
        """Advance the clock by *delta_us* microseconds."""
        if delta_us <= 0:
            raise ReplayClockError("delta_us must be positive")
        base = self._current_event_time if self._current_event_time is not None else 0
        return self.advance_to(base + delta_us)

    # -- internals ----------------------------------------------------------

    def _peek(self) -> Optional[MarketEvent]:
        """Look at the next event WITHOUT consuming it (PIT-safe lookahead).

        This is the ONLY place the underlying iterator is touched; a
        break/stop never loses an event.
        """
        if self._pending is None:
            try:
                self._pending = next(self._iter)
            except StopIteration:
                return None
        return self._pending

    def _take(self) -> None:
        self._pending = None

    def _consume(self, event: MarketEvent) -> None:
        if self._require_monotonic:
            prev = self._current_event_time
            if prev is not None and event.event_timestamp < prev:
                raise ReplayClockError(
                    f"non-monotonic clock: event at {event.event_timestamp} "
                    f"after clock at {prev}"
                )
        if event.event_type is EventType.TRADE and event.sequence_number is not None:
            seq = event.sequence_number
            if seq in self._trade_seqs_seen:
                raise ReplayPITError(
                    f"duplicate trade sequence {seq} (event {event.source})"
                )
            self._trade_seqs_seen.add(seq)
        self._current_event_time = event.event_timestamp
        self._current_receive_time = event.receive_timestamp
        self._event_sequence += 1
        self._last_event = event
        if self._build_book and event.event_type in (
            EventType.ORDER_BOOK_SNAPSHOT,
            EventType.ORDER_BOOK_UPDATE,
        ):
            if self._book is None:
                self._book = OrderBook(event.exchange, event.symbol)
            try:
                self._book.apply(event)
            except OrderBookError as exc:
                raise ReplayPITError(f"order book rejected event: {exc}") from exc
