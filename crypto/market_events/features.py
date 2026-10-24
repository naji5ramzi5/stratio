"""Feature-calculator interface (Phase 1 scope: interface only).

The contract a strategy-facing layer will consume:

    MarketState (from ReplayEngine.current_market_state / a TRADE event)
    → FeatureCalculator.calculate(state) → FeatureSet (dict of float)

Rules:
- calculators are stateless and deterministic (pure functions of the input)
- they never hold cross-event state; any memory (windows, running stats)
  belongs to a later phase's feature pipeline, NOT here
- output keys are stable dotted identifiers, e.g. "micro.mid_bps"

The three example calculators below exist ONLY to validate the interface
(and feed the integration tests / benchmark). They are not a strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, runtime_checkable

from .model import MarketEvent
from .orderbook import OrderBook

FeatureSet = Dict[str, float]


@dataclass(frozen=True)
class MarketState:
    event_time_us: int
    event_type: str
    best_bid: Optional[float]
    best_ask: Optional[float]
    mid: Optional[float]
    spread_bps: Optional[float]
    bid_depth: Optional[float] = None
    ask_depth: Optional[float] = None

    @classmethod
    def from_book(cls, book: Optional[OrderBook], event_time_us: int, event_type: str) -> "MarketState":
        if book is None:
            return cls(event_time_us, event_type, None, None, None, None)
        return cls(
            event_time_us=event_time_us,
            event_type=event_type,
            best_bid=book.best_bid,
            best_ask=book.best_ask,
            mid=book.mid,
            spread_bps=book.spread_bps,
            bid_depth=book.depth_bps("bid", book.mid) if book.mid else None,
            ask_depth=book.depth_bps("ask", book.mid) if book.mid else None,
        )


@runtime_checkable
class FeatureCalculator(Protocol):
    name: str

    def calculate(self, state: MarketState) -> FeatureSet:
        ...


class MidPriceCalculator:
    """Validates the interface: returns the mid price (quote currency)."""

    name = "mid"

    def calculate(self, state: MarketState) -> FeatureSet:
        if state.mid is None:
            return {"mid": float("nan")}
        return {"mid": state.mid}


class SpreadBpsCalculator:
    """Validates the interface: returns the quoted spread in bps."""

    name = "spread_bps"

    def calculate(self, state: MarketState) -> FeatureSet:
        if state.spread_bps is None:
            return {"spread_bps": float("nan")}
        return {"spread_bps": state.spread_bps}


class BookImbalanceCalculator:
    """Validates the interface: bid depth / (bid depth + ask depth) in bps-quotes."""

    name = "book_imbalance"

    def calculate(self, state: MarketState) -> FeatureSet:
        b, a = state.bid_depth, state.ask_depth
        if b is None or a is None or (b + a) <= 0:
            return {"book_imbalance": float("nan")}
        return {"book_imbalance": b / (b + a)}


EXAMPLE_CALCULATORS = [
    MidPriceCalculator(),
    SpreadBpsCalculator(),
    BookImbalanceCalculator(),
]


def compute_all(state: MarketState, calculators=EXAMPLE_CALCULATORS) -> FeatureSet:
    out: FeatureSet = {}
    for calc in calculators:
        for key, value in calc.calculate(state).items():
            out[f"{calc.name}.{key}"] = value
    return out
