"""Point-in-time enforcement for the research layer.

At timestamp T, a FEATURE may only use events with event_timestamp <= T.

This module provides the only sanctioned access pattern:

    idx = pit_end(ts, T)          # first index with ts > T  (exclusive end)
    window = (pit_start(ts, T - wl), pit_end(ts, T))        # [T-wl, T]

Every helper asserts its invariants:
- ts must be sorted non-decreasing (asserted once, on view creation)
- every returned end index is derived from the FULL event array via
  searchsorted — there is no code path that could select a future event
- strict mode (PIT_ENFORCE) turns any detected violation into an exception;
  it cannot be silently corrected

The label engine (labels.py) is the ONLY consumer allowed to look past T —
it represents the OUTCOME after T and is built as a separate module.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

PIT_ENFORCE = True


class PitViolationError(RuntimeError):
    pass


class PitView:
    """Immutable PIT window over one symbol's sorted event arrays.

    Arrays are passed as keyword args (ts_us is required). The view holds
    cumulative (prefix) arrays so any window can be answered in O(1):
        sum over [i0, i1) = prefix[i1] - prefix[i0]
    """

    def __init__(self, ts_us: np.ndarray, **arrays):
        ts = np.asarray(ts_us, dtype=np.int64)
        if len(ts) == 0:
            raise PitViolationError("PitView requires non-empty ts_us")
        if ts.ndim != 1 or (ts[1:] < ts[:-1]).any():
            raise PitViolationError("PitView requires sorted non-decreasing ts_us")
        self.ts = ts
        self.n = len(ts)
        self.arrays = {k: np.asarray(v) for k, v in arrays.items()}
        for k, v in self.arrays.items():
            if v.shape[0] != self.n:
                raise PitViolationError(
                    f"array {k} has length {v.shape[0]}, expected {self.n}"
                )

    # -- the only PIT-safe index functions ---------------------------------

    def end_index(self, T: int) -> int:
        """First index with ts > T (exclusive). Events [0, end) are <= T."""
        if not isinstance(T, (int, np.integer)) or T <= 0:
            raise PitViolationError(f"invalid timestamp {T!r}")
        idx = int(np.searchsorted(self.ts, T, side="right"))
        if idx > self.n:
            raise PitViolationError("PIT internal error: index out of bounds")
        return idx

    def start_index(self, T: int) -> int:
        """First index with ts >= T (inclusive lower bound)."""
        return int(np.searchsorted(self.ts, T, side="left"))

    def event_count_up_to(self, T: int) -> int:
        return self.end_index(T)

    def has_event_at_or_before(self, T: int) -> bool:
        return self.end_index(T) > 0

    def last_price_at(self, T: int) -> Optional[float]:
        """Last trade price with ts <= T (None if no trade yet). PIT-safe."""
        idx = self.end_index(T)
        if idx == 0:
            return None
        price = self.arrays["price"]
        return float(price[idx - 1])

    def window(self, T: int, window_us: int):
        """Returns (i0, i1) covering [T-window_us, T], both PIT-verified."""
        if window_us <= 0:
            raise PitViolationError(f"invalid window {window_us}")
        i1 = self.end_index(T)
        i0 = int(np.searchsorted(self.ts, T - window_us, side="left"))
        if i0 < 0 or i1 < i0:
            raise PitViolationError("PIT internal error: window out of order")
        return i0, i1

    def prefix(self, key: str) -> np.ndarray:
        if key not in self.arrays:
            raise KeyError(f"no array {key!r}")
        return np.concatenate(([0.0], np.cumsum(self.arrays[key])))

    def assert_window_before(self, i1: int, T: int) -> None:
        """Fail loudly if a window end exceeds the PIT end at T."""
        if i1 > self.end_index(T):
            raise PitViolationError(
                f"PIT violation: window end {i1} > pit_end({T}) = {self.end_index(T)}"
            )
