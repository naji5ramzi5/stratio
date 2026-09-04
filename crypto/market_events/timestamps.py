"""Timestamp model for the event layer.

Canonical unit: **int64 microseconds since Unix epoch (UTC)**.

Rules:
- event_timestamp:  the timestamp the exchange attributes to the event.
- receive_timestamp: the timestamp at which WE received/ingested the event.
- receive_timestamp must NEVER be auto-filled from wall clock. If a source
  does not provide it, the adapter must pass it explicitly (e.g. download
  time from a manifest) or leave it as None and let validation flag it.

Precision conversion table (all lossless for the units Binance publishes):
    seconds → µs : * 1_000_000
    ms      → µs : * 1_000
    µs      → µs : identity
"""

from __future__ import annotations

from typing import Union

MICROS_PER_SECOND = 1_000_000
MICROS_PER_MILLI = 1_000

Timestamp = Union[int, None]


def us_from_s(seconds: Union[int, float]) -> int:
    if seconds is None:
        raise ValueError("cannot convert None to microseconds")
    return int(seconds * MICROS_PER_SECOND)


def us_from_ms(millis: Union[int, float]) -> int:
    if millis is None:
        raise ValueError("cannot convert None to microseconds")
    return int(millis * MICROS_PER_MILLI)


def us_from_us(micros: Union[int, float]) -> int:
    if micros is None:
        raise ValueError("cannot convert None to microseconds")
    return int(micros)


def normalize(ts: Timestamp, unit: str = "us") -> Timestamp:
    """Convert *ts* (given in *unit*) to canonical microseconds.

    ``unit`` in {"s", "ms", "us"}. Returns None iff ts is None
    (never invents a timestamp).
    """
    if ts is None:
        return None
    if unit == "s":
        return us_from_s(ts)
    if unit == "ms":
        return us_from_ms(ts)
    if unit == "us":
        return us_from_us(ts)
    raise ValueError(f"unknown timestamp unit: {unit!r}")
