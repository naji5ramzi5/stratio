"""
rich/time_sync.py — Server time synchronization with Binance.

Ensures local timestamps used in HMAC signatures are within Binance's
recvWindow. A large clock offset causes all authenticated requests to fail.

Usage:
    sync = TimeSync()
    offset_ms = await sync.measure_offset(adapter)
    server_time = sync.server_time()
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("rich.time_sync")


class TimeSync:
    """Tracks the offset between local time and Binance server time."""

    def __init__(self, max_offset_ms: float = 1000.0):
        self._offset_ms: float = 0.0   # local_time + offset = server_time
        self._last_sync_ts: float = 0.0
        self._max_offset_ms = max_offset_ms
        self._sync_count: int = 0

    @property
    def offset_ms(self) -> float:
        return self._offset_ms

    @property
    def last_sync_time(self) -> Optional[str]:
        if not self._last_sync_ts:
            return None
        return datetime.fromtimestamp(self._last_sync_ts, tz=timezone.utc).isoformat()

    def update(self, server_time_ms: int) -> float:
        """
        Record the exchange server time (in ms) and compute offset.
        Call this after every GET /api/v3/time response.

        Returns the new offset in milliseconds.
        """
        local_ms = time.time() * 1000
        self._offset_ms = server_time_ms - local_ms
        self._last_sync_ts = time.time()
        self._sync_count += 1

        abs_offset = abs(self._offset_ms)
        if abs_offset > self._max_offset_ms:
            logger.warning(
                f"[TimeSync] Clock offset LARGE: {self._offset_ms:.1f}ms "
                f"(max allowed={self._max_offset_ms}ms)"
            )
        else:
            logger.debug(f"[TimeSync] Offset: {self._offset_ms:.1f}ms")

        return self._offset_ms

    def server_time_ms(self) -> int:
        """Return estimated current server time in milliseconds."""
        return int(time.time() * 1000 + self._offset_ms)

    def is_healthy(self) -> bool:
        """True if offset is within acceptable bounds."""
        return abs(self._offset_ms) <= self._max_offset_ms

    def needs_sync(self, max_age_seconds: float = 60.0) -> bool:
        """True if we haven't synced recently."""
        if not self._last_sync_ts:
            return True
        return (time.time() - self._last_sync_ts) > max_age_seconds

    def to_dict(self) -> dict:
        return {
            "offset_ms": round(self._offset_ms, 2),
            "last_sync": self.last_sync_time,
            "sync_count": self._sync_count,
            "is_healthy": self.is_healthy(),
        }
