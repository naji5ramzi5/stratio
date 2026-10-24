"""
rich/signal_validator.py — Signal Validator for Bot5.

Every TRADE_READY signal from Bot4 must pass through this gate.
No signal reaches the Risk Engine without a VALID result.

Validation checks:
  1. Schema completeness (required fields)
  2. Signal age (STALE check)
  3. Duplicate detection (idempotency)
  4. Source authenticity
  5. Symbol validity
  6. Side validity
  7. Confidence threshold
  8. Conflict with existing open position
  9. Conflict with existing open orders

Result:
  VALID | STALE | INVALID | DUPLICATE | BLOCKED
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from .config import RichConfig
from .db import RichDB
from .models import SignalContract, SignalStatus, TradeSide

logger = logging.getLogger("rich.signal_validator")


class ValidationResult:
    def __init__(self, status: SignalStatus, reason: str = ""):
        self.status = status
        self.reason = reason
        self.is_valid = status == SignalStatus.VALID

    def __repr__(self) -> str:
        return f"ValidationResult({self.status.value}, reason={self.reason!r})"


class SignalValidator:
    """
    Stateless validator — all state comes from RichDB and RichConfig.

    Idempotency policy:
      Same (signal_id, strategy, symbol, side) tuple must produce at most
      ONE trade. Duplicate check is based on signal_id alone (UNIQUE in DB).
    """

    REQUIRED_FIELDS = ["signal_id", "symbol", "side", "signal_timestamp"]
    VALID_SOURCES = {"BOT4_MASTER"}
    VALID_SIDES = {TradeSide.BUY, TradeSide.SELL}

    def __init__(self, config: RichConfig, db: RichDB):
        self._config = config
        self._db = db

    def validate(self, signal: SignalContract,
                 current_positions: Optional[list] = None,
                 current_open_orders: Optional[list] = None) -> ValidationResult:
        """
        Run all validation checks in order.
        Returns on the first failure — short-circuit logic.
        """
        # 1. Schema completeness
        result = self._check_schema(signal)
        if not result.is_valid:
            return result

        # 2. Source check
        result = self._check_source(signal)
        if not result.is_valid:
            return result

        # 3. Duplicate check (idempotency)
        result = self._check_duplicate(signal)
        if not result.is_valid:
            return result

        # 4. Signal age (STALE)
        result = self._check_freshness(signal)
        if not result.is_valid:
            return result

        # 5. Symbol
        result = self._check_symbol(signal)
        if not result.is_valid:
            return result

        # 6. Side
        result = self._check_side(signal)
        if not result.is_valid:
            return result

        # 7. Confidence threshold
        result = self._check_confidence(signal)
        if not result.is_valid:
            return result

        # 8. Existing position conflict
        if current_positions is not None:
            result = self._check_position_conflict(signal, current_positions)
            if not result.is_valid:
                return result

        # 9. Existing order conflict
        if current_open_orders is not None:
            result = self._check_order_conflict(signal, current_open_orders)
            if not result.is_valid:
                return result

        logger.info(f"[Validator] Signal {signal.signal_id} for {signal.symbol}: VALID")
        return ValidationResult(SignalStatus.VALID, "all checks passed")

    # ── Individual Checks ──────────────────────────────────────────────────────

    def _check_schema(self, sig: SignalContract) -> ValidationResult:
        if not sig.signal_id:
            return ValidationResult(SignalStatus.INVALID, "signal_id is empty")
        if not sig.symbol:
            return ValidationResult(SignalStatus.INVALID, "symbol is empty")
        if not sig.signal_timestamp:
            return ValidationResult(SignalStatus.INVALID, "signal_timestamp is empty")
        try:
            datetime.fromisoformat(sig.signal_timestamp.replace("Z", "+00:00"))
        except ValueError:
            return ValidationResult(SignalStatus.INVALID,
                                    f"signal_timestamp is not valid ISO8601: {sig.signal_timestamp!r}")
        return ValidationResult(SignalStatus.VALID)

    def _check_source(self, sig: SignalContract) -> ValidationResult:
        if sig.source not in self.VALID_SOURCES:
            return ValidationResult(
                SignalStatus.INVALID,
                f"Unknown source: {sig.source!r}. Expected one of {self.VALID_SOURCES}"
            )
        return ValidationResult(SignalStatus.VALID)

    def _check_duplicate(self, sig: SignalContract) -> ValidationResult:
        """
        Idempotency: if signal_id already exists in DB, it was already processed.
        One signal_id → at most one trade entry attempt.
        """
        if self._db.signal_exists(sig.signal_id):
            logger.warning(f"[Validator] DUPLICATE signal_id={sig.signal_id}")
            return ValidationResult(
                SignalStatus.DUPLICATE,
                f"signal_id {sig.signal_id!r} already processed"
            )
        return ValidationResult(SignalStatus.VALID)

    def _check_freshness(self, sig: SignalContract) -> ValidationResult:
        age = sig.age_seconds()
        max_age = self._config.max_signal_age_seconds
        if age > max_age:
            logger.warning(
                f"[Validator] STALE signal {sig.signal_id}: age={age:.0f}s > max={max_age}s"
            )
            return ValidationResult(
                SignalStatus.STALE,
                f"Signal expired: age={age:.0f}s > max_age={max_age}s"
            )
        return ValidationResult(SignalStatus.VALID)

    def _check_symbol(self, sig: SignalContract) -> ValidationResult:
        sym = sig.symbol.strip().upper()
        if not sym:
            return ValidationResult(SignalStatus.INVALID, "symbol is empty")
        if not sym.endswith("USDT"):
            return ValidationResult(
                SignalStatus.BLOCKED,
                f"Only USDT pairs supported in v0.1; got {sym!r}"
            )
        return ValidationResult(SignalStatus.VALID)

    def _check_side(self, sig: SignalContract) -> ValidationResult:
        if sig.side not in self.VALID_SIDES:
            return ValidationResult(SignalStatus.INVALID, f"Invalid side: {sig.side!r}")
        # v0.1: SPOT only → no short selling via direct SELL of base asset
        # SHORT signals would require borrowing (margin) — blocked in v0.1
        if sig.side == TradeSide.SELL:
            return ValidationResult(
                SignalStatus.BLOCKED,
                "SHORT (SELL) signals not supported in v0.1 spot-only mode. "
                "Only BUY signals are executed."
            )
        return ValidationResult(SignalStatus.VALID)

    def _check_confidence(self, sig: SignalContract) -> ValidationResult:
        # Minimum confidence: currently advisory, but we log it.
        # Future: make configurable threshold.
        min_conf = 50.0
        if sig.confidence < min_conf:
            logger.info(
                f"[Validator] Low confidence {sig.confidence:.1f} < {min_conf} "
                f"for {sig.signal_id} — proceeding (advisory)"
            )
        return ValidationResult(SignalStatus.VALID)  # advisory only in v0.1

    def _check_position_conflict(self, sig: SignalContract,
                                  positions: list) -> ValidationResult:
        """Reject if we already have an open position for this symbol."""
        open_symbols = [p.get("symbol", "") for p in positions
                        if p.get("position_state") != "CLOSED"]
        if sig.symbol in open_symbols:
            return ValidationResult(
                SignalStatus.BLOCKED,
                f"Already have open position for {sig.symbol}. "
                f"MAX_OPEN_POSITIONS={self._config.max_open_positions}"
            )
        return ValidationResult(SignalStatus.VALID)

    def _check_order_conflict(self, sig: SignalContract,
                               orders: list) -> ValidationResult:
        """Reject if there's already a pending order for this symbol."""
        pending_symbols = [
            o.get("symbol", "") for o in orders
            if o.get("state") in (
                "SUBMITTING", "UNKNOWN", "ACKNOWLEDGED", "PARTIALLY_FILLED"
            )
        ]
        if sig.symbol in pending_symbols:
            return ValidationResult(
                SignalStatus.BLOCKED,
                f"Already have pending order for {sig.symbol}"
            )
        return ValidationResult(SignalStatus.VALID)
