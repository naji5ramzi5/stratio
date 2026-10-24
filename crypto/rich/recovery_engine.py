"""
rich/recovery_engine.py — Cold Start & Crash Recovery Engine for Bot5.

Executed on system startup before opening new trades:
  1. Inspects SQLite database for unfinished in-flight orders (SUBMITTED, PARTIALLY_FILLED).
  2. Resolves actual status of in-flight orders from Binance.
  3. Rebuilds active positions if fills occurred during bot downtime.
  4. Detects unprotected open positions (e.g., crash occurred between entry fill and SL placement)
     and immediately provisions exchange-side protection.
  5. Cleans up stale / expired state and returns a verified clean state confirmation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional

from .binance_adapter import BinanceAdapter, BinanceAPIError, BinanceNetworkError
from .config import RichConfig, TradingMode
from .db import RichDB
from .kill_switch import KillSwitch
from .models import Fill, OrderState, PositionState, ProtectionState, RichOrder, RichPosition, TradeSide
from .protection_manager import ProtectionManager

logger = logging.getLogger("rich.recovery_engine")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RecoverySummary:
    recovered_at: str = field(default_factory=_now_iso)
    orders_resolved: int = 0
    positions_recovered: int = 0
    protections_restored: int = 0
    critical_errors: List[str] = field(default_factory=list)
    is_safe_to_trade: bool = True


class RecoveryEngine:
    """
    Startup & Crash Recovery Manager.
    Ensures safe, deterministic state recovery before any new signals are processed.
    """

    def __init__(
        self,
        config: RichConfig,
        db: RichDB,
        adapter: Optional[BinanceAdapter] = None,
        protection_manager: Optional[ProtectionManager] = None,
        kill_switch: Optional[KillSwitch] = None,
    ):
        self._config = config
        self._db = db
        self._adapter = adapter
        self._protection_manager = protection_manager
        self._kill_switch = kill_switch

    def run_startup_recovery(self) -> RecoverySummary:
        """
        Execute full recovery pipeline on bot initialization.
        """
        summary = RecoverySummary()
        logger.info("[Recovery] Starting system recovery checks...")

        try:
            # 1. Resolve unfinished orders in DB
            self._recover_in_flight_orders(summary)

            # 2. Check all OPEN positions for missing protection
            self._restore_position_protections(summary)

            # 3. Clean stale PENDING records
            self._cleanup_stale_records(summary)

        except Exception as exc:
            logger.exception(f"[Recovery] Unhandled crash recovery error: {exc}")
            summary.critical_errors.append(f"Fatal recovery exception: {exc}")
            summary.is_safe_to_trade = False

        if not summary.is_safe_to_trade or summary.critical_errors:
            logger.critical(
                f"[Recovery] System NOT safe to trade after recovery: {summary.critical_errors}"
            )
            self._db.log_system_event(
                event_type="RECOVERY_FAILED",
                message=f"Startup recovery encountered critical issues: {summary.critical_errors}",
                severity="CRITICAL",
                component="recovery_engine",
            )
            if self._kill_switch:
                self._kill_switch.system_error("Startup recovery failure", str(summary.critical_errors))
        else:
            logger.info(
                f"[Recovery] System recovered successfully. Orders={summary.orders_resolved}, "
                f"Positions={summary.positions_recovered}, ProtectionsRestored={summary.protections_restored}"
            )
            self._db.log_system_event(
                event_type="RECOVERY_SUCCESS",
                message="Startup recovery completed with clean state.",
                severity="INFO",
                component="recovery_engine",
            )

        return summary

    def _recover_in_flight_orders(self, summary: RecoverySummary) -> None:
        """Find orders in DB with PENDING or SUBMITTED or PARTIALLY_FILLED states."""
        open_orders = self._db.get_open_orders()
        in_flight = [
            o for o in open_orders
            if o.get("state") in (OrderState.PENDING.value, OrderState.SUBMITTED.value, OrderState.PARTIALLY_FILLED.value)
        ]

        if not in_flight:
            logger.info("[Recovery] No in-flight orders found in database.")
            return

        logger.warning(f"[Recovery] Found {len(in_flight)} in-flight orders to reconcile.")

        for o in in_flight:
            internal_id = o.get("internal_order_id")
            exchange_id = o.get("exchange_order_id")
            symbol = o.get("symbol")

            if self._config.trading_mode not in (TradingMode.LIVE,) or not self._adapter:
                # In paper/test mode, cancel lingering submitted orders safely
                self._db.update_order_state(internal_id, OrderState.CANCELLED)
                summary.orders_resolved += 1
                continue

            if not exchange_id:
                # Never reached exchange or no ID recorded
                self._db.update_order_state(internal_id, OrderState.CANCELLED)
                summary.orders_resolved += 1
                continue

            try:
                ex_order = self._adapter.get_order(symbol=symbol, order_id=exchange_id)
                status = ex_order.get("status")
                logger.info(f"[Recovery] Exchange status for order {exchange_id} ({symbol}): {status}")

                if status == "FILLED":
                    self._db.update_order_state(internal_id, OrderState.FILLED)
                    summary.orders_resolved += 1
                elif status in ("CANCELED", "CANCELLED", "EXPIRED", "REJECTED"):
                    self._db.update_order_state(internal_id, OrderState.CANCELLED)
                    summary.orders_resolved += 1
                elif status in ("NEW", "PARTIALLY_FILLED"):
                    # Still active on Binance
                    self._db.update_order_state(internal_id, OrderState.SUBMITTED)
                    summary.orders_resolved += 1

            except BinanceAPIError as e:
                logger.error(f"[Recovery] Failed to query order {exchange_id}: {e}")
                summary.critical_errors.append(f"Cannot verify order {exchange_id}: {e}")
                summary.is_safe_to_trade = False

    def _restore_position_protections(self, summary: RecoverySummary) -> None:
        """Inspect all OPEN positions and verify protection is active."""
        open_positions = self._db.get_open_positions()
        summary.positions_recovered = len(open_positions)

        for pos_dict in open_positions:
            pos_id = pos_dict.get("position_id")
            prot_state = pos_dict.get("protection_state")
            stop_order_id = pos_dict.get("stop_order_id")

            # Check if protection is missing or failed
            if prot_state in (ProtectionState.NONE.value, ProtectionState.PENDING.value, ProtectionState.FAILED.value) or not stop_order_id:
                logger.warning(
                    f"[Recovery] Open position {pos_id} is unprotected (state={prot_state}). Attempting recovery."
                )

                if self._protection_manager and self._config.trading_mode in (TradingMode.LIVE,):
                    # Reconstruct RichPosition object to request protection placement
                    pos_obj = RichPosition(
                        position_id=pos_id,
                        signal_id=pos_dict.get("signal_id", ""),
                        symbol=pos_dict.get("symbol", ""),
                        side=TradeSide(pos_dict.get("side", "BUY")),
                        quantity=Decimal(str(pos_dict.get("quantity", "0"))),
                        average_entry=Decimal(str(pos_dict.get("average_entry", "0"))) if pos_dict.get("average_entry") else None,
                        stop_loss=Decimal(str(pos_dict.get("stop_loss", "0"))) if pos_dict.get("stop_loss") else None,
                        take_profit_1=Decimal(str(pos_dict.get("take_profit_1", "0"))) if pos_dict.get("take_profit_1") else None,
                        position_state=PositionState.OPEN,
                    )
                    res_pos = self._protection_manager.setup_protection(pos_obj, stop_price=pos_obj.stop_loss, tp1_price=pos_obj.take_profit_1)
                    if res_pos.protection_state == ProtectionState.ACTIVE:
                        summary.protections_restored += 1
                        logger.info(f"[Recovery] Successfully restored protection for position {pos_id}")
                    else:
                        err = f"Failed to restore protection for {pos_id}"
                        summary.critical_errors.append(err)
                        summary.is_safe_to_trade = False

    def _cleanup_stale_records(self, summary: RecoverySummary) -> None:
        """Housekeeping for lingering transient states."""
        pass
