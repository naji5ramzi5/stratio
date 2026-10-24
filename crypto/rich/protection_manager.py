"""
rich/protection_manager.py — Protection Manager for Bot5.

Responsible for placing and verifying protective orders after entry:
  - Stop Loss (exchange-side OCO or STOP_LOSS_LIMIT order)
  - Take Profit targets (TP1, TP2, TP3)
  - Break Even migration
  - Protection failure detection and emergency policy

CRITICAL RULE:
  If position is OPEN and protection creation fails:
  → State = PROTECTION_FAILURE
  → Kill switch triggers emergency stop for new entries
  → Alert sent immediately via Telegram
  → Position continues to be monitored (not abandoned)
  → Human decision required

IMPORTANT:
  Protection is based on configurable policy:
    stop_loss_type = "exchange" | "local"
  "exchange" = actual STOP_LOSS_LIMIT order sent to Binance
  "local" = monitored locally (less safe but used in PAPER mode)
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Tuple

from .binance_adapter import BinanceAdapter, BinanceAPIError, BinanceNetworkError
from .config import RichConfig, TradingMode
from .db import RichDB
from .models import ProtectionState, RichPosition, TradeSide

logger = logging.getLogger("rich.protection_manager")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProtectionResult:
    def __init__(self, success: bool, reason: str = "", order_id: str = ""):
        self.success = success
        self.reason = reason
        self.order_id = order_id


class ProtectionManager:
    """
    Places and monitors protective orders for open positions.

    In PAPER mode: local tracking only (no exchange orders).
    In LIVE mode: exchange-side stop-loss via STOP_LOSS_LIMIT.
    """

    # Default stop distance: 2% below entry for BUY
    DEFAULT_STOP_DISTANCE_PCT = Decimal("0.02")
    # Default TP1: 1.5% above entry for BUY
    DEFAULT_TP1_DISTANCE_PCT = Decimal("0.015")

    def __init__(self, config: RichConfig, db: RichDB,
                 adapter: Optional[BinanceAdapter] = None,
                 kill_switch=None):
        self._config = config
        self._db = db
        self._adapter = adapter
        self._kill_switch = kill_switch

    def setup_protection(self, position: RichPosition,
                         stop_price: Optional[Decimal] = None,
                         tp1_price: Optional[Decimal] = None,
                         tp2_price: Optional[Decimal] = None,
                         tp3_price: Optional[Decimal] = None) -> RichPosition:
        """
        Set up protection for an open position.

        In LIVE mode: places exchange-side stop-loss order.
        In PAPER mode: records locally.

        Returns the updated position with protection state.
        """
        if not position.average_entry:
            logger.error(f"[ProtMgr] Cannot protect {position.position_id}: no avg_entry")
            return self._handle_failure(position, "No average_entry to compute stop")

        # Calculate stop price if not provided
        if stop_price is None:
            stop_price = self._default_stop(position)

        # Calculate TP1 if not provided
        if tp1_price is None:
            tp1_price = self._default_tp1(position)

        position.stop_loss = stop_price
        position.take_profit_1 = tp1_price
        if tp2_price:
            position.take_profit_2 = tp2_price
        if tp3_price:
            position.take_profit_3 = tp3_price

        position.protection_state = ProtectionState.PENDING

        if self._config.trading_mode in (TradingMode.LIVE,):
            return self._place_exchange_protection(position)
        else:
            # PAPER / SHADOW: local protection tracking
            position.protection_state = ProtectionState.ACTIVE
            logger.info(
                f"[ProtMgr] LOCAL protection set for {position.symbol}: "
                f"SL={stop_price} TP1={tp1_price}"
            )
            self._db.save_position(self._pos_dict(position))
            return position

    def _place_exchange_protection(self, position: RichPosition) -> RichPosition:
        """Place a real stop-loss order on the exchange."""
        if not self._adapter:
            return self._handle_failure(position, "No Binance adapter in LIVE mode")

        # For SPOT BUY position: place a STOP_LOSS_LIMIT SELL order
        # Side is opposite to position side
        exit_side = "SELL" if position.side == TradeSide.BUY else "BUY"
        stop_price = position.stop_loss

        # Limit price slightly below stop to allow fill (1 tick below)
        # In production, this should use tick_size from symbol info
        stop_limit_offset = stop_price * Decimal("0.003")  # 0.3% below stop
        stop_limit_price = stop_price - stop_limit_offset if exit_side == "SELL" else stop_price + stop_limit_offset

        try:
            resp = self._adapter.place_order(
                symbol=position.symbol,
                side=exit_side,
                order_type="STOP_LOSS_LIMIT",
                quantity=position.quantity,
                price=stop_limit_price,
                stop_price=stop_price,
                client_order_id=f"rch_sl_{position.position_id[:12]}",
                time_in_force="GTC",
            )
            position.stop_order_id = str(resp.get("orderId", ""))
            position.protection_state = ProtectionState.ACTIVE
            logger.info(
                f"[ProtMgr] EXCHANGE stop-loss placed: {position.symbol} "
                f"stop={stop_price} limit={stop_limit_price} "
                f"order_id={position.stop_order_id}"
            )
            self._db.save_position(self._pos_dict(position))
            return position

        except (BinanceAPIError, BinanceNetworkError) as e:
            logger.error(f"[ProtMgr] Failed to place exchange stop: {e}")
            return self._handle_failure(position, f"Exchange stop order failed: {e}")

    def _handle_failure(self, position: RichPosition, reason: str) -> RichPosition:
        """
        PROTECTION FAILURE: Position open with no confirmed protection.
        This is a critical state. Do NOT ignore.
        """
        position.protection_state = ProtectionState.FAILED
        self._db.save_position(self._pos_dict(position))
        self._db.log_system_event(
            event_type="PROTECTION_FAILURE",
            message=f"Position {position.position_id} unprotected: {reason}",
            severity="CRITICAL",
            component="protection_manager",
            details={"position_id": position.position_id, "reason": reason},
        )
        if self._kill_switch:
            self._kill_switch.protection_failure(
                f"Protection failed for {position.symbol}: {reason}"
            )
        logger.critical(
            f"[ProtMgr] PROTECTION FAILURE for {position.position_id}: {reason}"
        )
        return position

    def check_protection_status(self, position: RichPosition) -> ProtectionState:
        """
        Verify protection is still active on exchange.
        Called during reconciliation and periodic checks.
        """
        if position.protection_state != ProtectionState.ACTIVE:
            return position.protection_state

        if self._config.trading_mode not in (TradingMode.LIVE,):
            return ProtectionState.ACTIVE  # PAPER: local tracking is sufficient

        if not self._adapter or not position.stop_order_id:
            return ProtectionState.FAILED

        try:
            resp = self._adapter.get_order(
                symbol=position.symbol,
                order_id=position.stop_order_id,
            )
            status = resp.get("status", "")
            if status in ("NEW", "PARTIALLY_FILLED"):
                return ProtectionState.ACTIVE
            elif status == "FILLED":
                position.protection_state = ProtectionState.TRIGGERED
                self._db.save_position(self._pos_dict(position))
                logger.info(f"[ProtMgr] Stop triggered for {position.position_id}")
                return ProtectionState.TRIGGERED
            elif status in ("CANCELED", "CANCELLED", "EXPIRED"):
                logger.error(
                    f"[ProtMgr] Stop order was cancelled/expired for {position.position_id}!"
                )
                return self._handle_failure(
                    position, f"Stop order {position.stop_order_id} is {status}"
                ).protection_state
        except (BinanceAPIError, BinanceNetworkError) as e:
            logger.error(f"[ProtMgr] Cannot verify protection: {e}")
            return ProtectionState.FAILED

        return ProtectionState.UNKNOWN if False else position.protection_state

    def _default_stop(self, position: RichPosition) -> Decimal:
        """Default stop: 2% adverse from entry."""
        if position.side == TradeSide.BUY:
            return position.average_entry * (1 - self.DEFAULT_STOP_DISTANCE_PCT)
        return position.average_entry * (1 + self.DEFAULT_STOP_DISTANCE_PCT)

    def _default_tp1(self, position: RichPosition) -> Decimal:
        """Default TP1: 1.5x risk from entry."""
        if position.side == TradeSide.BUY:
            return position.average_entry * (1 + self.DEFAULT_TP1_DISTANCE_PCT)
        return position.average_entry * (1 - self.DEFAULT_TP1_DISTANCE_PCT)

    def _pos_dict(self, pos: RichPosition) -> dict:
        return {
            "position_id": pos.position_id,
            "signal_id": pos.signal_id,
            "symbol": pos.symbol,
            "side": pos.side.value,
            "quantity": str(pos.quantity),
            "average_entry": str(pos.average_entry) if pos.average_entry else None,
            "current_price": str(pos.current_price) if pos.current_price else None,
            "unrealized_pnl": str(pos.unrealized_pnl),
            "realized_pnl": str(pos.realized_pnl),
            "fees_paid": str(pos.fees_paid),
            "slippage_cost": str(pos.slippage_cost),
            "stop_loss": str(pos.stop_loss) if pos.stop_loss else None,
            "stop_order_id": pos.stop_order_id,
            "take_profit_1": str(pos.take_profit_1) if pos.take_profit_1 else None,
            "take_profit_2": str(pos.take_profit_2) if pos.take_profit_2 else None,
            "take_profit_3": str(pos.take_profit_3) if pos.take_profit_3 else None,
            "tp_order_ids": pos.tp_order_ids,
            "protection_state": pos.protection_state.value,
            "break_even_activated": pos.break_even_activated,
            "trailing_active": pos.trailing_active,
            "position_state": pos.position_state.value,
            "opened_at": pos.opened_at,
            "closed_at": pos.closed_at,
            "exit_reason": pos.exit_reason,
            "gross_pnl": str(pos.gross_pnl),
            "net_pnl": str(pos.net_pnl),
            "mae": str(pos.mae) if pos.mae else None,
            "mfe": str(pos.mfe) if pos.mfe else None,
            "signal_expected_return_pct": pos.signal_expected_return_pct,
            "configuration_version": pos.configuration_version,
            "strategy_version": pos.strategy_version,
        }
