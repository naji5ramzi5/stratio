"""
rich/position_manager.py — Position lifecycle management for Bot5.

Manages open positions:
  - Creation after fill
  - Mark-to-market (price updates)
  - MAE/MFE tracking
  - PNL calculation (using actual fills, not predicted prices)
  - Position close (full or partial)
  - Break-even activation tracking
  - State persistence to SQLite
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from .config import RichConfig
from .db import RichDB
from .models import (
    Fill, OrderState, PositionState, ProtectionState,
    RichOrder, RichPosition, SignalContract, TradeSide
)

logger = logging.getLogger("rich.position_manager")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PositionManager:
    """Manages the lifecycle of positions from entry fill to close."""

    def __init__(self, config: RichConfig, db: RichDB):
        self._config = config
        self._db = db

    def open_position(
        self,
        signal: SignalContract,
        entry_order: RichOrder,
        fills: list,  # list of Fill objects
    ) -> RichPosition:
        """
        Create a new position from a filled entry order.

        PNL is always calculated from ACTUAL fills — never from predicted price.
        """
        if not fills:
            raise ValueError("Cannot open position without fills")

        # Weighted average entry price from actual fills
        total_qty = Decimal("0")
        total_cost = Decimal("0")
        total_commission = Decimal("0")

        for f in fills:
            qty = Decimal(str(f.quantity)) if hasattr(f, 'quantity') else Decimal(str(f.get("quantity", "0")))
            price = Decimal(str(f.price)) if hasattr(f, 'price') else Decimal(str(f.get("price", "0")))
            comm = Decimal(str(f.commission)) if hasattr(f, 'commission') else Decimal(str(f.get("commission", "0")))
            total_qty += qty
            total_cost += qty * price
            total_commission += comm

        avg_entry = total_cost / total_qty if total_qty > 0 else Decimal("0")

        # Slippage vs signal reference
        ref_price = Decimal(str(signal.entry_reference)) if signal.entry_reference else avg_entry
        if signal.side == TradeSide.BUY:
            slippage = (avg_entry - ref_price) * total_qty
        else:
            slippage = (ref_price - avg_entry) * total_qty

        position = RichPosition(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            side=signal.side,
            quantity=total_qty,
            average_entry=avg_entry,
            current_price=avg_entry,
            fees_paid=total_commission,
            slippage_cost=slippage,
            entry_order_id=entry_order.internal_order_id,
            position_state=PositionState.OPEN,
            opened_at=_now_iso(),
            protection_state=ProtectionState.NONE,
            signal_expected_return_pct=float(signal.predicted_change_pct),
            configuration_version=self._config.configuration_version,
            strategy_version=signal.strategy_version,
            mae=Decimal("0"),
            mfe=Decimal("0"),
        )

        self._db.save_position(self._pos_to_dict(position))
        logger.info(
            f"[PositionMgr] Opened {signal.symbol} {signal.side.value} "
            f"qty={total_qty} avg_entry={avg_entry} fees={total_commission}"
        )
        return position

    def update_price(self, position: RichPosition, current_price: Decimal) -> RichPosition:
        """Update mark-to-market price and recalculate unrealized PNL + MAE/MFE."""
        position.current_price = current_price

        if position.average_entry:
            if position.side == TradeSide.BUY:
                position.unrealized_pnl = (current_price - position.average_entry) * position.quantity
            else:
                position.unrealized_pnl = (position.average_entry - current_price) * position.quantity

        # Track MAE (Maximum Adverse Excursion)
        if position.unrealized_pnl < 0:
            if position.mae is None or abs(position.unrealized_pnl) > abs(position.mae):
                position.mae = position.unrealized_pnl

        # Track MFE (Maximum Favorable Excursion)
        if position.unrealized_pnl > 0:
            if position.mfe is None or position.unrealized_pnl > position.mfe:
                position.mfe = position.unrealized_pnl

        self._db.save_position(self._pos_to_dict(position))
        return position

    def close_position(
        self,
        position: RichPosition,
        exit_fills: list,
        exit_reason: str,
    ) -> RichPosition:
        """
        Close a position (fully) using actual exit fills.

        Calculates:
          gross_pnl = entry_exit_price_difference * quantity
          fees = entry_fees + exit_fees
          net_pnl = gross_pnl - fees - slippage
        """
        if not exit_fills:
            raise ValueError("Cannot close position without exit fills")

        total_exit_qty = Decimal("0")
        total_exit_cost = Decimal("0")
        total_exit_commission = Decimal("0")

        for f in exit_fills:
            qty = Decimal(str(f.quantity)) if hasattr(f, 'quantity') else Decimal(str(f.get("quantity", "0")))
            price = Decimal(str(f.price)) if hasattr(f, 'price') else Decimal(str(f.get("price", "0")))
            comm = Decimal(str(f.commission)) if hasattr(f, 'commission') else Decimal(str(f.get("commission", "0")))
            total_exit_qty += qty
            total_exit_cost += qty * price
            total_exit_commission += comm

        avg_exit = total_exit_cost / total_exit_qty if total_exit_qty > 0 else Decimal("0")

        # Gross PNL
        if position.side == TradeSide.BUY and position.average_entry:
            gross_pnl = (avg_exit - position.average_entry) * total_exit_qty
        elif position.side == TradeSide.SELL and position.average_entry:
            gross_pnl = (position.average_entry - avg_exit) * total_exit_qty
        else:
            gross_pnl = Decimal("0")

        total_fees = position.fees_paid + total_exit_commission
        net_pnl = gross_pnl - total_fees - position.slippage_cost

        position.realized_pnl = gross_pnl
        position.fees_paid = total_fees
        position.gross_pnl = gross_pnl
        position.net_pnl = net_pnl
        position.unrealized_pnl = Decimal("0")
        position.current_price = avg_exit
        position.position_state = PositionState.CLOSED
        position.closed_at = _now_iso()
        position.exit_reason = exit_reason

        self._db.save_position(self._pos_to_dict(position))

        # Write trade journal
        self._write_trade_journal(position, signal_id=position.signal_id)

        logger.info(
            f"[PositionMgr] Closed {position.symbol} {position.side.value} "
            f"gross_pnl={gross_pnl:.4f} fees={total_fees:.4f} "
            f"net_pnl={net_pnl:.4f} reason={exit_reason}"
        )
        return position

    def activate_break_even(self, position: RichPosition) -> RichPosition:
        """Mark position as break-even (SL moved to cost basis including fees)."""
        if not position.average_entry:
            return position
        fees_per_unit = position.fees_paid / position.quantity if position.quantity > 0 else Decimal("0")
        break_even_price = position.average_entry + fees_per_unit
        position.stop_loss = break_even_price
        position.break_even_activated = True
        self._db.save_position(self._pos_to_dict(position))
        logger.info(
            f"[PositionMgr] Break-even activated for {position.position_id}: "
            f"SL={break_even_price}"
        )
        return position

    def get_open_positions(self) -> list:
        return self._db.get_open_positions()

    def _pos_to_dict(self, pos: RichPosition) -> dict:
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

    def _write_trade_journal(self, pos: RichPosition, signal_id: str) -> None:
        trade_id = "tj_" + pos.position_id
        self._db.save_trade_journal({
            "trade_id": trade_id,
            "signal_id": signal_id,
            "signal_version": "",
            "strategy_version": pos.strategy_version,
            "configuration_version": pos.configuration_version,
            "symbol": pos.symbol,
            "side": pos.side.value,
            "signal_price": str(pos.average_entry),
            "actual_entry": str(pos.average_entry),
            "executed_quantity": str(pos.quantity),
            "average_fill_price": str(pos.average_entry),
            "stop_loss": str(pos.stop_loss) if pos.stop_loss else None,
            "take_profit_1": str(pos.take_profit_1) if pos.take_profit_1 else None,
            "fees": str(pos.fees_paid),
            "slippage_cost": str(pos.slippage_cost),
            "gross_pnl": str(pos.gross_pnl),
            "net_pnl": str(pos.net_pnl),
            "signal_expected_return_pct": pos.signal_expected_return_pct,
            "opened_at": pos.opened_at,
            "closed_at": pos.closed_at,
            "holding_seconds": pos.holding_seconds(),
            "mae": str(pos.mae) if pos.mae else None,
            "mfe": str(pos.mfe) if pos.mfe else None,
            "exit_reason": pos.exit_reason,
        })
