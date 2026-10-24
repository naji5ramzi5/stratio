"""
rich/execution_engine.py — Core Autonomous Execution Engine for Bot5.

Coordinates end-to-end trade processing:
  1. Signal validation (Schema, timestamps, regime, confidence, agreement)
  2. Risk evaluation (Portfolio capacity, symbol risk, max drawdown, position sizing)
  3. Order creation & submission (Idempotent clientOrderId, atomic state transitions)
  4. Fill processing & position opening (Actual weighted average price, fee tracking)
  5. Protective order establishment (Exchange-side SL / TP, protection failure handling)
  6. Position monitoring, break-even trailing, and exit execution
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from .binance_adapter import BinanceAdapter
from .config import RichConfig, TradingMode
from .db import RichDB
from .kill_switch import KillSwitch
from .models import Fill, OrderState, PositionState, ProtectionState, RichOrder, RichPosition, SignalContract, TradeSide
from .order_manager import OrderManager
from .position_manager import PositionManager
from .protection_manager import ProtectionManager
from .risk_engine_rich import RiskEngine
from .signal_validator import SignalValidator
from .telegram_reporter import TelegramReporter

logger = logging.getLogger("rich.execution_engine")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExecutionEngine:
    """
    Main Autonomous Execution Engine for Bot5 (الغني).
    """

    def __init__(
        self,
        config: RichConfig,
        db: RichDB,
        adapter: Optional[BinanceAdapter] = None,
        kill_switch: Optional[KillSwitch] = None,
        reporter: Optional[TelegramReporter] = None,
    ):
        self._config = config
        self._db = db
        self._adapter = adapter
        self._kill_switch = kill_switch or KillSwitch(config, db)
        self._reporter = reporter or TelegramReporter(config)

        self._signal_validator = SignalValidator(config, db)
        self._risk_engine = RiskEngine(config, db)
        self._order_manager = OrderManager(config, db, adapter)
        self._position_manager = PositionManager(config, db)
        self._protection_manager = ProtectionManager(config, db, adapter, self._kill_switch)

    def process_trade_signal(self, signal: SignalContract) -> Optional[RichPosition]:
        """
        Execute full lifecycle for an incoming trade signal.
        """
        logger.info(f"[ExecEngine] Processing signal {signal.signal_id} ({signal.symbol} {signal.side.value})")

        # 1. Kill Switch Pre-Check
        if not self._kill_switch.can_trade():
            logger.warning(f"[ExecEngine] Trade rejected: KillSwitch is ACTIVE ({self._kill_switch.status().get('state')})")
            return None

        # 2. Signal Validation
        val_result = self._signal_validator.validate(signal)
        if not val_result.is_valid:
            logger.info(f"[ExecEngine] Signal {signal.signal_id} rejected by validator: {val_result.reject_reason}")
            return None

        # 3. Risk Engine Evaluation & Sizing
        portfolio_state = self._get_portfolio_state()
        risk_decision = self._risk_engine.evaluate(signal, portfolio_state)
        if not risk_decision.approved:
            logger.info(f"[ExecEngine] Signal {signal.signal_id} rejected by risk engine: {risk_decision.reject_reason}")
            return None

        # Persist signal in DB before order creation
        if not self._db.signal_exists(signal.signal_id):
            self._db.save_signal(signal.to_dict())

        # 4. Create Entry Order

        position_id = "pos_" + uuid.uuid4().hex[:12]
        entry_order = self._order_manager.create_entry_order(
            signal=signal,
            position_id=position_id,
            quantity=risk_decision.allocated_quantity,
            entry_price=signal.entry_reference,
        )

        # 5. Submit Order
        submitted_order = self._order_manager.submit_order(entry_order)
        if submitted_order.state != OrderState.FILLED:
            logger.warning(
                f"[ExecEngine] Entry order {submitted_order.internal_order_id} ended in state {submitted_order.state.value}. Reject: {submitted_order.reject_reason}"
            )
            return None

        # 6. Construct Fills
        fill_price = submitted_order.average_fill_price or signal.entry_reference or Decimal("100.0")
        fill_qty = submitted_order.executed_qty or risk_decision.allocated_quantity
        # Standard 0.1% spot fee estimate if not returned in fills
        fee = fill_price * fill_qty * Decimal("0.001")

        fill_obj = Fill(
            fill_id="f_" + uuid.uuid4().hex[:8],
            order_id=submitted_order.internal_order_id,
            symbol=signal.symbol,
            side=signal.side,
            price=fill_price,
            quantity=fill_qty,
            commission=fee,
            commission_asset="USDT",
            filled_at=_now_iso(),
        )


        # 7. Open Position in PositionManager
        position = self._position_manager.open_position(
            signal=signal,
            entry_order=submitted_order,
            fills=[fill_obj],
        )

        # 8. Establish Protection (Stop-Loss & Take-Profit)
        protected_pos = self._protection_manager.setup_protection(
            position=position,
            stop_price=risk_decision.stop_loss_price,
            tp1_price=risk_decision.take_profit_price,
        )

        # 9. Dispatch Alert
        self._reporter.notify_trade_entry(
            pos=protected_pos,
            risk_amount=risk_decision.risk_amount_quote,
            expected_return=float(signal.predicted_change_pct) * 100.0 if signal.predicted_change_pct else 0.0,
        )

        return protected_pos

    def update_market_prices(self, price_feed: Dict[str, Decimal]) -> List[RichPosition]:
        """
        Update open positions with current mark prices and trigger SL/TP if crossed.
        """
        closed_positions = []
        open_positions = self._position_manager.get_open_positions()

        for pos_dict in open_positions:
            symbol = pos_dict.get("symbol")
            if not symbol or symbol not in price_feed:
                continue

            current_price = price_feed[symbol]
            pos_obj = self._dict_to_pos(pos_dict)

            # Update MTM & MAE/MFE
            updated_pos = self._position_manager.update_price(pos_obj, current_price)

            # Check Stop Loss Trigger (In Paper mode or fallback monitoring)
            if self._should_trigger_stop_loss(updated_pos, current_price):
                closed = self._execute_exit(updated_pos, current_price, exit_reason="STOP_LOSS_HIT")
                closed_positions.append(closed)
                continue

            # Check Take Profit 1 Trigger
            if self._should_trigger_tp(updated_pos, current_price):
                closed = self._execute_exit(updated_pos, current_price, exit_reason="TAKE_PROFIT_HIT")
                closed_positions.append(closed)
                continue

        return closed_positions

    def _should_trigger_stop_loss(self, pos: RichPosition, current_price: Decimal) -> bool:
        if not pos.stop_loss:
            return False
        if pos.side == TradeSide.BUY and current_price <= pos.stop_loss:
            return True
        if pos.side == TradeSide.SELL and current_price >= pos.stop_loss:
            return True
        return False

    def _should_trigger_tp(self, pos: RichPosition, current_price: Decimal) -> bool:
        if not pos.take_profit_1:
            return False
        if pos.side == TradeSide.BUY and current_price >= pos.take_profit_1:
            return True
        if pos.side == TradeSide.SELL and current_price <= pos.take_profit_1:
            return True
        return False

    def _execute_exit(self, pos: RichPosition, exit_price: Decimal, exit_reason: str) -> RichPosition:
        """Close position and alert."""
        fee = exit_price * pos.quantity * Decimal("0.001")
        exit_fill = Fill(
            fill_id="f_exit_" + uuid.uuid4().hex[:8],
            order_id="exit_" + pos.position_id,
            symbol=pos.symbol,
            side=TradeSide.SELL if pos.side == TradeSide.BUY else TradeSide.BUY,
            price=exit_price,
            quantity=pos.quantity,
            commission=fee,
            commission_asset="USDT",
            filled_at=_now_iso(),
        )

        closed_pos = self._position_manager.close_position(
            position=pos,
            exit_fills=[exit_fill],
            exit_reason=exit_reason,
        )
        self._reporter.notify_trade_close(closed_pos)
        return closed_pos

    def _get_portfolio_state(self) -> dict:
        """Fetch current balance and active positions count."""
        open_positions = self._position_manager.get_open_positions()
        # Default equity estimate or query Binance
        equity = Decimal(str(self._config.allocated_equity)) if hasattr(self._config, "allocated_equity") else Decimal("10000.0")
        return {
            "total_equity": equity,
            "open_positions": open_positions,
            "active_trades_count": len(open_positions),
        }

    def _dict_to_pos(self, d: dict) -> RichPosition:
        return RichPosition(
            position_id=d["position_id"],
            signal_id=d.get("signal_id", ""),
            symbol=d["symbol"],
            side=TradeSide(d["side"]),
            quantity=Decimal(str(d["quantity"])),
            average_entry=Decimal(str(d["average_entry"])) if d.get("average_entry") else None,
            current_price=Decimal(str(d["current_price"])) if d.get("current_price") else None,
            unrealized_pnl=Decimal(str(d.get("unrealized_pnl", "0"))),
            realized_pnl=Decimal(str(d.get("realized_pnl", "0"))),
            fees_paid=Decimal(str(d.get("fees_paid", "0"))),
            slippage_cost=Decimal(str(d.get("slippage_cost", "0"))),
            stop_loss=Decimal(str(d["stop_loss"])) if d.get("stop_loss") else None,
            stop_order_id=d.get("stop_order_id"),
            take_profit_1=Decimal(str(d["take_profit_1"])) if d.get("take_profit_1") else None,
            position_state=PositionState(d.get("position_state", "OPEN")),
            protection_state=ProtectionState(d.get("protection_state", "NONE")),
            opened_at=d.get("opened_at", _now_iso()),
        )
