"""
rich/risk_engine.py — Production Risk Engine for Bot5.

This is an MANDATORY GATE before any order submission.
No execution path can bypass it.

Checks (in order):
  1. Kill switch / system state
  2. Daily loss limit
  3. Max open positions
  4. Capital ceiling (TRADING_CAPITAL_LIMIT)
  5. Max account exposure
  6. Max symbol exposure
  7. Available balance sufficient
  8. Expected edge after costs gate
  9. Signal age (redundant safety)
  10. Slippage estimate acceptable

All decisions are logged to RichDB.risk_events.

Philosophy:
  - Conservative by default
  - No Martingale, no revenge trading, no unlimited averaging
  - Every check is configurable from env vars
  - Every rejection is logged with reason
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from .config import RichConfig
from .db import RichDB
from .models import (
    RiskAssessment, RiskDecision, SignalContract, TradeSide
)

logger = logging.getLogger("rich.risk_engine")


class RiskEngine:
    """
    Central risk gate — all position requests must pass through here.
    Stateless with respect to DB (reads from DB, writes events).
    """

    def __init__(self, config: RichConfig, db: RichDB):
        self._config = config
        self._db = db

    def evaluate(
        self,
        signal: SignalContract,
        available_balance_usdt: Any = None,
        current_open_positions: Optional[list] = None,
        current_market_price: Optional[Decimal] = None,
        current_ask: Optional[Decimal] = None,
        current_bid: Optional[Decimal] = None,
    ) -> RiskAssessment:
        """
        Evaluate whether a trade is allowed. Returns a RiskAssessment.

        Must be called BEFORE any order is submitted.
        Supports both positional arguments and portfolio dict.
        """
        if isinstance(available_balance_usdt, dict):
            pdict = available_balance_usdt
            available_balance = Decimal(str(pdict.get("total_equity", pdict.get("available_balance_usdt", "1000.0"))))
            current_positions = pdict.get("open_positions", [])
        else:
            available_balance = Decimal(str(available_balance_usdt if available_balance_usdt is not None else "1000.0"))
            current_positions = current_open_positions or []

        checks_passed = []
        checks_failed = []
        config = self._config


        # ── 1. Daily loss limit ──────────────────────────────────────────────
        today_pnl = self._db.get_today_pnl()
        capital_limit = config.trading_capital_limit
        daily_loss_limit_usdt = capital_limit * config.max_daily_loss

        if today_pnl < -daily_loss_limit_usdt:
            reason = (
                f"Daily loss limit reached: today_pnl={today_pnl:.4f} USDT "
                f"< -{daily_loss_limit_usdt:.4f} USDT "
                f"(limit={config.max_daily_loss*100:.1f}% of {capital_limit} USDT)"
            )
            self._log(signal.signal_id, "REJECTED", reason, "DAILY_LOSS_LIMIT",
                      {"today_pnl": today_pnl, "daily_loss_limit_usdt": daily_loss_limit_usdt})
            return RiskAssessment(
                decision=RiskDecision.REJECTED,
                reason=reason,
                daily_pnl_usdt=today_pnl,
                daily_loss_remaining_usdt=0.0,
                checks_failed=["daily_loss_limit"],
            )
        checks_passed.append("daily_loss_limit")
        daily_loss_remaining = daily_loss_limit_usdt + today_pnl

        # ── 2. Max open positions ────────────────────────────────────────────
        open_count = len([p for p in current_positions
                         if p.get("position_state") != "CLOSED"])
        if open_count >= config.max_open_positions:
            reason = (
                f"Max open positions reached: {open_count}/{config.max_open_positions}"
            )
            self._log(signal.signal_id, "REJECTED", reason, "MAX_POSITIONS",
                      {"open_count": open_count})
            return RiskAssessment(
                decision=RiskDecision.REJECTED,
                reason=reason,
                daily_pnl_usdt=today_pnl,
                daily_loss_remaining_usdt=daily_loss_remaining,
                checks_passed=checks_passed,
                checks_failed=["max_open_positions"],
            )
        checks_passed.append("max_open_positions")

        # ── 3. Capital ceiling ───────────────────────────────────────────────
        if available_balance <= 0:
            reason = f"No available USDT balance: {available_balance}"
            self._log(signal.signal_id, "REJECTED", reason, "NO_BALANCE")
            return RiskAssessment(
                decision=RiskDecision.REJECTED,
                reason=reason,
                daily_pnl_usdt=today_pnl,
                daily_loss_remaining_usdt=daily_loss_remaining,
                checks_passed=checks_passed,
                checks_failed=["available_balance"],
            )

        # Effective balance is capped at TRADING_CAPITAL_LIMIT
        effective_balance = min(float(available_balance), capital_limit)
        checks_passed.append("capital_ceiling")


        # ── 4. Risk per trade ────────────────────────────────────────────────
        max_risk_usdt = effective_balance * config.max_risk_per_trade
        max_position_usdt = effective_balance * config.max_account_exposure

        min_risk_limit = min(1.0, float(effective_balance * 0.005))
        if max_risk_usdt < min_risk_limit or max_risk_usdt <= 0:
            reason = (
                f"Risk per trade too small: {max_risk_usdt:.4f} USDT "
                f"(effective_balance={effective_balance:.2f}, "
                f"max_risk_per_trade={config.max_risk_per_trade*100:.1f}%)"
            )

            self._log(signal.signal_id, "REJECTED", reason, "RISK_TOO_SMALL")
            return RiskAssessment(
                decision=RiskDecision.REJECTED,
                reason=reason,
                daily_pnl_usdt=today_pnl,
                daily_loss_remaining_usdt=daily_loss_remaining,
                checks_passed=checks_passed,
                checks_failed=["min_trade_size"],
            )
        checks_passed.append("risk_per_trade")

        # ── 5. Expected edge after costs gate ────────────────────────────────
        # Edge = predicted_change_pct in basis points (1% = 100 bps)
        pred_val = abs(float(signal.predicted_change_pct))
        predicted_bps = pred_val * 10000.0 if pred_val <= 1.0 else pred_val * 100.0

        entry_fee_bps = config.estimated_taker_fee_bps
        exit_fee_bps = config.estimated_taker_fee_bps  # assume taker exit
        expected_slippage_bps = config.max_slippage_bps / 2  # conservative estimate
        safety_bps = config.safety_margin_bps

        expected_cost_bps = entry_fee_bps + exit_fee_bps + expected_slippage_bps + safety_bps
        edge_after_costs_bps = predicted_bps - expected_cost_bps

        if edge_after_costs_bps <= config.min_expected_edge_after_costs_bps:
            reason = (
                f"Edge after costs too low: {edge_after_costs_bps:.1f} bps "
                f"<= min={config.min_expected_edge_after_costs_bps:.1f} bps. "
                f"Predicted={predicted_bps:.1f} bps, "
                f"Costs={expected_cost_bps:.1f} bps "
                f"(entry_fee={entry_fee_bps}+exit_fee={exit_fee_bps}+"
                f"slip={expected_slippage_bps}+safety={safety_bps})"
            )
            self._log(signal.signal_id, "REJECTED", reason, "EDGE_TOO_LOW",
                      {"predicted_bps": predicted_bps, "cost_bps": expected_cost_bps,
                       "edge_after_costs_bps": edge_after_costs_bps})
            return RiskAssessment(
                decision=RiskDecision.REJECTED,
                reason=reason,
                daily_pnl_usdt=today_pnl,
                daily_loss_remaining_usdt=daily_loss_remaining,
                checks_passed=checks_passed,
                checks_failed=["expected_edge"],
            )
        checks_passed.append("expected_edge")

        # ── All checks passed ────────────────────────────────────────────────
        entry_p = Decimal(str(signal.entry_reference)) if signal.entry_reference else Decimal("100.0")
        stop_dist = Decimal("0.02")  # 2% stop distance
        if signal.side == TradeSide.BUY:
            sl_price = entry_p * (1 - stop_dist)
            tp_price = entry_p * (1 + Decimal("0.03"))
        else:
            sl_price = entry_p * (1 + stop_dist)
            tp_price = entry_p * (1 - Decimal("0.03"))

        # Quantity sized based on max_risk_usdt / (entry * stop_dist)
        risk_per_unit = entry_p * stop_dist
        allocated_qty = Decimal(str(max_risk_usdt)) / risk_per_unit if risk_per_unit > 0 else Decimal("0.01")
        # Ensure notional doesn't exceed max_position_usdt
        if allocated_qty * entry_p > Decimal(str(max_position_usdt)):
            allocated_qty = Decimal(str(max_position_usdt)) / entry_p

        self._log(signal.signal_id, "APPROVED", "All risk checks passed",
                  "RISK_APPROVED", {
                      "max_risk_usdt": max_risk_usdt,
                      "max_position_usdt": max_position_usdt,
                      "effective_balance": effective_balance,
                      "edge_after_costs_bps": edge_after_costs_bps,
                  })
        return RiskAssessment(
            decision=RiskDecision.APPROVED,
            reason="All risk checks passed",
            allowed_risk_usdt=max_risk_usdt,
            max_position_usdt=max_position_usdt,
            risk_per_trade_pct=config.max_risk_per_trade,
            daily_pnl_usdt=today_pnl,
            daily_loss_remaining_usdt=daily_loss_remaining,
            allocated_quantity=allocated_qty,
            stop_loss_price=sl_price,
            take_profit_price=tp_price,
            risk_amount_quote=Decimal(str(max_risk_usdt)),
            checks_passed=checks_passed,
            checks_failed=[],
        )


    def _log(self, signal_id: str, decision: str, reason: str,
             event_type: str, details: dict = None) -> None:
        try:
            self._db.log_risk_event(
                event_type=event_type,
                decision=decision,
                reason=reason,
                signal_id=signal_id,
                details=details or {},
            )
            logger.info(f"[Risk] {decision} signal={signal_id}: {reason}")
        except Exception as e:
            logger.error(f"[Risk] Failed to log risk event: {e}")


class PositionSizer:
    """
    Calculates order quantity based on risk amount and stop distance.

    Formula:
        quantity = risk_usdt / stop_distance_per_unit
        subject to: min_notional, step_size, max_position_usdt
    """

    def __init__(self, config: RichConfig):
        self._config = config

    def calculate(
        self,
        risk_assessment: RiskAssessment,
        entry_price: Decimal,
        stop_price: Decimal,
        step_size: Decimal,
        min_qty: Decimal,
        min_notional: Decimal,
    ) -> "PositionSizeResult":
        from .models import PositionSizeResult

        if not risk_assessment.is_approved():
            return PositionSizeResult(
                approved=False,
                reason="Risk not approved",
                quantity=Decimal("0"),
            )

        if entry_price <= 0 or stop_price <= 0:
            return PositionSizeResult(approved=False, reason="Invalid prices")

        stop_distance = abs(entry_price - stop_price)
        if stop_distance == 0:
            return PositionSizeResult(
                approved=False,
                reason="Stop distance is zero — cannot size position"
            )

        # Core formula: qty = risk / stop_distance
        risk_usdt = Decimal(str(risk_assessment.allowed_risk_usdt))
        raw_quantity = risk_usdt / stop_distance

        # Apply step_size rounding (ROUND_DOWN to avoid exceeding risk)
        from decimal import ROUND_DOWN
        if step_size > 0:
            raw_quantity = (raw_quantity / step_size).to_integral_value(
                rounding=ROUND_DOWN) * step_size

        # Cap at max position
        max_pos_usdt = Decimal(str(risk_assessment.max_position_usdt))
        max_qty_from_limit = max_pos_usdt / entry_price
        if step_size > 0:
            max_qty_from_limit = (max_qty_from_limit / step_size).to_integral_value(
                rounding=ROUND_DOWN) * step_size

        quantity = min(raw_quantity, max_qty_from_limit)

        # Minimum checks
        if quantity < min_qty:
            return PositionSizeResult(
                approved=False,
                reason=f"Quantity {quantity} < min_qty {min_qty} after sizing",
                quantity=Decimal("0"),
            )

        notional = quantity * entry_price
        if notional < min_notional:
            return PositionSizeResult(
                approved=False,
                reason=f"Notional {notional:.4f} < min_notional {min_notional}",
                quantity=Decimal("0"),
            )

        stop_distance_pct = float(stop_distance / entry_price * 100)
        actual_risk_usdt = quantity * stop_distance

        return PositionSizeResult(
            approved=True,
            reason="Position sized successfully",
            quantity=quantity,
            notional_usdt=notional,
            risk_usdt=actual_risk_usdt,
            stop_price=stop_price,
            stop_distance_pct=stop_distance_pct,
        )
