"""
rich/order_manager.py — Order lifecycle management for Bot5.

Manages:
  - Order creation and persistence
  - State machine transitions
  - Unknown-state handling (CRITICAL: network timeout after submission)
  - Partial fill tracking
  - Fill persistence
  - clientOrderId generation (unique, deterministic, idempotency-safe)

CRITICAL RULE:
  If order was submitted and we got no response (network timeout),
  the order state is UNKNOWN — NOT failed.
  We MUST query by clientOrderId before taking any action.
  We MUST NOT submit another order.
"""
from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Optional

from .binance_adapter import BinanceAdapter, BinanceAPIError, BinanceNetworkError
from .config import RichConfig, TradingMode
from .db import RichDB
from .models import (
    Fill, OrderState, RichOrder, SignalContract, TradeSide
)

logger = logging.getLogger("rich.order_manager")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_client_order_id(signal_id: str, suffix: str = "") -> str:
    """
    Generate a deterministic, unique clientOrderId.

    Format: rch_{signal_id_hash8}_{suffix_or_ts8}
    Binance limit: max 36 chars.
    """
    sig_hash = hashlib.sha256(signal_id.encode()).hexdigest()[:8]
    if suffix:
        tag = suffix[:8]
    else:
        # Use current timestamp hex for uniqueness
        tag = format(int(time.time() * 1000) & 0xFFFFFFFF, "08x")
    raw = f"rch_{sig_hash}_{tag}"
    return raw[:36]


class OrderManager:
    """
    Manages the full lifecycle of every order in Bot5.

    State Machine:
        SIGNAL_RECEIVED → VALIDATING → APPROVED → SUBMITTING
        → (network ok) → ACKNOWLEDGED → PARTIALLY_FILLED → FILLED
        → (network timeout) → UNKNOWN → (query) → ACKNOWLEDGED or FILLED or NOT_FOUND
        → REJECTED / CANCELLED / EXPIRED (terminal errors)
    """

    def __init__(self, config: RichConfig, db: RichDB,
                 adapter: Optional[BinanceAdapter] = None):
        self._config = config
        self._db = db
        self._adapter = adapter  # None in PAPER mode

    # ── Order Creation ────────────────────────────────────────────────────────

    def create_entry_order(self, signal: SignalContract,
                           position_id: str,
                           quantity: Decimal,
                           entry_price: Optional[Decimal] = None) -> RichOrder:
        """
        Create and persist an entry order (before submission).
        Returns the RichOrder ready for submission.
        """
        client_order_id = _generate_client_order_id(signal.signal_id, "entry")
        now = _now_iso()

        order = RichOrder(
            client_order_id=client_order_id,
            signal_id=signal.signal_id,
            position_id=position_id,
            symbol=signal.symbol,
            side=signal.side,
            order_type=self._config.entry_order_type,
            requested_qty=quantity,
            requested_price=entry_price,
            state=OrderState.APPROVED,
            created_at=now,
            updated_at=now,
        )

        self._db.save_order(order.to_dict())
        self._db.update_order_state(
            order.internal_order_id, OrderState.APPROVED.value,
            {"signal_id": signal.signal_id, "qty": str(quantity)}
        )
        logger.info(
            f"[OrderMgr] Created entry order {order.internal_order_id} "
            f"client_id={client_order_id} for signal={signal.signal_id} "
            f"symbol={signal.symbol} qty={quantity}"
        )
        return order

    # ── Order Submission ──────────────────────────────────────────────────────

    def submit_order(self, order: RichOrder) -> RichOrder:
        """
        Submit an order to the exchange (or paper broker).

        CRITICAL PROTOCOL:
          1. Set state = SUBMITTING (persisted) BEFORE sending to exchange
          2. Send to exchange
          3. On success → ACKNOWLEDGED
          4. On network timeout → UNKNOWN (query required)
          5. On API error → REJECTED or classify
          6. NEVER submit twice without querying first if state is UNKNOWN
        """
        if self._config.trading_mode == TradingMode.PAPER:
            return self._submit_paper(order)
        if self._config.trading_mode == TradingMode.SHADOW:
            return self._submit_shadow(order)
        if self._config.trading_mode == TradingMode.LIVE_LOCKED:
            logger.warning(
                f"[OrderMgr] LIVE_LOCKED: would submit {order.client_order_id} "
                "but execution is locked. Dry-run only."
            )
            order.state = OrderState.REJECTED
            order.reject_reason = "LIVE_LOCKED mode — no real orders allowed"
            self._db.update_order_state(
                order.internal_order_id, OrderState.REJECTED.value,
                {"reason": order.reject_reason}
            )
            return order

        # LIVE mode
        return self._submit_live(order)

    def _submit_live(self, order: RichOrder) -> RichOrder:
        """Submit to real Binance API."""
        if not self._adapter:
            raise RuntimeError("BinanceAdapter not configured for LIVE mode")

        # Step 1: Mark SUBMITTING in DB BEFORE network call
        order.state = OrderState.SUBMITTING
        order.submitted_at = _now_iso()
        self._db.update_order_state(
            order.internal_order_id, OrderState.SUBMITTING.value,
            {"client_order_id": order.client_order_id}
        )
        self._db.save_order(order.to_dict())

        try:
            resp = self._adapter.place_order(
                symbol=order.symbol,
                side=order.side.value,
                order_type=order.order_type,
                quantity=order.requested_qty,
                price=order.requested_price,
                client_order_id=order.client_order_id,
            )
            # Success
            order.exchange_order_id = str(resp.get("orderId", ""))
            order.raw_response = resp

            ex_status = resp.get("status", "")
            if ex_status == "FILLED":
                order.state = OrderState.FILLED
                order.executed_qty = Decimal(str(resp.get("executedQty", "0")))
                order.average_fill_price = Decimal(str(resp.get("cummulativeQuoteQty", "0"))) / order.executed_qty if order.executed_qty else None
                order.filled_at = _now_iso()
            elif ex_status == "PARTIALLY_FILLED":
                order.state = OrderState.PARTIALLY_FILLED
                order.executed_qty = Decimal(str(resp.get("executedQty", "0")))
            elif ex_status in ("NEW", "ACCEPTED"):
                order.state = OrderState.ACKNOWLEDGED
            else:
                order.state = OrderState.ACKNOWLEDGED

            self._db.update_order_state(
                order.internal_order_id, order.state.value,
                {"exchange_order_id": order.exchange_order_id, "status": ex_status}
            )
            self._db.save_order(order.to_dict())
            logger.info(
                f"[OrderMgr] Order submitted: exchange_id={order.exchange_order_id} "
                f"status={ex_status} state={order.state.value}"
            )

        except BinanceNetworkError as e:
            # CRITICAL: network timeout — state is UNKNOWN
            # DO NOT submit again. Query first.
            order.state = OrderState.UNKNOWN
            self._db.update_order_state(
                order.internal_order_id, OrderState.UNKNOWN.value,
                {"error": str(e), "client_order_id": order.client_order_id}
            )
            logger.error(
                f"[OrderMgr] NETWORK ERROR submitting {order.client_order_id}: {e}. "
                "State = UNKNOWN. Must reconcile before any retry."
            )

        except BinanceAPIError as e:
            error_class = self._classify_api_error(e.code)
            order.state = OrderState.REJECTED
            order.reject_reason = f"BinanceError {e.code}: {e.msg} [{error_class}]"
            self._db.update_order_state(
                order.internal_order_id, OrderState.REJECTED.value,
                {"error_code": e.code, "error_msg": e.msg, "class": error_class}
            )
            self._db.save_order(order.to_dict())
            logger.error(
                f"[OrderMgr] Order REJECTED {order.client_order_id}: "
                f"code={e.code} msg={e.msg} class={error_class}"
            )

        return order

    def _submit_paper(self, order: RichOrder) -> RichOrder:
        """Simulate fill in PAPER mode."""
        order.state = OrderState.SUBMITTING
        order.submitted_at = _now_iso()

        # Paper: immediate fill at reference price with simulated slippage
        slippage_factor = Decimal("1.001")  # 10 bps slippage
        if order.requested_price is not None:
            req_p = Decimal(str(order.requested_price))
            fill_price = req_p * slippage_factor
        else:
            # MARKET order: default fallback
            fill_price = Decimal("100.0")


        order.executed_qty = order.requested_qty
        order.average_fill_price = fill_price
        order.exchange_order_id = f"PAPER_{order.client_order_id}"
        order.filled_at = _now_iso()
        order.state = OrderState.FILLED

        self._db.update_order_state(
            order.internal_order_id, OrderState.FILLED.value,
            {"paper_fill_price": str(fill_price), "mode": "PAPER"}
        )
        self._db.save_order(order.to_dict())

        # Save fill record
        fill = Fill(
            order_id=order.internal_order_id,
            exchange_trade_id=f"PAPER_{int(time.time() * 1000)}",
            symbol=order.symbol,
            side=order.side,
            quantity=order.executed_qty,
            price=fill_price,
            commission=fill_price * order.executed_qty * Decimal("0.001"),
            commission_asset="USDT",
            is_maker=False,
            filled_at=order.filled_at,
        )
        self._db.save_fill({
            "fill_id": fill.fill_id,
            "order_id": fill.order_id,
            "exchange_trade_id": fill.exchange_trade_id,
            "symbol": fill.symbol,
            "side": fill.side.value,
            "quantity": str(fill.quantity),
            "price": str(fill.price),
            "commission": str(fill.commission),
            "commission_asset": fill.commission_asset,
            "is_maker": fill.is_maker,
            "filled_at": fill.filled_at,
        })

        logger.info(
            f"[OrderMgr] PAPER fill: {order.symbol} {order.side.value} "
            f"qty={order.executed_qty} @ {fill_price}"
        )
        return order

    def _submit_shadow(self, order: RichOrder) -> RichOrder:
        """Shadow mode: compute virtual fill but do not send to exchange."""
        logger.info(f"[OrderMgr] SHADOW: would submit {order.client_order_id} (not sent)")
        order.state = OrderState.REJECTED
        order.reject_reason = "SHADOW mode — virtual only"
        self._db.update_order_state(
            order.internal_order_id, OrderState.REJECTED.value,
            {"reason": "shadow_mode"}
        )
        return order

    # ── Unknown State Recovery ────────────────────────────────────────────────

    def recover_unknown_order(self, order: RichOrder) -> RichOrder:
        """
        Query Binance for an order in UNKNOWN state.

        MUST be called before any retry or giving up.
        Never submits a duplicate.
        """
        if not self._adapter:
            logger.warning(f"[OrderMgr] Cannot recover UNKNOWN order {order.client_order_id} — no adapter")
            return order

        logger.info(f"[OrderMgr] Querying UNKNOWN order by client_id={order.client_order_id}")
        try:
            resp = self._adapter.get_order(
                symbol=order.symbol,
                client_order_id=order.client_order_id,
            )
            status = resp.get("status", "")
            order.exchange_order_id = str(resp.get("orderId", ""))
            order.executed_qty = Decimal(str(resp.get("executedQty", "0")))

            if status == "FILLED":
                order.state = OrderState.FILLED
                order.filled_at = _now_iso()
                cqty = Decimal(str(resp.get("cummulativeQuoteQty", "0")))
                if order.executed_qty > 0:
                    order.average_fill_price = cqty / order.executed_qty
            elif status == "PARTIALLY_FILLED":
                order.state = OrderState.PARTIALLY_FILLED
            elif status in ("NEW", "ACCEPTED"):
                order.state = OrderState.ACKNOWLEDGED
            elif status in ("CANCELED", "CANCELLED"):
                order.state = OrderState.CANCELLED
            elif status == "REJECTED":
                order.state = OrderState.REJECTED
                order.reject_reason = resp.get("rejectReason", "UNKNOWN_REJECT")
            elif status == "EXPIRED":
                order.state = OrderState.EXPIRED
            else:
                order.state = OrderState.UNKNOWN_STATE
                logger.warning(f"[OrderMgr] Unrecognized exchange status: {status!r}")

            order.raw_response = resp
            self._db.update_order_state(
                order.internal_order_id, order.state.value,
                {"recovered_from": "UNKNOWN", "exchange_status": status,
                 "exchange_order_id": order.exchange_order_id}
            )
            self._db.save_order(order.to_dict())
            logger.info(
                f"[OrderMgr] Recovered order {order.client_order_id}: "
                f"exchange_status={status} → internal_state={order.state.value}"
            )

        except BinanceAPIError as e:
            if e.code == -2013:  # Order does not exist
                logger.warning(
                    f"[OrderMgr] Order {order.client_order_id} not found on exchange. "
                    "Marking UNKNOWN_STATE — manual review required."
                )
                order.state = OrderState.UNKNOWN_STATE
                self._db.update_order_state(
                    order.internal_order_id, OrderState.UNKNOWN_STATE.value,
                    {"reason": "not_found_on_exchange", "error_code": e.code}
                )
            else:
                logger.error(f"[OrderMgr] API error querying unknown order: {e}")

        except BinanceNetworkError as e:
            logger.error(f"[OrderMgr] Network error during unknown-order query: {e}")

        return order

    # ── Error Classification ──────────────────────────────────────────────────

    def _classify_api_error(self, code: int) -> str:
        """Classify a Binance API error code."""
        non_retryable = {
            -1100: "INVALID_PARAM",
            -1111: "PRECISION_ERROR",
            -1121: "INVALID_SYMBOL",
            -2010: "NEW_ORDER_REJECTED",
            -2011: "CANCEL_REJECTED",
            -2013: "ORDER_NOT_FOUND",
            -2014: "INVALID_API_KEY",
            -2015: "AUTH_ERROR",
            -2019: "INSUFFICIENT_MARGIN",
        }
        if code in non_retryable:
            return f"NON_RETRYABLE:{non_retryable[code]}"
        if code == -1003:
            return "RETRYABLE:RATE_LIMIT"
        return f"UNKNOWN:{code}"
