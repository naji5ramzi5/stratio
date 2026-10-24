"""Execution Engine — Phase 6.
Separates SIGNAL GENERATION from ORDER EXECUTION.
The strategy says "I want to open this position."
The execution engine decides "How should this actually be executed?"
"""
from __future__ import annotations

import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

logger = logging.getLogger("execution_engine")


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial_fill"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


@dataclass
class Order:
    """Represents an order to be executed."""
    order_id: str
    symbol: str
    side: str          # "BUY" or "SELL"
    quantity: float
    order_type: OrderType
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    fill_price: float = 0.0
    fill_quantity: float = 0.0
    slippage_pct: float = 0.0
    fee: float = 0.0
    timestamp: str = ""
    latency_ms: float = 0.0


@dataclass
class ExecutionResult:
    """Result of executing a pair trade."""
    success: bool
    leg_a: Optional[Order] = None
    leg_b: Optional[Order] = None
    total_slippage_pct: float = 0.0
    total_fees: float = 0.0
    implementation_shortfall_pct: float = 0.0
    latency_ms: float = 0.0
    reason: str = ""


@dataclass
class ExecutionConfig:
    """Configuration for the execution engine."""
    # Slippage model
    base_slippage_bps: float = 5.0
    slippage_mode: str = "fixed"  # "fixed" | "pct" | "volatility"

    # Fees
    maker_fee_bps: float = 10.0
    taker_fee_bps: float = 10.0

    # Execution
    max_latency_ms: float = 500.0
    partial_fill_threshold: float = 0.8  # accept if >= 80% filled
    retry_on_reject: bool = True
    max_retries: int = 3

    # Risk checks
    max_spread_pct: float = 2.0
    max_volatility_pct: float = 200.0
    stale_data_seconds: int = 300


class ExecutionEngine:
    """Executes pair trades with realistic simulation."""

    def __init__(self, config: Optional[ExecutionConfig] = None, live: bool = False):
        self.config = config or ExecutionConfig()
        self.live = live
        self.order_history: List[Order] = []
        self.execution_log: List[ExecutionResult] = []

    def execute_pair_trade(self, pair: str, direction: int,
                           size_usd: float, prices: Dict[str, float],
                           volatilities: Dict[str, float] = {}) -> ExecutionResult:
        """Execute a two-leg pair trade.

        Args:
            pair: "BTCUSDT/ETHUSDT"
            direction: +1 (long A, short B) or -1 (short A, long B)
            size_usd: total notional per leg
            prices: {"BTCUSDT": 65000, "ETHUSDT": 3500}
            volatilities: {"BTCUSDT": 0.5, "ETHUSDT": 0.6}

        Returns:
            ExecutionResult with fill details
        """
        a, b = pair.split("/")

        # 1. Pre-trade risk checks
        check = self._pre_trade_check(pair, prices, volatilities)
        if not check["ok"]:
            return ExecutionResult(False, reason=check["reason"])

        # 2. Calculate order quantities
        qty_a = size_usd / prices[a]
        qty_b = size_usd / prices[b]

        # 3. Simulate execution with slippage
        start_time = time.time()

        # Leg A
        side_a = "BUY" if direction > 0 else "SELL"
        leg_a = self._execute_leg(a, side_a, qty_a, prices[a],
                                  volatilities.get(a, 0.5))

        # Leg B
        side_b = "SELL" if direction > 0 else "BUY"
        leg_b = self._execute_leg(b, side_b, qty_b, prices[b],
                                  volatilities.get(b, 0.5))

        latency = (time.time() - start_time) * 1000

        # 4. Calculate execution quality
        total_slip = leg_a.slippage_pct + leg_b.slippage_pct
        total_fee = leg_a.fee + leg_b.fee

        # Implementation shortfall: difference between signal price and fill price
        signal_value = size_usd * 2  # both legs
        fill_value = leg_a.fill_price * leg_a.fill_quantity + leg_b.fill_price * leg_b.fill_quantity
        shortfall = abs(signal_value - fill_value) / signal_value * 100

        success = (leg_a.status == OrderStatus.FILLED and leg_b.status == OrderStatus.FILLED)

        result = ExecutionResult(
            success=success, leg_a=leg_a, leg_b=leg_b,
            total_slippage_pct=total_slip, total_fees=total_fee,
            implementation_shortfall_pct=shortfall, latency_ms=latency,
            reason="filled" if success else "partial or rejected"
        )

        self.execution_log.append(result)
        return result

    def _pre_trade_check(self, pair: str, prices: Dict, vols: Dict) -> Dict:
        """Pre-trade risk checks."""
        if not prices:
            return {"ok": False, "reason": "no prices available"}

        # Check volatility (not spread — pairs have different price levels)
        for sym, vol in vols.items():
            if vol > self.config.max_volatility_pct:
                return {"ok": False, "reason": f"volatility too high for {sym}: {vol:.0f}%"}

        return {"ok": True, "reason": ""}

    def _execute_leg(self, symbol: str, side: str, quantity: float,
                     price: float, volatility: float) -> Order:
        """Execute a single leg with slippage simulation."""
        order = Order(
            order_id=f"{symbol}_{side}_{time.time_ns()}",
            symbol=symbol, side=side, quantity=quantity,
            order_type=OrderType.MARKET,
        )

        # Simulate slippage
        slip_pct = self._compute_slippage(volatility)
        slip_direction = 1 if side == "BUY" else -1  # buys slip up, sells slip down
        fill_price = price * (1 + slip_direction * slip_pct / 100)

        # Simulate fee
        fee = fill_price * quantity * self.config.taker_fee_bps / 10000.0

        order.fill_price = fill_price
        order.fill_quantity = quantity
        order.slippage_pct = slip_pct
        order.fee = fee
        order.status = OrderStatus.FILLED

        self.order_history.append(order)
        return order

    def _compute_slippage(self, volatility: float) -> float:
        """Compute slippage based on mode."""
        base = self.config.base_slippage_bps
        if self.config.slippage_mode == "fixed":
            return base
        elif self.config.slippage_mode == "pct":
            return base * (1 + volatility)
        elif self.config.slippage_mode == "volatility":
            return base * (1 + 5 * volatility)
        return base

    def get_execution_quality(self) -> Dict:
        """Aggregate execution quality metrics."""
        if not self.execution_log:
            return {}
        n = len(self.execution_log)
        success_rate = sum(1 for r in self.execution_log if r.success) / n
        avg_slip = np.mean([r.total_slippage_pct for r in self.execution_log])
        avg_shortfall = np.mean([r.implementation_shortfall_pct for r in self.execution_log])
        avg_fees = np.mean([r.total_fees for r in self.execution_log])
        return {
            "n_executions": n,
            "success_rate": round(success_rate, 3),
            "avg_slippage_bps": round(avg_slip, 2),
            "avg_implementation_shortfall_pct": round(avg_shortfall, 3),
            "avg_fees": round(avg_fees, 2),
        }


if __name__ == "__main__":
    engine = ExecutionEngine()
    result = engine.execute_pair_trade(
        pair="BTCUSDT/ETHUSDT", direction=1, size_usd=1000,
        prices={"BTCUSDT": 65000, "ETHUSDT": 3500},
        volatilities={"BTCUSDT": 0.5, "ETHUSDT": 0.6},
    )
    print(f"Success: {result.success}")
    print(f"Leg A: {result.leg_a.symbol} {result.leg_a.side} @ {result.leg_a.fill_price:.2f} (slip {result.leg_a.slippage_pct:.1f} bps)")
    print(f"Leg B: {result.leg_b.symbol} {result.leg_b.side} @ {result.leg_b.fill_price:.2f} (slip {result.leg_b.slippage_pct:.1f} bps)")
    print(f"Total slippage: {result.total_slippage_pct:.1f} bps")
    print(f"Fees: ${result.total_fees:.2f}")
    print(f"Shortfall: {result.implementation_shortfall_pct:.3f}%")
