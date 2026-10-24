"""
rich/models.py — Core data models and enums for Bot5.

Every domain object (Signal, Order, Position, Fill, etc.) is defined here.
These are the language of the system — used by every module.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Enumerations ─────────────────────────────────────────────────────────────

class TradeSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class MarketRegime(str, Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"


class RelativeBias(str, Enum):
    LEAN_LONG = "LEAN_LONG"
    FADE_PUMP = "FADE_PUMP"
    NEUTRAL = "NEUTRAL"


class SignalStatus(str, Enum):
    VALID = "VALID"
    STALE = "STALE"
    INVALID = "INVALID"
    DUPLICATE = "DUPLICATE"
    BLOCKED = "BLOCKED"


class OrderState(str, Enum):
    """Complete order state machine."""
    SIGNAL_RECEIVED = "SIGNAL_RECEIVED"
    VALIDATING = "VALIDATING"
    APPROVED = "APPROVED"
    SUBMITTING = "SUBMITTING"
    UNKNOWN = "UNKNOWN"               # network timeout — status unknown
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    PROTECTED = "PROTECTED"          # stop/tp placed
    MANAGING = "MANAGING"            # active position mgmt
    EXIT_PENDING = "EXIT_PENDING"
    CLOSED = "CLOSED"
    # Abnormal
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"
    UNKNOWN_STATE = "UNKNOWN_STATE"  # persistent unknown after reconciliation
    DESYNC = "DESYNC"


class PositionState(str, Enum):
    PENDING_ENTRY = "PENDING_ENTRY"
    OPEN = "OPEN"
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"
    CLOSED = "CLOSED"
    RECOVERED = "RECOVERED"          # restored after crash
    UNPROTECTED = "UNPROTECTED"      # protection missing — emergency


class ProtectionState(str, Enum):
    NONE = "NONE"
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"
    TRIGGERED = "TRIGGERED"


class SystemState(str, Enum):
    STARTING = "STARTING"
    RECONCILING = "RECONCILING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    LIVE_LOCKED = "LIVE_LOCKED"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    DESYNC = "DESYNC"
    API_FAILURE = "API_FAILURE"
    RISK_LIMIT = "RISK_LIMIT"
    PROTECTION_FAILURE = "PROTECTION_FAILURE"
    SHUTDOWN = "SHUTDOWN"


class HealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"
    EMERGENCY = "EMERGENCY"


class RiskDecision(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


# ── Signal Contract ───────────────────────────────────────────────────────────

@dataclass
class SignalContract:
    """
    Canonical signal passed from Bot4 → Bot5.

    This is a versioned contract. Any change to Bot4's output must be
    reflected here with a schema_version bump so we can trace which version
    produced each trade.
    """
    # Identity
    signal_id: str
    signal_version: str       # version of the signal format/schema
    strategy_version: str     # Bot4 strategy version
    source: str = "BOT4_MASTER"

    # Market
    symbol: str = ""
    side: TradeSide = TradeSide.BUY
    signal_timestamp: str = ""   # ISO8601

    # Conviction
    confidence: float = 0.0      # 0–100
    predicted_change_pct: float = 0.0
    expected_vol_pct: float = 0.0

    # Edge
    expected_edge: str = ""
    relative_bias: str = ""
    market_regime: str = ""

    # Price references (informational — not used directly for orders)
    entry_reference: float = 0.0
    stop_reference: float = 0.0
    target_reference: float = 0.0

    # Strategy metadata
    strategy: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Validation status (set by SignalValidator)
    validation_status: SignalStatus = SignalStatus.VALID
    validation_reason: str = ""

    @classmethod
    def from_master_signal(cls, raw: dict) -> "SignalContract":
        """Convert a master_signals.json record to a SignalContract."""
        decision = raw.get("decision", {})
        prediction = raw.get("prediction", {})
        # Derive side from prediction direction
        pred_change = float(raw.get("pred_change") or prediction.get("predicted_change_pct", 0))
        side = TradeSide.BUY if pred_change > 0 else TradeSide.SELL

        signal_id = raw.get("signal_id") or _generate_signal_id(raw)

        return cls(
            signal_id=signal_id,
            signal_version=raw.get("signal_version", "1.0"),
            strategy_version=raw.get("strategy_version", "unknown"),
            source="BOT4_MASTER",
            symbol=raw.get("symbol", ""),
            side=side,
            signal_timestamp=raw.get("ts", "") or raw.get("signal_timestamp", ""),
            confidence=float(decision.get("score", 0)),
            predicted_change_pct=pred_change,
            expected_vol_pct=float(prediction.get("expected_vol_pct", 0)),
            expected_edge="",
            relative_bias=raw.get("bias") or prediction.get("relative_bias", "NEUTRAL"),
            market_regime=prediction.get("market_regime", "UNKNOWN"),
            entry_reference=float(raw.get("price") or raw.get("entry_reference", 0)),
            stop_reference=float(raw.get("stop_reference", 0)),
            target_reference=float(raw.get("target_reference", 0)),
            strategy=raw.get("strategy", "BOT4_CONVERGENCE"),
            metadata={
                "ai_score": raw.get("ai_score"),
                "prob_up": raw.get("prob_up"),
                "prob_up_raw": raw.get("prob_up_raw"),
                "decision_raw": decision,
                "prediction_created_at": raw.get("pred_created_at"),
            },
        )

    def age_seconds(self) -> float:
        """Seconds since signal_timestamp."""
        try:
            ts = datetime.fromisoformat(self.signal_timestamp.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return (now - ts.astimezone(timezone.utc)).total_seconds()
        except Exception:
            return float("inf")

    def to_dict(self) -> dict:
        return {
            "signal_id": self.signal_id,
            "signal_version": self.signal_version,
            "strategy_version": self.strategy_version,
            "source": self.source,
            "symbol": self.symbol,
            "side": self.side.value,
            "signal_timestamp": self.signal_timestamp,
            "confidence": self.confidence,
            "predicted_change_pct": self.predicted_change_pct,
            "expected_vol_pct": self.expected_vol_pct,
            "expected_edge": self.expected_edge,
            "relative_bias": self.relative_bias,
            "market_regime": self.market_regime,
            "entry_reference": self.entry_reference,
            "stop_reference": self.stop_reference,
            "target_reference": self.target_reference,
            "strategy": self.strategy,
            "metadata": self.metadata,
            "validation_status": self.validation_status.value,
            "validation_reason": self.validation_reason,
        }


def _generate_signal_id(raw: dict) -> str:
    """Generate a deterministic signal_id from the raw record."""
    import hashlib, json
    key = json.dumps({
        "ts": raw.get("ts", ""),
        "symbol": raw.get("symbol", ""),
        "pred_change": raw.get("pred_change", 0),
        "ai_score": raw.get("ai_score", 0),
    }, sort_keys=True)
    return "sig_" + hashlib.sha256(key.encode()).hexdigest()[:16]


# ── Order Models ──────────────────────────────────────────────────────────────

@dataclass
class RichOrder:
    """Represents a single order in Bot5."""
    internal_order_id: str = field(default_factory=lambda: "ord_" + str(uuid.uuid4())[:16])
    client_order_id: str = ""        # sent to exchange
    exchange_order_id: str = ""      # returned by exchange
    signal_id: str = ""
    position_id: str = ""

    symbol: str = ""
    side: TradeSide = TradeSide.BUY
    order_type: str = "MARKET"
    requested_qty: Decimal = Decimal("0")
    requested_price: Optional[Decimal] = None
    executed_qty: Decimal = Decimal("0")
    average_fill_price: Optional[Decimal] = None

    state: OrderState = OrderState.SIGNAL_RECEIVED
    created_at: str = ""
    updated_at: str = ""
    submitted_at: str = ""
    filled_at: str = ""

    reject_reason: str = ""
    raw_response: Optional[dict] = None

    def is_terminal(self) -> bool:
        return self.state in (
            OrderState.FILLED, OrderState.REJECTED, OrderState.CANCELLED,
            OrderState.EXPIRED, OrderState.CLOSED
        )

    def to_dict(self) -> dict:
        return {
            "internal_order_id": self.internal_order_id,
            "client_order_id": self.client_order_id,
            "exchange_order_id": self.exchange_order_id,
            "signal_id": self.signal_id,
            "position_id": self.position_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type,
            "requested_qty": str(self.requested_qty),
            "requested_price": str(self.requested_price) if self.requested_price else None,
            "executed_qty": str(self.executed_qty),
            "average_fill_price": str(self.average_fill_price) if self.average_fill_price else None,
            "state": self.state.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "reject_reason": self.reject_reason,
        }


@dataclass
class Fill:
    """A single fill event for an order."""
    fill_id: str = field(default_factory=lambda: "fill_" + str(uuid.uuid4())[:12])
    order_id: str = ""              # internal_order_id
    exchange_trade_id: str = ""
    symbol: str = ""
    side: TradeSide = TradeSide.BUY
    quantity: Decimal = Decimal("0")
    price: Decimal = Decimal("0")
    commission: Decimal = Decimal("0")
    commission_asset: str = "USDT"
    is_maker: bool = False
    filled_at: str = ""

    def notional_usdt(self) -> Decimal:
        return self.quantity * self.price


# ── Position Models ───────────────────────────────────────────────────────────

@dataclass
class RichPosition:
    """Represents an active or closed position."""
    position_id: str = field(default_factory=lambda: "pos_" + str(uuid.uuid4())[:16])
    signal_id: str = ""
    symbol: str = ""
    side: TradeSide = TradeSide.BUY

    # Entry
    quantity: Decimal = Decimal("0")
    average_entry: Optional[Decimal] = None
    entry_order_id: str = ""

    # Current
    current_price: Optional[Decimal] = None
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    fees_paid: Decimal = Decimal("0")
    slippage_cost: Decimal = Decimal("0")

    # Protection
    stop_loss: Optional[Decimal] = None
    stop_order_id: str = ""
    take_profit_1: Optional[Decimal] = None
    take_profit_2: Optional[Decimal] = None
    take_profit_3: Optional[Decimal] = None
    tp_order_ids: List[str] = field(default_factory=list)
    protection_state: ProtectionState = ProtectionState.NONE
    break_even_activated: bool = False
    trailing_active: bool = False

    # State
    position_state: PositionState = PositionState.PENDING_ENTRY
    opened_at: str = ""
    closed_at: str = ""
    exit_reason: str = ""

    # Attribution
    signal_expected_return_pct: float = 0.0
    gross_pnl: Decimal = Decimal("0")
    net_pnl: Decimal = Decimal("0")

    # MAE/MFE for analysis
    mae: Optional[Decimal] = None    # Maximum Adverse Excursion
    mfe: Optional[Decimal] = None    # Maximum Favorable Excursion

    # Versioning
    configuration_version: str = ""
    strategy_version: str = ""

    def holding_seconds(self) -> float:
        if not self.opened_at:
            return 0.0
        try:
            opened = datetime.fromisoformat(self.opened_at.replace("Z", "+00:00"))
            closed = (
                datetime.fromisoformat(self.closed_at.replace("Z", "+00:00"))
                if self.closed_at else datetime.now(timezone.utc)
            )
            return (closed.astimezone(timezone.utc) - opened.astimezone(timezone.utc)).total_seconds()
        except Exception:
            return 0.0

    def to_dict(self) -> dict:
        return {
            "position_id": self.position_id,
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": str(self.quantity),
            "average_entry": str(self.average_entry) if self.average_entry else None,
            "current_price": str(self.current_price) if self.current_price else None,
            "unrealized_pnl": str(self.unrealized_pnl),
            "realized_pnl": str(self.realized_pnl),
            "fees_paid": str(self.fees_paid),
            "slippage_cost": str(self.slippage_cost),
            "stop_loss": str(self.stop_loss) if self.stop_loss else None,
            "take_profit_1": str(self.take_profit_1) if self.take_profit_1 else None,
            "take_profit_2": str(self.take_profit_2) if self.take_profit_2 else None,
            "take_profit_3": str(self.take_profit_3) if self.take_profit_3 else None,
            "protection_state": self.protection_state.value,
            "break_even_activated": self.break_even_activated,
            "trailing_active": self.trailing_active,
            "position_state": self.position_state.value,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "exit_reason": self.exit_reason,
            "gross_pnl": str(self.gross_pnl),
            "net_pnl": str(self.net_pnl),
            "fees_paid": str(self.fees_paid),
            "configuration_version": self.configuration_version,
            "strategy_version": self.strategy_version,
        }


# ── Risk Models ───────────────────────────────────────────────────────────────

@dataclass
class RiskAssessment:
    """Result of a risk engine evaluation."""
    decision: RiskDecision
    reason: str
    allowed_risk_usdt: float = 0.0
    max_position_usdt: float = 0.0
    risk_per_trade_pct: float = 0.0
    daily_pnl_usdt: float = 0.0
    daily_loss_remaining_usdt: float = 0.0
    allocated_quantity: Decimal = Decimal("0")
    stop_loss_price: Optional[Decimal] = None
    take_profit_price: Optional[Decimal] = None
    risk_amount_quote: Decimal = Decimal("0")
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)

    @property
    def approved(self) -> bool:
        return self.decision == RiskDecision.APPROVED

    @property
    def reject_reason(self) -> str:
        return self.reason

    def is_approved(self) -> bool:
        return self.decision == RiskDecision.APPROVED



@dataclass
class PositionSizeResult:
    """Output of the position sizer."""
    approved: bool
    reason: str
    quantity: Decimal = Decimal("0")
    notional_usdt: Decimal = Decimal("0")
    risk_usdt: Decimal = Decimal("0")
    stop_price: Optional[Decimal] = None
    stop_distance_pct: float = 0.0


# ── Reconciliation Models ─────────────────────────────────────────────────────

class ReconciliationResult(str, Enum):
    SYNCED = "SYNCED"
    WARNING = "WARNING"
    DESYNC = "DESYNC"
    CRITICAL = "CRITICAL"


@dataclass
class ReconciliationReport:
    result: ReconciliationResult
    timestamp: str
    discrepancies: List[dict] = field(default_factory=list)
    local_positions: List[str] = field(default_factory=list)
    exchange_positions: List[str] = field(default_factory=list)
    local_open_orders: List[str] = field(default_factory=list)
    exchange_open_orders: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# ── Execution Models ──────────────────────────────────────────────────────────

@dataclass
class ExecutionMetrics:
    """Per-trade execution quality metrics."""
    signal_id: str
    order_id: str
    symbol: str

    signal_price: Optional[Decimal] = None
    reference_price: Optional[Decimal] = None
    requested_price: Optional[Decimal] = None
    average_fill_price: Optional[Decimal] = None
    best_bid_at_submission: Optional[Decimal] = None
    best_ask_at_submission: Optional[Decimal] = None

    slippage_absolute: Optional[Decimal] = None
    slippage_bps: float = 0.0

    signal_to_order_latency_ms: float = 0.0
    order_to_ack_latency_ms: float = 0.0
    order_to_fill_latency_ms: float = 0.0

    gross_pnl: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")
    slippage_cost: Decimal = Decimal("0")
    net_pnl: Decimal = Decimal("0")
