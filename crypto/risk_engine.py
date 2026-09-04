"""Centralized Risk Engine — Phase 5.
No strategy component can bypass this.
Implements: position limits, drawdown protection, correlation risk,
volatility scaling, Kelly sizing, circuit breaker."""
from __future__ import annotations

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger("risk_engine")


class RiskLevel(Enum):
    NORMAL = "normal"
    WARNING = "warning"
    REDUCED = "reduced_risk"
    HALTED = "halted"


@dataclass
class RiskLimits:
    """Configurable risk limits."""
    # Position limits
    max_position_pct: float = 0.05        # 5% per pair
    max_pair_exposure_pct: float = 0.05   # 5% per pair
    max_portfolio_exposure_pct: float = 0.6  # 60% total
    max_concurrent_positions: int = 15
    max_leverage: float = 1.0             # no leverage for spot

    # Drawdown protection
    daily_loss_limit_pct: float = 3.0     # 3% daily max loss
    weekly_loss_limit_pct: float = 7.0    # 7% weekly max loss
    max_drawdown_pct: float = 15.0        # 15% max drawdown
    consecutive_loss_limit: int = 5       # halt after 5 consecutive losses

    # Volatility scaling
    vol_scaling_enabled: bool = True
    target_volatility_pct: float = 10.0   # target 10% annualized vol

    # Kelly
    kelly_fraction: float = 0.25          # quarter-Kelly
    max_kelly_pct: float = 0.05           # cap Kelly at 5%

    # Circuit breaker
    circuit_breaker_enabled: bool = True
    max_spread_pct: float = 2.0           # reject if spread > 2%
    max_volatility_pct: float = 200.0     # reject if vol > 200% annualized
    stale_data_seconds: int = 300         # reject if data older than 5 min


@dataclass
class Position:
    """Represents a proposed or active position."""
    pair: str
    direction: int              # +1 long spread, -1 short spread
    size_pct: float             # % of capital
    entry_z: float
    stop_loss_z: float = 3.5
    take_profit_z: float = 0.5
    timestamp: str = ""


@dataclass
class RiskCheck:
    """Result of a risk check."""
    approved: bool
    reason: str
    adjusted_size_pct: float = 0.0
    risk_level: RiskLevel = RiskLevel.NORMAL


class RiskEngine:
    """Centralized risk management. All position requests must pass through here."""

    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()
        self.level = RiskLevel.NORMAL
        self.daily_pnl_pct = 0.0
        self.weekly_pnl_pct = 0.0
        self.max_drawdown_ever = 0.0
        self.peak_capital = 1.0
        self.consecutive_losses = 0
        self.circuit_breaker_tripped = False
        self.last_check_time = ""

    def check_position(self, position: Position, portfolio_state: Dict) -> RiskCheck:
        """Check if a proposed position is acceptable.

        Args:
            position: the proposed position
            portfolio_state: dict with keys:
                - current_exposure_pct: float
                - n_open_positions: int
                - daily_pnl_pct: float
                - weekly_pnl_pct: float
                - current_drawdown_pct: float
                - open_pairs: list of str

        Returns:
            RiskCheck with approval status and possibly adjusted size
        """
        # 1. Circuit breaker check
        if self.circuit_breaker_tripped:
            return RiskCheck(False, "circuit breaker TRIPPED", 0.0, RiskLevel.HALTED)

        # 2. Drawdown protection
        current_dd = portfolio_state.get("current_drawdown_pct", 0.0)
        if current_dd >= self.limits.max_drawdown_pct:
            return RiskCheck(False, f"max drawdown exceeded: {current_dd:.1f}% >= {self.limits.max_drawdown_pct}%",
                           0.0, RiskLevel.HALTED)

        # 3. Daily loss limit
        daily_pnl = portfolio_state.get("daily_pnl_pct", 0.0)
        if daily_pnl <= -self.limits.daily_loss_limit_pct:
            return RiskCheck(False, f"daily loss limit hit: {daily_pnl:.1f}%", 0.0, RiskLevel.HALTED)

        # 4. Weekly loss limit
        weekly_pnl = portfolio_state.get("weekly_pnl_pct", 0.0)
        if weekly_pnl <= -self.limits.weekly_loss_limit_pct:
            return RiskCheck(False, f"weekly loss limit hit: {weekly_pnl:.1f}%", 0.0, RiskLevel.HALTED)

        # 5. Consecutive losses
        if self.consecutive_losses >= self.limits.consecutive_loss_limit:
            return RiskCheck(False, f"consecutive loss limit: {self.consecutive_losses}", 0.0, RiskLevel.HALTED)

        # 6. Position count limit
        n_open = portfolio_state.get("n_open_positions", 0)
        if n_open >= self.limits.max_concurrent_positions:
            return RiskCheck(False, f"max positions reached: {n_open}/{self.limits.max_concurrent_positions}",
                           0.0, RiskLevel.WARNING)

        # 7. Portfolio exposure limit
        current_exposure = portfolio_state.get("current_exposure_pct", 0.0)
        if current_exposure + position.size_pct > self.limits.max_portfolio_exposure_pct:
            adjusted = max(0, self.limits.max_portfolio_exposure_pct - current_exposure)
            if adjusted < 0.01:
                return RiskCheck(False, f"portfolio exposure full: {current_exposure:.0%}", 0.0, RiskLevel.WARNING)
            position.size_pct = adjusted

        # 8. Single position limit
        if position.size_pct > self.limits.max_position_pct:
            position.size_pct = self.limits.max_position_pct

        # 9. Pair already open
        open_pairs = portfolio_state.get("open_pairs", [])
        if position.pair in open_pairs:
            return RiskCheck(False, f"pair {position.pair} already open", 0.0, RiskLevel.WARNING)

        # 10. Volatility scaling (reduce size in high vol)
        if self.limits.vol_scaling_enabled:
            current_vol = portfolio_state.get("current_volatility_pct", 10.0)
            vol_scalar = self.limits.target_volatility_pct / max(current_vol, 1.0)
            vol_scalar = np.clip(vol_scalar, 0.25, 1.5)  # don't reduce below 25% or increase above 150%
            position.size_pct *= vol_scalar

        # Final: ensure minimum size
        if position.size_pct < 0.01:
            return RiskCheck(False, f"size too small after adjustments: {position.size_pct:.1%}", 0.0, RiskLevel.WARNING)

        # Determine risk level
        risk_level = self._determine_risk_level(current_dd, daily_pnl, n_open)

        return RiskCheck(True, "approved", position.size_pct, risk_level)

    def _determine_risk_level(self, drawdown: float, daily_pnl: float, n_positions: int) -> RiskLevel:
        """Determine current risk level."""
        if drawdown > self.limits.max_drawdown_pct * 0.7:
            return RiskLevel.REDUCED
        if daily_pnl < -self.limits.daily_loss_limit_pct * 0.5:
            return RiskLevel.WARNING
        if n_positions > self.limits.max_concurrent_positions * 0.8:
            return RiskLevel.WARNING
        return RiskLevel.NORMAL

    def record_trade_result(self, pnl_pct: float):
        """Update risk state after a trade closes."""
        if pnl_pct < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        self.daily_pnl_pct += pnl_pct
        self.weekly_pnl_pct += pnl_pct

    def update_capital(self, capital_ratio: float):
        """Update peak capital and drawdown tracking."""
        if capital_ratio > self.peak_capital:
            self.peak_capital = capital_ratio
        current_dd = (1 - capital_ratio / self.peak_capital) * 100
        self.max_drawdown_ever = max(self.max_drawdown_ever, current_dd)

    def trip_circuit_breaker(self, reason: str):
        """Trip the circuit breaker."""
        self.circuit_breaker_tripped = True
        self.level = RiskLevel.HALTED
        logger.critical(f"CIRCUIT BREAKER TRIPPED: {reason}")

    def reset_circuit_breaker(self):
        """Reset circuit breaker (requires manual intervention)."""
        self.circuit_breaker_tripped = False
        self.level = RiskLevel.NORMAL
        logger.info("Circuit breaker reset")

    def daily_reset(self):
        """Reset daily counters."""
        self.daily_pnl_pct = 0.0

    def weekly_reset(self):
        """Reset weekly counters."""
        self.weekly_pnl_pct = 0.0


def kelly_size(win_rate: float, avg_win: float, avg_loss: float, fraction: float = 0.25) -> float:
    """Proper Kelly Criterion: f* = (p*b - q) / b where b = avg_win/avg_loss."""
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    b = avg_win / avg_loss  # win/loss ratio
    p = win_rate
    q = 1 - p
    kelly = (b * p - q) / b
    return max(0.0, min(kelly * fraction, 0.05))


def correlation_risk_check(new_pair_returns: np.ndarray, existing_returns: List[np.ndarray],
                          max_avg_correlation: float = 0.7) -> Tuple[bool, str]:
    """Check if a new pair is too correlated with existing positions."""
    if len(existing_returns) == 0 or len(new_pair_returns) == 0:
        return True, "no existing positions"

    correlations = []
    for existing in existing_returns:
        min_len = min(len(new_pair_returns), len(existing))
        if min_len < 10:
            continue
        corr = np.corrcoef(new_pair_returns[-min_len:], existing[-min_len:])[0, 1]
        if not np.isnan(corr):
            correlations.append(abs(corr))

    if not correlations:
        return True, "insufficient data for correlation check"

    avg_corr = np.mean(correlations)
    if avg_corr > max_avg_correlation:
        return False, f"too correlated with existing positions: avg r={avg_corr:.2f}"

    return True, f"correlation OK: avg r={avg_corr:.2f}"


if __name__ == "__main__":
    # Demo
    engine = RiskEngine()

    # Test a position
    pos = Position(pair="BTCUSDT/LTCUSDT", direction=1, size_pct=0.05, entry_z=-2.5)
    portfolio = {
        "current_exposure_pct": 0.3,
        "n_open_positions": 5,
        "daily_pnl_pct": -1.0,
        "weekly_pnl_pct": -2.0,
        "current_drawdown_pct": 5.0,
        "open_pairs": ["ETHUSDT/XRPUSDT"],
        "current_volatility_pct": 15.0,
    }

    result = engine.check_position(pos, portfolio)
    print(f"Position {pos.pair}: approved={result.approved}, size={result.adjusted_size_pct:.1%}, level={result.risk_level.value}")
    print(f"Reason: {result.reason}")

    # Test Kelly
    k = kelly_size(win_rate=0.55, avg_win=0.02, avg_loss=0.015)
    print(f"\nKelly size: {k:.1%}")

    # Test correlation
    np.random.seed(42)
    r1 = np.random.normal(0, 0.01, 100)
    r2 = r1 + np.random.normal(0, 0.005, 100)  # highly correlated
    ok, msg = correlation_risk_check(r1, [r2])
    print(f"Correlation check: {ok}, {msg}")
