"""Phase 3: Live Paper Validation Engine.
Validates whether the statistical arbitrage edge survives real-time market conditions.

Creates a baseline, runs live paper trading, logs all signals,
compares live vs backtest, and produces a final verdict."""
from __future__ import annotations

import json
import os
import numpy as np
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger("phase3_validation")

BASELINE_FILE = "phase2_baseline.json"
SIGNALS_FILE = "live_signals.json"
VALIDATION_FILE = "validation_report.json"


class Verdict(Enum):
    RED = "red"         # Edge does not survive
    YELLOW = "yellow"   # Insufficient evidence
    GREEN = "green"     # Edge validated


@dataclass
class Baseline:
    """Phase 2 baseline — frozen configuration and results."""
    strategy_version: str = "pairs_v2"
    git_hash: str = ""
    created_at: str = ""
    pair_universe: int = 0
    timeframes: List[str] = field(default_factory=list)
    parameters: Dict = field(default_factory=dict)
    fees_bps: float = 20.0
    slippage_bps: float = 5.0
    risk_parameters: Dict = field(default_factory=dict)
    # OOS results
    oos_results: Dict = field(default_factory=dict)

    def save(self, path: str = BASELINE_FILE):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str = BASELINE_FILE) -> Optional[Baseline]:
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class LiveSignal:
    """A single live signal record."""
    signal_id: str
    timestamp: str
    pair: str
    timeframe: str
    price_a: float
    price_b: float
    spread: float
    hedge_ratio: float
    z_score: float
    half_life: float
    cointegration_pvalue: float
    stability_score: float
    market_regime: str
    expected_slippage_bps: float
    expected_fees_bps: float
    risk_score: float
    decision: str          # "accepted" or "rejected"
    rejection_reason: str = ""
    direction: int = 0      # +1 long spread, -1 short spread


@dataclass
class LiveTrade:
    """A simulated live paper trade."""
    trade_id: str
    signal_id: str
    pair: str
    timeframe: str
    direction: int
    entry_timestamp: str
    entry_price_a: float
    entry_price_b: float
    entry_z: float
    signal_price_a: float      # price when signal was generated
    signal_price_b: float
    execution_price_a: float   # simulated execution (with slippage)
    execution_price_b: float
    slippage_a_bps: float
    slippage_b_bps: float
    fees_a: float
    fees_b: float
    size_usd: float
    exit_timestamp: str = ""
    exit_price_a: float = 0.0
    exit_price_b: float = 0.0
    exit_z: float = 0.0
    pnl_raw: float = 0.0
    pnl_net: float = 0.0
    fees_total: float = 0.0
    slippage_total_bps: float = 0.0
    implementation_shortfall_pct: float = 0.0
    bars_held: int = 0
    status: str = "open"       # open, closed, stopped, expired


class BaselineCreator:
    """Creates the Phase 2 baseline from current system state."""

    @staticmethod
    def create() -> Baseline:
        baseline = Baseline(
            strategy_version="pairs_v2",
            git_hash=BaselineCreator._get_git_hash(),
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            pair_universe=82,
            timeframes=["5m", "15m", "1h", "4h", "1d"],
            parameters={
                "z_entry": 2.0,
                "z_exit": 0.5,
                "z_stop": 3.5,
                "max_hold_bars": {"5m": 288, "15m": 96, "1h": 168, "4h": 42, "1d": 14},
                "cost_bps": 25.0,
            },
            fees_bps=20.0,
            slippage_bps=5.0,
            risk_parameters={
                "max_position_pct": 0.05,
                "max_portfolio_exposure_pct": 0.6,
                "max_drawdown_pct": 15.0,
                "daily_loss_limit_pct": 3.0,
                "weekly_loss_limit_pct": 7.0,
                "consecutive_loss_limit": 5,
                "vol_scaling_enabled": True,
            },
            oos_results=BaselineCreator._collect_oos_results(),
        )
        baseline.save()
        return baseline

    @staticmethod
    def _get_git_hash() -> str:
        try:
            import subprocess
            result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
            return result.stdout.strip()[:12]
        except Exception:
            return "unknown"

    @staticmethod
    def _collect_oos_results() -> Dict:
        """Collect OOS results from pairs_db.json."""
        if not os.path.exists("pairs_db.json"):
            return {}
        with open("pairs_db.json") as f:
            db = json.load(f)
        results = db.get("results", [])
        if not results:
            return {}
        sharpes = [r.get("sharpe", 0) for r in results]
        win_rates = [r.get("win_rate", 0) for r in results]
        return {
            "n_pairs_tested": 1596,
            "n_pairs_selected": len(results),
            "mean_sharpe": round(np.mean(sharpes), 2),
            "median_sharpe": round(np.median(sharpes), 2),
            "max_sharpe": round(max(sharpes), 2),
            "min_sharpe": round(min(sharpes), 2),
            "mean_win_rate": round(np.mean(win_rates), 3),
            "total_trades": sum(r.get("n_trades", 0) for r in results),
        }


class SharpeAuditor:
    """Independent Sharpe ratio verification."""

    @staticmethod
    def audit(returns: np.ndarray, trades: List[Dict]) -> Dict:
        """Full Sharpe audit with multiple methods."""
        if len(returns) < 3:
            return {"error": "insufficient data"}

        # Method 1: Per-trade Sharpe (trade-based)
        trade_returns = np.array([t.get("pnl", 0) for t in trades if t.get("pnl") is not None])
        if len(trade_returns) < 3:
            return {"error": "insufficient trades"}

        trade_sharpe = float(np.mean(trade_returns) / (np.std(trade_returns, ddof=1) + 1e-9))

        # Method 2: Equity-curve Sharpe (time-series)
        equity = np.cumsum(trade_returns)
        equity_returns = np.diff(equity)
        if len(equity_returns) > 1 and np.std(equity_returns, ddof=1) > 0:
            equity_sharpe = float(np.mean(equity_returns) / np.std(equity_returns, ddof=1) * np.sqrt(252))
        else:
            equity_sharpe = 0.0

        # Method 3: Annualized per-trade Sharpe
        n_trades = len(trade_returns)
        trades_per_year = 252  # assuming daily-like frequency
        annual_sharpe = trade_sharpe * np.sqrt(min(n_trades, trades_per_year))

        # Confidence interval
        se = np.sqrt((1 + 0.5 * trade_sharpe**2) / n_trades)
        ci_low = trade_sharpe - 1.96 * se
        ci_high = trade_sharpe + 1.96 * se

        # Bootstrap
        bootstrapped = []
        for _ in range(1000):
            sample = np.random.choice(trade_returns, size=n_trades, replace=True)
            if np.std(sample, ddof=1) > 0:
                bootstrapped.append(float(np.mean(sample) / np.std(sample, ddof=1)))
        bootstrap_ci = (round(np.percentile(bootstrapped, 5), 2), round(np.percentile(bootstrapped, 95), 2)) if bootstrapped else (0, 0)

        return {
            "trade_based_sharpe": round(trade_sharpe, 2),
            "equity_curve_sharpe": round(equity_sharpe, 2),
            "annualized_sharpe": round(annual_sharpe, 2),
            "n_trades": n_trades,
            "confidence_interval_95": (round(ci_low, 2), round(ci_high, 2)),
            "bootstrap_ci_95": bootstrap_ci,
            "discrepancy": round(abs(trade_sharpe - equity_sharpe), 2),
            "is_reliable": n_trades >= 20 and abs(trade_sharpe - equity_sharpe) < 2.0,
        }

    @staticmethod
    def investigate_extreme(sharpe: float, trades: List[Dict]) -> Dict:
        """Investigate pairs with extreme Sharpe (>10)."""
        if sharpe <= 10:
            return {"is_extreme": False}

        returns = [t.get("pnl", 0) for t in trades if t.get("pnl") is not None]
        if not returns:
            return {"is_extreme": True, "warning": "no return data"}

        returns = np.array(returns)
        return {
            "is_extreme": True,
            "sharpe": round(sharpe, 2),
            "n_trades": len(returns),
            "mean_return": round(float(np.mean(returns)) * 100, 3),
            "median_return": round(float(np.median(returns)) * 100, 3),
            "max_win": round(float(np.max(returns)) * 100, 3),
            "max_loss": round(float(np.min(returns)) * 100, 3),
            "std_return": round(float(np.std(returns, ddof=1)) * 100, 3),
            "skewness": round(float(np.mean(((returns - np.mean(returns)) / (np.std(returns) + 1e-9))**3)), 3),
            "warning": "EXTREME_SHARPE" if sharpe > 20 else "HIGH_SHARPE",
            "assessment": "statistically_unreliable" if len(returns) < 10 else "investigate_further",
        }


class VerdictEngine:
    """Final verdict: RED / YELLOW / GREEN."""

    @staticmethod
    def evaluate(live_results: Dict, oos_results: Dict, n_days: int) -> Tuple[Verdict, str]:
        """Evaluate whether the edge survives."""
        reasons = []

        # 1. Minimum observation period
        if n_days < 7:
            return Verdict.YELLOW, f"Only {n_days} days of data — need at least 7"

        # 2. Compare win rate
        oos_wr = oos_results.get("mean_win_rate", 0.5)
        live_wr = live_results.get("win_rate", 0)
        if live_wr < oos_wr - 0.15:
            reasons.append(f"Win rate degraded: OOS={oos_wr:.0%} vs Live={live_wr:.0%}")

        # 3. Compare trade count
        oos_trades = oos_results.get("total_trades", 0)
        live_trades = live_results.get("n_trades", 0)
        expected_trades = oos_trades * (n_days / 365)  # scale by time
        if expected_trades > 5 and live_trades < expected_trades * 0.3:
            reasons.append(f"Trade count too low: expected ~{expected_trades:.0f}, got {live_trades}")

        # 4. Check for catastrophic drawdown
        live_dd = live_results.get("max_drawdown_pct", 0)
        if live_dd > 25:
            reasons.append(f"Excessive drawdown: {live_dd:.1f}%")

        # 5. Check execution quality
        live_slip = live_results.get("avg_slippage_bps", 0)
        if live_slip > 20:
            reasons.append(f"Excessive slippage: {live_slip:.1f} bps")

        # Determine verdict
        if len(reasons) >= 3:
            return Verdict.RED, " | ".join(reasons)
        elif len(reasons) >= 1:
            return Verdict.YELLOW, " | ".join(reasons)
        else:
            return Verdict.GREEN, "Live performance consistent with OOS expectations"


if __name__ == "__main__":
    # Create baseline
    print("Creating Phase 2 Baseline...")
    baseline = BaselineCreator.create()
    print(f"Baseline created: {baseline.strategy_version}")
    print(f"Pair universe: {baseline.pair_universe}")
    print(f"OOS results: {baseline.oos_results}")

    # Sharpe audit demo
    print("\nSharpe Audit Demo...")
    np.random.seed(42)
    sample_returns = np.random.normal(0.001, 0.01, 50)
    sample_trades = [{"pnl": r} for r in sample_returns]
    audit = SharpeAuditor.audit(sample_returns, sample_trades)
    print(f"Audit: {audit}")

    # Extreme Sharpe demo
    extreme_trades = [{"pnl": 0.05}, {"pnl": 0.03}, {"pnl": 0.04}]
    extreme = SharpeAuditor.investigate_extreme(50.0, extreme_trades)
    print(f"Extreme: {extreme}")

    # Verdict demo
    verdict, reason = VerdictEngine.evaluate(
        live_results={"win_rate": 0.55, "n_trades": 10, "max_drawdown_pct": 8.0, "avg_slippage_bps": 5.0},
        oos_results={"mean_win_rate": 0.52, "total_trades": 1000},
        n_days=14,
    )
    print(f"\nVerdict: {verdict.value} — {reason}")
