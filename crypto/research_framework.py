"""Research & Experiment Framework — Phase 7-8.
Every experiment must be reproducible.
Tracks: experiment ID, strategy version, git hash, dataset, parameters, results."""
from __future__ import annotations

import json
import os
import hashlib
import numpy as np
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional

logger = logging.getLogger("research_framework")

EXPERIMENTS_FILE = "experiments.json"


@dataclass
class Experiment:
    """A single experiment record."""
    experiment_id: str
    timestamp: str
    strategy_version: str
    git_hash: str = ""
    dataset: str = ""
    pair_universe: str = ""
    timeframe: str = ""
    parameters: Dict = field(default_factory=dict)
    fees_bps: float = 20.0
    slippage_bps: float = 5.0
    training_period: str = ""
    oos_period: str = ""
    # Results
    n_pairs_tested: int = 0
    n_pairs_selected: int = 0
    metrics: Dict = field(default_factory=dict)
    notes: str = ""


class ExperimentRegistry:
    """Registry of all experiments for reproducibility."""

    def __init__(self, path: str = EXPERIMENTS_FILE):
        self.path = path
        self.experiments: List[Experiment] = []
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                data = json.load(f)
            self.experiments = [Experiment(**e) for e in data.get("experiments", [])]
        except Exception:
            pass

    def save(self):
        data = {
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "experiments": [asdict(e) for e in self.experiments[-100:]],  # keep last 100
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def register(self, exp: Experiment):
        self.experiments.append(exp)
        self.save()
        logger.info(f"Registered experiment: {exp.experiment_id}")
        return exp

    def create_experiment(self, strategy_version: str, parameters: Dict,
                          n_pairs_tested: int, n_pairs_selected: int,
                          metrics: Dict, **kwargs) -> Experiment:
        """Create and register a new experiment."""
        exp_id = hashlib.md5(
            f"{strategy_version}_{json.dumps(parameters, sort_keys=True)}_{datetime.now().timestamp()}".encode()
        ).hexdigest()[:12]

        exp = Experiment(
            experiment_id=exp_id,
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            strategy_version=strategy_version,
            parameters=parameters,
            n_pairs_tested=n_pairs_tested,
            n_pairs_selected=n_pairs_selected,
            metrics=metrics,
            **kwargs,
        )
        return self.register(exp)

    def get_summary(self) -> Dict:
        """Summary of all experiments."""
        if not self.experiments:
            return {"n_experiments": 0}
        return {
            "n_experiments": len(self.experiments),
            "date_range": f"{self.experiments[0].timestamp[:10]} to {self.experiments[-1].timestamp[:10]}",
            "strategies_tested": len(set(e.strategy_version for e in self.experiments)),
        }

    def get_best_experiments(self, metric: str = "sharpe_ratio", top_n: int = 5) -> List[Experiment]:
        """Get top-N experiments by a metric."""
        scored = []
        for exp in self.experiments:
            val = exp.metrics.get(metric, 0)
            if val is not None:
                scored.append((val, exp))
        scored.sort(key=lambda x: -x[0])
        return [exp for _, exp in scored[:top_n]]


class PerformanceAnalyzer:
    """Computes comprehensive performance metrics for a backtest result."""

    @staticmethod
    def compute_metrics(returns: np.ndarray, trades: List[Dict],
                        capital: float = 10000) -> Dict:
        """Compute full metrics suite."""
        if len(returns) == 0:
            return {"error": "no returns"}

        # Returns
        total_ret = float(np.sum(returns))
        annual_ret = float(np.mean(returns) * 252 * 100)  # assuming daily

        # Risk
        vol = float(np.std(returns, ddof=1) * np.sqrt(252) * 100)
        downside = returns[returns < 0]
        downside_vol = float(np.std(downside, ddof=1) * np.sqrt(252) * 100) if len(downside) > 1 else vol

        # Drawdown
        equity = np.cumsum(returns)
        peak = np.maximum.accumulate(equity)
        dd = equity - peak
        max_dd = float(abs(np.min(dd))) * 100 if len(dd) > 0 else 0.0

        # Risk-adjusted
        sharpe = float(np.mean(returns) / (np.std(returns, ddof=1) + 1e-9) * np.sqrt(252))
        sortino = float(np.mean(returns) / (downside_vol / 100 / np.sqrt(252) + 1e-9) * np.sqrt(252))
        calmar = annual_ret / max_dd if max_dd > 0 else 0.0

        # Trading
        n_trades = len(trades)
        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) <= 0]
        win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
        avg_win = float(np.mean([t["pnl"] for t in wins])) if wins else 0.0
        avg_loss = float(np.mean([t["pnl"] for t in losses])) if losses else 0.0
        profit_factor = (sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses))) if losses and sum(t["pnl"] for t in losses) != 0 else float('inf')
        expectancy = float(np.mean([t["pnl"] for t in trades])) if trades else 0.0

        return {
            "total_return_pct": round(total_ret * 100, 2),
            "annualized_return_pct": round(annual_ret, 2),
            "volatility_pct": round(vol, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "sharpe_ratio": round(sharpe, 2),
            "sortino_ratio": round(sortino, 2),
            "calmar_ratio": round(calmar, 2),
            "n_trades": n_trades,
            "win_rate": round(win_rate, 3),
            "avg_win_pct": round(avg_win * 100, 3),
            "avg_loss_pct": round(avg_loss * 100, 3),
            "profit_factor": round(profit_factor, 2),
            "expectancy_pct": round(expectancy * 100, 3),
        }

    @staticmethod
    def monte_carlo_robustness(returns: str, n_simulations: int = 1000) -> Dict:
        """Bootstrap analysis to test if performance is robust."""
        if len(returns) < 5:
            return {"error": "insufficient data"}
        bootstrapped_sharpes = []
        for _ in range(n_simulations):
            sample = np.random.choice(returns, size=len(returns), replace=True)
            if np.std(sample, ddof=1) > 0:
                sharpe = float(np.mean(sample) / np.std(sample, ddof=1) * np.sqrt(252))
                bootstrapped_sharpes.append(sharpe)
        if not bootstrapped_sharpes:
            return {}
        return {
            "mean_sharpe": round(np.mean(bootstrapped_sharpes), 2),
            "std_sharpe": round(np.std(bootstrapped_sharpes), 2),
            "ci_5pct": round(np.percentile(bootstrapped_sharpes, 5), 2),
            "ci_95pct": round(np.percentile(bootstrapped_sharpes, 95), 2),
            "p_value_positive": round(np.mean([s > 0 for s in bootstrapped_sharpes]), 3),
        }


if __name__ == "__main__":
    registry = ExperimentRegistry()

    # Demo: register an experiment
    exp = registry.create_experiment(
        strategy_version="pairs_v2",
        parameters={"z_entry": 2.0, "z_exit": 0.5, "cost_bps": 25},
        n_pairs_tested=1500,
        n_pairs_selected=82,
        metrics={"sharpe_ratio": 4.2, "win_rate": 0.55, "max_dd": 8.5},
    )

    print(f"Experiment registered: {exp.experiment_id}")
    print(f"Registry summary: {registry.get_summary()}")

    # Demo: Monte Carlo
    np.random.seed(42)
    sample_returns = np.random.normal(0.001, 0.01, 100)
    mc = PerformanceAnalyzer.monte_carlo_robustness(sample_returns)
    print(f"\nMonte Carlo: {mc}")
