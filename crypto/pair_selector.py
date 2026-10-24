"""Dynamic Pair Selection — Phase 4.
Pair lifecycle management with stability scoring and breakdown detection."""
from __future__ import annotations

import json
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger("pair_selector")

class PairState(Enum):
    DISCOVERED = "discovered"       # newly found
    VALIDATED = "validated"         # passed OOS test
    ACTIVE = "active"               # currently tradeable
    DEGRADED = "degraded"           # quality declining
    SUSPENDED = "suspended"         # temporarily paused
    RETIRED = "retired"             # permanently removed


@dataclass
class PairMetrics:
    """Track a pair's statistical health over time."""
    cointegration_pvalue: float = 1.0
    hedge_ratio: float = 1.0
    hedge_ratio_stability: float = 0.0  # std of hedge over windows
    half_life: float = 0.0
    spread_variance: float = 0.0
    zscore_divergence: float = 0.0      # current |z|
    residual_stationarity: float = 0.0   # ADF p-value of residuals
    last_update: str = ""

    @property
    def is_healthy(self) -> bool:
        return (self.cointegration_pvalue < 0.05 and
                self.half_life > 0 and self.half_life < 100 and
                self.zscore_divergence < 4.0)


@dataclass
class PairRecord:
    """Full lifecycle record for a pair."""
    pair: str
    timeframe: str
    state: PairState = PairState.DISCOVERED
    metrics: PairMetrics = field(default_factory=PairMetrics)
    performance: Dict = field(default_factory=dict)  # sharpe, win_rate, etc.
    discovery_date: str = ""
    last_trade_date: str = ""
    degradation_count: int = 0
    notes: List[str] = field(default_factory=list)


class PairLifecycleManager:
    """Manages the full lifecycle of trading pairs."""

    def __init__(self, registry_path: str = "pair_registry.json"):
        self.registry_path = registry_path
        self.pairs: Dict[str, PairRecord] = {}
        self.load()

    def load(self):
        if not self.exists(self.registry_path):
            return
        try:
            with open(self.registry_path) as f:
                data = json.load(f)
            for key, rec in data.get("pairs", {}).items():
                metrics = PairMetrics(**rec.get("metrics", {}))
                self.pairs[key] = PairRecord(
                    pair=rec["pair"], timeframe=rec["timeframe"],
                    state=PairState(rec.get("state", "discovered")),
                    metrics=metrics,
                    performance=rec.get("performance", {}),
                    discovery_date=rec.get("discovery_date", ""),
                    last_trade_date=rec.get("last_trade_date", ""),
                    degradation_count=rec.get("degradation_count", 0),
                    notes=rec.get("notes", []),
                )
        except Exception:
            pass

    @staticmethod
    def exists(path):
        import os
        return os.path.exists(path)

    def save(self):
        data = {
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pairs": {
                key: {
                    "pair": rec.pair, "timeframe": rec.timeframe,
                    "state": rec.state.value,
                    "metrics": rec.metrics.__dict__,
                    "performance": rec.performance,
                    "discovery_date": rec.discovery_date,
                    "last_trade_date": rec.last_trade_date,
                    "degradation_count": rec.degradation_count,
                    "notes": rec.notes[-10:],  # keep last 10
                }
                for key, rec in self.pairs.items()
            },
        }
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def compute_stability_score(self, metrics: PairMetrics, performance: Dict) -> float:
        """Compute a composite quality score for a pair.

        Higher = better. Range 0..100.
        """
        score = 50.0

        # Cointegration strength (0-20 points)
        if metrics.cointegration_pvalue < 0.01:
            score += 20
        elif metrics.cointegration_pvalue < 0.05:
            score += 10

        # Hedge ratio stability (0-15 points)
        if metrics.hedge_ratio_stability < 0.05:
            score += 15
        elif metrics.hedge_ratio_stability < 0.15:
            score += 8

        # Half-life quality (0-15 points)
        if 4 < metrics.half_life < 50:
            score += 15
        elif 2 < metrics.half_life < 100:
            score += 7

        # OOS performance (0-30 points)
        sharpe = performance.get("sharpe", 0)
        if sharpe > 3:
            score += 30
        elif sharpe > 1.5:
            score += 20
        elif sharpe > 0.5:
            score += 10

        win_rate = performance.get("win_rate", 0)
        if win_rate > 0.6:
            score += 10
        elif win_rate > 0.5:
            score += 5

        # Liquidity proxy (0-10 points) — more trades = better
        n_trades = performance.get("n_trades", 0)
        if n_trades >= 30:
            score += 10
        elif n_trades >= 10:
            score += 5

        # Degradation penalty
        if metrics.zscore_divergence > 3.5:
            score -= 20

        return float(np.clip(score, 0, 100))

    def update_pair(self, pair_key: str, metrics: PairMetrics, performance: Dict):
        """Update a pair's metrics and transition state if needed."""
        if pair_key not in self.pairs:
            return

        rec = self.pairs[pair_key]
        rec.metrics = metrics
        rec.performance = performance
        rec.metrics.last_update = datetime.now(timezone.utc).isoformat(timespec="seconds")

        # State transitions
        old_state = rec.state
        score = self.compute_stability_score(metrics, performance)

        if rec.state == PairState.DISCOVERED and score >= 60:
            rec.state = PairState.VALIDATED
            rec.notes.append(f"validated score={score:.0f}")
        elif rec.state == PairState.VALIDATED and score >= 70:
            rec.state = PairState.ACTIVE
            rec.notes.append(f"activated score={score:.0f}")
        elif rec.state == PairState.ACTIVE:
            if score < 50:
                rec.state = PairState.DEGRADED
                rec.degradation_count += 1
                rec.notes.append(f"degraded score={score:.0f}")
            elif score < 30:
                rec.state = PairState.SUSPENDED
                rec.notes.append(f"suspended score={score:.0f}")
        elif rec.state == PairState.DEGRADED:
            if score >= 60:
                rec.state = PairState.ACTIVE  # recovery
                rec.notes.append(f"recovered score={score:.0f}")
            elif rec.degradation_count >= 3:
                rec.state = PairState.RETIRED
                rec.notes.append(f"retired after {rec.degradation_count} degradations")

        if rec.state != old_state:
            logger.info(f"Pair {pair_key}: {old_state.value} → {rec.state.value} (score={score:.0f})")

    def get_active_pairs(self) -> List[PairRecord]:
        """Return all currently active pairs."""
        return [rec for rec in self.pairs.values() if rec.state == PairState.ACTIVE]

    def get_best_pairs(self, top_n: int = 10) -> List[PairRecord]:
        """Return top-N pairs by stability score."""
        scored = []
        for rec in self.pairs.values():
            if rec.state in (PairState.ACTIVE, PairState.VALIDATED):
                score = self.compute_stability_score(rec.metrics, rec.performance)
                scored.append((score, rec))
        scored.sort(key=lambda x: -x[0])
        return [rec for _, rec in scored[:top_n]]

    def initialize_from_db(self, db_path: str = "pairs_db.json"):
        """Load pairs from the scan database into the registry."""
        import os
        if not os.path.exists(db_path):
            print(f"DB not found: {db_path}")
            return
        with open(db_path) as f:
            db = json.load(f)
        results = db.get("results", [])
        for r in results:
            key = f"{r['pair']}@{r['tf']}"
            if key in self.pairs:
                # Update existing pair with latest performance
                self.pairs[key].performance = {"sharpe": r.get("sharpe", 0),
                                                "win_rate": r.get("win_rate", 0),
                                                "n_trades": r.get("n_trades", 0)}
                self.update_pair(key, self.pairs[key].metrics, self.pairs[key].performance)
                self.update_pair(key, self.pairs[key].metrics, self.pairs[key].performance)
                continue
            metrics = PairMetrics(
                cointegration_pvalue=0.01,
                hedge_ratio=1.0,
                hedge_ratio_stability=0.1,
                half_life=20.0,
            )
            performance = {"sharpe": r.get("sharpe", 0), "win_rate": r.get("win_rate", 0),
                           "n_trades": r.get("n_trades", 0)}
            rec = PairRecord(
                pair=r["pair"], timeframe=r["tf"],
                state=PairState.DISCOVERED,
                metrics=metrics,
                performance=performance,
                discovery_date=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            self.pairs[key] = rec
            # Auto-transition based on score (chain transitions)
            self.update_pair(key, metrics, performance)
            self.update_pair(key, metrics, performance)
        self.save()
        print(f"Initialized {len(self.pairs)} pairs in registry")


if __name__ == "__main__":
    mgr = PairLifecycleManager()
    mgr.initialize_from_db()
    print(f"\nTotal pairs: {len(mgr.pairs)}")
    for state in PairState:
        count = sum(1 for r in mgr.pairs.values() if r.state == state)
        print(f"  {state.value}: {count}")

    print(f"\nTop 10 pairs by score:")
    for rec in mgr.get_best_pairs(10):
        score = mgr.compute_stability_score(rec.metrics, rec.performance)
        print(f"  {rec.pair:24s} {rec.timeframe:4s} state={rec.state.value:10s} score={score:.0f}")
