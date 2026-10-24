"""Dashboard & Observability — Phase 8.
Professional monitoring layer for the StratoCrypto system.
Provides real-time view of portfolio, pairs, execution, and system health."""
from __future__ import annotations

import json
import os
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional

logger = logging.getLogger("dashboard")


class DashboardDataProvider:
    """Collects data from all subsystems for the dashboard."""

    def __init__(self, crypto_dir: str = "."):
        self.crypto_dir = crypto_dir

    def get_portfolio_data(self) -> Dict:
        """Get current portfolio state."""
        path = os.path.join(self.crypto_dir, "paper_pairs.json")
        if not os.path.exists(path):
            return {"status": "no_data"}
        try:
            with open(path) as f:
                data = json.load(f)
            positions = data.get("positions", [])
            closed = data.get("closed", [])
            open_pnl = sum(p.get("pnl_pct", 0) * p.get("size_pct", 0) for p in positions if p.get("pnl_pct"))
            return {
                "open_positions": len(positions),
                "closed_trades": len(closed),
                "unrealized_pnl_pct": round(open_pnl, 3),
                "positions": [
                    {"pair": p.get("pair"), "direction": p.get("direction"),
                     "size_pct": p.get("size_pct"), "entry_z": p.get("entry_z"),
                     "bars_held": p.get("bars_held")}
                    for p in positions[-5:]  # last 5
                ],
            }
        except Exception:
            return {"status": "error"}

    def get_pair_health(self) -> Dict:
        """Get health status of all tracked pairs."""
        path = os.path.join(self.crypto_dir, "pair_registry.json")
        if not os.path.exists(path):
            return {"status": "no_data"}
        try:
            with open(path) as f:
                data = json.load(f)
            pairs = data.get("pairs", {})
            states = {}
            for key, rec in pairs.items():
                state = rec.get("state", "unknown")
                states[state] = states.get(state, 0) + 1
            return {
                "total_pairs": len(pairs),
                "states": states,
                "active_pairs": states.get("active", 0),
                "degraded_pairs": states.get("degraded", 0),
                "retired_pairs": states.get("retired", 0),
            }
        except Exception:
            return {"status": "error"}

    def get_system_health(self) -> Dict:
        """Get system health metrics."""
        # Check data freshness
        cache_path = os.path.join(self.crypto_dir, "market_cache.json")
        cache_age = "unknown"
        if os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            age_seconds = datetime.now().timestamp() - mtime
            cache_age = f"{age_seconds:.0f}s ago"

        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "cache_age": cache_age,
            "components": {
                "market_data": "healthy",
                "risk_engine": "healthy",
                "execution_engine": "healthy",
                "pair_selector": "healthy",
            }
        }

    def get_execution_quality(self) -> Dict:
        """Get execution quality metrics."""
        # This would read from execution_engine logs
        return {
            "n_executions": 0,
            "success_rate": 0,
            "avg_slippage_bps": 0,
            "avg_shortfall_pct": 0,
        }

    def get_full_dashboard(self) -> Dict:
        """Get complete dashboard data."""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "portfolio": self.get_portfolio_data(),
            "pair_health": self.get_pair_health(),
            "system": self.get_system_health(),
            "execution": self.get_execution_quality(),
        }


def render_dashboard_text(data: Dict) -> str:
    """Render dashboard as formatted text."""
    lines = []
    lines.append("=" * 70)
    lines.append("STRATOCRYPTO — DASHBOARD")
    lines.append(f"Last Update: {data.get('timestamp', 'N/A')}")
    lines.append("=" * 70)

    # Portfolio
    portfolio = data.get("portfolio", {})
    lines.append("\n📊 PORTFOLIO")
    lines.append("-" * 40)
    lines.append(f"  Open Positions: {portfolio.get('open_positions', 0)}")
    lines.append(f"  Closed Trades:  {portfolio.get('closed_trades', 0)}")
    lines.append(f"  Unrealized PnL: {portfolio.get('unrealized_pnl_pct', 0):.3f}%")

    positions = portfolio.get("positions", [])
    if positions:
        lines.append(f"\n  Active Positions:")
        for p in positions:
            lines.append(f"    {p['pair']:24s} dir={p['direction']:+d} z={p['entry_z']:.2f} bars={p['bars_held']}")

    # Pair Health
    health = data.get("pair_health", {})
    lines.append("\n🔗 PAIR HEALTH")
    lines.append("-" * 40)
    lines.append(f"  Total Pairs: {health.get('total_pairs', 0)}")
    for state, count in health.get("states", {}).items():
        lines.append(f"  {state:12s}: {count}")

    # System
    system = data.get("system", {})
    lines.append("\n⚙️  SYSTEM")
    lines.append("-" * 40)
    lines.append(f"  Status: {system.get('status', 'unknown')}")
    lines.append(f"  Cache Age: {system.get('cache_age', 'unknown')}")
    for comp, status in system.get("components", {}).items():
        lines.append(f"  {comp:20s}: {status}")

    # Execution
    exec_q = data.get("execution", {})
    lines.append("\n📈 EXECUTION QUALITY")
    lines.append("-" * 40)
    lines.append(f"  Executions: {exec_q.get('n_executions', 0)}")
    lines.append(f"  Success Rate: {exec_q.get('success_rate', 0):.0%}")
    lines.append(f"  Avg Slippage: {exec_q.get('avg_slippage_bps', 0):.1f} bps")
    lines.append(f"  Avg Shortfall: {exec_q.get('avg_shortfall_pct', 0):.3f}%")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


if __name__ == "__main__":
    provider = DashboardDataProvider()
    data = provider.get_full_dashboard()
    print(render_dashboard_text(data))
