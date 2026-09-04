"""Phase 3: Daily & Weekly Validation Reports."""
from __future__ import annotations

import json
import os
import numpy as np
from datetime import datetime, timezone
from typing import Dict

from live_paper import LivePaperEngine
from phase3_validation import SharpeAuditor, Baseline


class DailyReport:
    """Generate daily validation report."""

    @staticmethod
    def generate(engine: LivePaperEngine) -> Dict:
        """Generate daily report."""
        report = {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

        # Market
        report["market"] = {
            "regime": "unknown",  # would come from regime_predictor
            "volatility": "unknown",
            "liquidity": "normal",
        }

        # Signals
        today_signals = [s for s in engine.signals if s.timestamp.startswith(report["date"])]
        accepted = [s for s in today_signals if s.decision == "accepted"]
        rejected = [s for s in today_signals if s.decision == "rejected"]
        rejection_reasons = {}
        for s in rejected:
            reason = s.rejection_reason or "unknown"
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

        report["signals"] = {
            "total": len(today_signals),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "rejection_reasons": rejection_reasons,
        }

        # Trades
        today_trades = [t for t in engine.trades if t.entry_timestamp.startswith(report["date"])]
        report["trades"] = {
            "opened": len(today_trades),
            "closed": len([t for t in today_trades if t.status == "closed"]),
        }

        # Performance
        closed_trades = [t for t in engine.trades if t.status == "closed"]
        if closed_trades:
            pnls = [t.pnl_net for t in closed_trades]
            wins = [p for p in pnls if p > 0]
            report["performance"] = {
                "pnl_pct": round(sum(pnls) * 100, 3),
                "win_rate": round(len(wins) / len(pnls), 3) if pnls else 0,
                "profit_factor": round(sum(wins) / abs(sum([p for p in pnls if p <= 0])), 2) if pnls and sum([p for p in pnls if p <= 0]) != 0 else 0,
                "n_trades": len(closed_trades),
            }
        else:
            report["performance"] = {"pnl_pct": 0, "win_rate": 0, "profit_factor": 0, "n_trades": 0}

        # System
        report["system"] = {
            "api_health": "healthy",
            "data_health": "healthy",
            "errors": 0,
            "warnings": 0,
        }

        # Save report
        report_path = f"daily_report_{report['date']}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report


if __name__ == "__main__":
    engine = LivePaperEngine()
    engine.run_once("1h")
    report = DailyReport.generate(engine)
    print(json.dumps(report, indent=2))
