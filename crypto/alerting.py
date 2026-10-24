"""Alerting System — Phase 9.
Alerts for critical events with enough context to understand why."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable

logger = logging.getLogger("alerting")


@dataclass
class Alert:
    """A single alert."""
    alert_id: str
    timestamp: str
    severity: str           # "info", "warning", "critical"
    category: str          # "pair", "risk", "execution", "system"
    title: str
    message: str
    context: Dict = field(default_factory=dict)
    acknowledged: bool = False


class AlertManager:
    """Manages all alerts for the StratoCrypto system."""

    def __init__(self, alerts_path: str = "alerts.json"):
        self.alerts_path = alerts_path
        self.alerts: List[Alert] = []
        self.handlers: List[Callable] = []
        self.load()

    def load(self):
        if not os.path.exists(self.alerts_path):
            return
        try:
            with open(self.alerts_path) as f:
                data = json.load(f)
            self.alerts = [Alert(**a) for a in data.get("alerts", [])[-100:]]
        except Exception:
            pass

    def save(self):
        data = {
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "alerts": [
                {"alert_id": a.alert_id, "timestamp": a.timestamp, "severity": a.severity,
                 "category": a.category, "title": a.title, "message": a.message,
                 "context": a.context, "acknowledged": a.acknowledged}
                for a in self.alerts[-100:]
            ],
        }
        with open(self.alerts_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def add_handler(self, handler: Callable):
        """Add an alert handler (e.g., telegram sender)."""
        self.handlers.append(handler)

    def emit(self, severity: str, category: str, title: str,
             message: str, context: Dict = None):
        """Emit an alert."""
        alert = Alert(
            alert_id=f"{category}_{datetime.now().timestamp():.0f}",
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            severity=severity,
            category=category,
            title=title,
            message=message,
            context=context or {},
        )
        self.alerts.append(alert)
        self.save()
        logger.warning(f"ALERT [{severity}] {category}: {title}")

        # Call handlers
        for handler in self.handlers:
            try:
                handler(alert)
            except Exception:
                pass

        return alert

    # Specific alert types
    def pair_degraded(self, pair: str, score: float, reason: str):
        self.emit("warning", "pair", f"Pair degraded: {pair}",
                  f"Pair {pair} degraded (score={score:.0f}): {reason}",
                  {"pair": pair, "score": score, "reason": reason})

    def pair_breakdown(self, pair: str, pvalue: float):
        self.emit("critical", "pair", f"Cointegration breakdown: {pair}",
                  f"Pair {pair} cointegration broke down (p={pvalue:.4f})",
                  {"pair": pair, "pvalue": pvalue})

    def drawdown_warning(self, drawdown_pct: float, limit_pct: float):
        self.emit("warning", "risk", f"Drawdown warning: {drawdown_pct:.1f}%",
                  f"Drawdown {drawdown_pct:.1f}% approaching limit {limit_pct:.1f}%",
                  {"drawdown_pct": drawdown_pct, "limit_pct": limit_pct})

    def risk_halt(self, reason: str):
        self.emit("critical", "risk", "RISK HALT",
                  f"Trading halted: {reason}", {"reason": reason})

    def execution_failure(self, pair: str, reason: str):
        self.emit("warning", "execution", f"Execution failed: {pair}",
                  f"Failed to execute {pair}: {reason}",
                  {"pair": pair, "reason": reason})

    def stale_data(self, symbol: str, age_seconds: int):
        self.emit("warning", "system", f"Stale data: {symbol}",
                  f"Data for {symbol} is {age_seconds}s old",
                  {"symbol": symbol, "age_seconds": age_seconds})

    def new_high_quality_pair(self, pair: str, score: float):
        self.emit("info", "pair", f"New quality pair: {pair}",
                  f"Discovered high-quality pair {pair} (score={score:.0f})",
                  {"pair": pair, "score": score})

    def get_unacknowledged(self, severity: Optional[str] = None) -> List[Alert]:
        """Get unacknowledged alerts, optionally filtered by severity."""
        alerts = [a for a in self.alerts if not a.acknowledged]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts

    def acknowledge(self, alert_id: str):
        """Acknowledge an alert."""
        for a in self.alerts:
            if a.alert_id == alert_id:
                a.acknowledged = True
                break
        self.save()


def telegram_handler(alert: Alert) -> str:
    """Format alert for Telegram."""
    emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(alert.severity, "❓")
    msg = f"{emoji} *{alert.title}*\n{alert.message}"
    if alert.context:
        for k, v in alert.context.items():
            msg += f"\n  {k}: `{v}`"
    return msg


if __name__ == "__main__":
    mgr = AlertManager()
    mgr.add_handler(lambda a: print(telegram_handler(a)))

    # Demo alerts
    mgr.new_high_quality_pair("BTCUSDT/LTCUSDT", score=95)
    mgr.pair_degraded("ETHUSDT/XRPUSDT", score=45, reason="z-score divergence")
    mgr.drawdown_warning(12.5, 15.0)

    print(f"\nUnacknowledged: {len(mgr.get_unacknowledged())}")
    print(f"Critical: {len(mgr.get_unacknowledged('critical'))}")
