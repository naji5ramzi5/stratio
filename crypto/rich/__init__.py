"""
Bot5 — الغني 💰
Autonomous Execution + Risk + Position Management + Protection + Reconciliation Infrastructure

This package is strictly an EXECUTION LAYER.
It does NOT generate signals. It consumes TRADE_READY signals from Bot4.

Architecture:
  Bot4 → TRADE_READY → rich.signal_bridge
     → rich.signal_validator
     → rich.risk_engine
     → rich.position_sizer
     → rich.entry_engine
     → rich.binance_adapter (or rich.paper_broker)
     → rich.order_manager
     → rich.protection_manager
     → rich.position_manager
     → rich.reconciliation_engine
     → rich.telegram_rich_bot (notifications only)
     → rich.db (SQLite)

NEVER modify Bot1–Bot4 from this package.
NEVER write secrets to logs.
NEVER bypass the Risk Engine.
"""

__version__ = "0.1.0"
__codename__ = "الغني"
