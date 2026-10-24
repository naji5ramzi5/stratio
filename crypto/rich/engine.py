"""
rich/engine.py — Bot5 «الغني» Master Autonomous Trading System Engine.

The main orchestrator running the continuous execution, risk, protection, and reconciliation event loop.
"""
from __future__ import annotations

import logging
import signal
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Optional

from .binance_adapter import BinanceAdapter
from .config import RichConfig, TradingMode
from .db import RichDB
from .execution_engine import ExecutionEngine
from .kill_switch import KillSwitch
from .reconciliation_engine import ReconciliationEngine
from .recovery_engine import RecoveryEngine
from .signal_bridge import SignalBridge
from .telegram_reporter import TelegramReporter
from .time_sync import TimeSync

logger = logging.getLogger("rich.engine")


class RichEngine:
    """
    Master Runtime for Bot5 (الغني 💰).
    """

    def __init__(self, config: Optional[RichConfig] = None):
        self.config = config or RichConfig.load()
        self.db = RichDB(self.config.db_path)
        self.kill_switch = KillSwitch(self.config, self.db)
        self.reporter = TelegramReporter(self.config)

        # Time synchronization & Binance adapter
        self.time_sync = TimeSync()
        self.adapter: Optional[BinanceAdapter] = None
        if self.config.trading_mode == TradingMode.LIVE:
            if not self.config.api_key or not self.config.api_secret:
                raise ValueError("Binance API credentials missing for LIVE mode!")
            self.adapter = BinanceAdapter(
                api_key=self.config.api_key,
                api_secret=self.config.api_secret,
                testnet=self.config.use_testnet,
                time_sync=self.time_sync,
            )

        # Core Engines
        self.execution_engine = ExecutionEngine(
            config=self.config,
            db=self.db,
            adapter=self.adapter,
            kill_switch=self.kill_switch,
            reporter=self.reporter,
        )
        self.reconciliation_engine = ReconciliationEngine(
            config=self.config,
            db=self.db,
            adapter=self.adapter,
            kill_switch=self.kill_switch,
        )
        self.recovery_engine = RecoveryEngine(
            config=self.config,
            db=self.db,
            adapter=self.adapter,
            kill_switch=self.kill_switch,
        )
        self.signal_bridge = SignalBridge(self.config, self.db)

        self._running = False
        self._last_reconciliation_ts = 0.0
        self._last_heartbeat_ts = 0.0

    def start(self) -> None:
        """Initialize and run the main trading loop."""
        logger.info(
            f"🚀 [Bot5 - الغني] Initializing in {self.config.trading_mode.value.upper()} mode..."
        )

        # Setup graceful signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        # 1. Synchronize time with Binance if live
        if self.adapter:
            try:
                self.time_sync.sync_with_binance(self.adapter)
                logger.info("[Bot5 - الغني] Binance time sync complete.")
            except Exception as e:
                logger.warning(f"[Bot5 - الغني] Time sync failed: {e}")

        # 2. Run Crash & Startup Recovery
        recovery_summary = self.recovery_engine.run_startup_recovery()
        if not recovery_summary.is_safe_to_trade:
            logger.critical("[Bot5 - الغني] Startup recovery failed! Entering SAFE/LOCKED state.")
            self.kill_switch.system_error("Startup recovery failure", str(recovery_summary.critical_errors))

        self._running = True
        logger.info("✨ [Bot5 - الغني] Event loop active and waiting for Bot4 signals.")

        while self._running:
            try:
                self._tick()
                time.sleep(1.0)
            except Exception as exc:
                logger.exception(f"[Bot5 - الغني] Error in tick loop: {exc}")
                time.sleep(2.0)

        self._cleanup()

    def _tick(self) -> None:
        """Single loop execution tick."""
        now = time.time()

        # 1. Periodic Reconciliation (every 60s)
        if now - self._last_reconciliation_ts >= 60.0:
            self._last_reconciliation_ts = now
            reconcile_report = self.reconciliation_engine.run_reconciliation()
            if not reconcile_report.is_clean:
                self.reporter.notify_reconciliation_warning(reconcile_report.summary())

        # 2. Ingest signals from Bot4
        if self.kill_switch.can_trade():
            new_signals = self.signal_bridge.poll_signals_file()
            for sig in new_signals:
                self.execution_engine.process_trade_signal(sig)

        # 3. Mark-to-market and trigger checks
        self._update_prices_and_triggers()

        # 4. Heartbeat (every 300s)
        if now - self._last_heartbeat_ts >= 300.0:
            self._last_heartbeat_ts = now
            open_count = len(self.db.get_open_positions())
            logger.info(f"[Bot5 - الغني] Heartbeat OK | Mode={self.config.trading_mode.value} | OpenPositions={open_count}")

    def _update_prices_and_triggers(self) -> None:
        """Fetch latest prices and update positions."""
        open_positions = self.db.get_open_positions()
        if not open_positions:
            return

        # Fetch current mark prices
        price_feed: Dict[str, Decimal] = {}
        symbols = set(p["symbol"] for p in open_positions if p.get("symbol"))

        for sym in symbols:
            if self.adapter:
                try:
                    ticker = self.adapter.get_ticker_price(sym)
                    price_feed[sym] = Decimal(str(ticker.get("price", "0")))
                except Exception as e:
                    logger.debug(f"Failed to fetch ticker for {sym}: {e}")
            else:
                # In paper simulation, use stored current price or entry reference
                pos = next((p for p in open_positions if p.get("symbol") == sym), None)
                if pos and pos.get("current_price"):
                    price_feed[sym] = Decimal(str(pos["current_price"]))

        if price_feed:
            self.execution_engine.update_market_prices(price_feed)

    def _handle_shutdown(self, signum, frame) -> None:
        logger.warning(f"[Bot5 - الغني] Shutdown signal received ({signum}). Initiating graceful termination...")
        self._running = False

    def _cleanup(self) -> None:
        logger.info("[Bot5 - الغني] Performing final safety checks and flushing queues...")
        self.reporter.shutdown()
        logger.info("[Bot5 - الغني] Engine shutdown complete. Goodbye.")
