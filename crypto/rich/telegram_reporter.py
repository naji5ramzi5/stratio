"""
rich/telegram_reporter.py — Secure Telegram Alerting & Performance Reporter for Bot5.

Features:
  - Formats rich, executive-grade operational cards for:
      * Trade Entry
      * Take Profit Hits (TP1 / TP2 / TP3)
      * Stop Loss & Trailing Exit
      * Kill Switch & Emergency Alerts
      * Protection Failure & Reconciliation Warnings
      * Daily / Periodic Health & Performance Digest
  - 100% Leak Prevention: Filters all messages through SecretRedactor.
  - Rate-limited queue to prevent Telegram 429 Too Many Requests.
  - Graceful fallback when Telegram credentials are unconfigured or offline.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional

from .config import RichConfig
from .models import RichPosition, TradeSide
from .secret_redactor import SecretRedactor

logger = logging.getLogger("rich.telegram_reporter")


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


class TelegramReporter:
    """
    Production Telegram Notification Dispatcher for Bot5 (الغني).
    """

    def __init__(self, config: RichConfig):
        self._config = config
        self._token = config.telegram_bot_token
        self._chat_id = config.telegram_chat_id
        self._enabled = bool(self._token and self._chat_id)
        self._queue: queue.Queue = queue.Queue(maxsize=1000)
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        if self._enabled:
            self._start_worker()
            logger.info("[TelegramReporter] Worker initialized and ready.")
        else:
            logger.info("[TelegramReporter] Running in silent mode (credentials not configured).")

    def _start_worker(self) -> None:
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True, name="RichTelegramWorker")
        self._worker_thread.start()

    def _process_queue(self) -> None:
        while not self._stop_event.is_set():
            try:
                msg = self._queue.get(timeout=1.0)
                if msg is None:
                    break
                self._send_http(msg)
                time.sleep(0.05)  # Safe rate-limiting
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[TelegramReporter] Worker error: {e}")

    def _send_http(self, text: str) -> bool:
        if not self._enabled:
            return False

        # Redact secrets unconditionally
        safe_text = SecretRedactor.redact_text(text)

        url = f"https://api.telegram.org/bot{self._token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": safe_text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        try:
            req_data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=req_data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception as exc:
            logger.warning(f"[TelegramReporter] Failed to send Telegram message: {exc}")
            return False

    def enqueue(self, message: str) -> None:
        """Enqueue message for asynchronous dispatch."""
        if not self._enabled:
            logger.info(f"[TelegramReporter - MOCK]:\n{SecretRedactor.redact_text(message)}")
            return
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            logger.error("[TelegramReporter] Queue full! Dropping notification.")

    def notify_trade_entry(self, pos: RichPosition, risk_amount: Decimal, expected_return: float) -> None:
        emoji = "🟢 <b>BUY</b>" if pos.side == TradeSide.BUY else "🔴 <b>SELL</b>"
        msg = (
            f"🚀 <b>BOT5 «الغني» — TRADE OPENED</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"Symbol: <code>{pos.symbol}</code> | {emoji}\n"
            f"Entry Price: <b>${pos.average_entry:,.4f}</b>\n"
            f"Quantity: <code>{pos.quantity}</code>\n"
            f"Stop Loss: <b>${pos.stop_loss:,.4f}</b>\n"
            f"Take Profit 1: <b>${pos.take_profit_1:,.4f}</b>\n"
            f"Risk Exposure: <b>${risk_amount:,.2f}</b>\n"
            f"Expected Return: <b>{expected_return:+.2f}%</b>\n"
            f"Time: <i>{_now_str()}</i>"
        )
        self.enqueue(msg)

    def notify_trade_close(self, pos: RichPosition) -> None:
        net_pnl = pos.net_pnl
        pnl_emoji = "💰 <b>PROFIT</b>" if net_pnl >= 0 else "🔻 <b>LOSS</b>"
        msg = (
            f"🏁 <b>BOT5 «الغني» — POSITION CLOSED</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"Symbol: <code>{pos.symbol}</code>\n"
            f"Status: {pnl_emoji}\n"
            f"Exit Reason: <code>{pos.exit_reason}</code>\n"
            f"Net PnL: <b>${net_pnl:+,.4f}</b>\n"
            f"Gross PnL: ${pos.gross_pnl:+,.4f}\n"
            f"Fees & Slippage: ${pos.fees_paid + pos.slippage_cost:,.4f}\n"
            f"Holding Duration: {pos.holding_seconds():.1f}s\n"
            f"Time: <i>{_now_str()}</i>"
        )
        self.enqueue(msg)

    def notify_kill_switch(self, level: str, reason: str, details: str = "") -> None:
        msg = (
            f"🚨 <b>BOT5 «الغني» — KILL SWITCH TRIGGERED</b> 🚨\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"Level: <b>{level}</b>\n"
            f"Reason: <i>{reason}</i>\n"
            f"{'Details: ' + details if details else ''}\n"
            f"Action: <b>New entries BLOCKED. Unprotected positions handled.</b>\n"
            f"Time: <i>{_now_str()}</i>"
        )
        self.enqueue(msg)

    def notify_protection_failure(self, pos_id: str, symbol: str, reason: str) -> None:
        msg = (
            f"⚠️ <b>BOT5 «الغني» — CRITICAL PROTECTION FAILURE</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"Position: <code>{pos_id}</code>\n"
            f"Symbol: <code>{symbol}</code>\n"
            f"Reason: <i>{reason}</i>\n"
            f"State: <b>UNPROTECTED! Kill switch activated.</b>\n"
            f"Time: <i>{_now_str()}</i>"
        )
        self.enqueue(msg)

    def notify_reconciliation_warning(self, summary: str) -> None:
        msg = (
            f"⚠️ <b>BOT5 «الغني» — RECONCILIATION MISMATCH</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"Summary: <i>{summary}</i>\n"
            f"Time: <i>{_now_str()}</i>"
        )
        self.enqueue(msg)

    def notify_daily_summary(self, stats: Dict[str, Any]) -> None:
        msg = (
            f"📊 <b>BOT5 «الغني» — DAILY PERFORMANCE DIGEST</b>\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"Total Trades: <b>{stats.get('total_trades', 0)}</b>\n"
            f"Win Rate: <b>{stats.get('win_rate_pct', 0.0):.1f}%</b>\n"
            f"Net Realized PnL: <b>${stats.get('net_pnl', Decimal('0')):+,.2f}</b>\n"
            f"Open Positions: <b>{stats.get('open_positions', 0)}</b>\n"
            f"System Health: <b>{stats.get('health_status', 'HEALTHY')}</b>\n"
            f"Time: <i>{_now_str()}</i>"
        )
        self.enqueue(msg)

    def shutdown(self) -> None:
        """Graceful shutdown of worker thread."""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._queue.put(None)
            self._worker_thread.join(timeout=2.0)
