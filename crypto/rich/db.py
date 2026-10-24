"""
rich/db.py — SQLite persistence layer for Bot5.

Source of truth for local state. Binance is the exchange-side truth.
When they differ → DESYNC.

Schema version is tracked in user_version pragma.
Migrations are applied forward-only — never DROP columns or tables.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("rich.db")

SCHEMA_VERSION = 1

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- ── Signals ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS signals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id           TEXT    NOT NULL UNIQUE,
    signal_version      TEXT    NOT NULL DEFAULT '1.0',
    strategy_version    TEXT    NOT NULL DEFAULT 'unknown',
    source              TEXT    NOT NULL DEFAULT 'BOT4_MASTER',
    symbol              TEXT    NOT NULL,
    side                TEXT    NOT NULL CHECK(side IN ('BUY','SELL')),
    signal_timestamp    TEXT    NOT NULL,
    confidence          REAL    NOT NULL DEFAULT 0,
    predicted_change_pct REAL   NOT NULL DEFAULT 0,
    expected_vol_pct    REAL    NOT NULL DEFAULT 0,
    relative_bias       TEXT    NOT NULL DEFAULT '',
    market_regime       TEXT    NOT NULL DEFAULT 'UNKNOWN',
    entry_reference     REAL    NOT NULL DEFAULT 0,
    stop_reference      REAL    NOT NULL DEFAULT 0,
    target_reference    REAL    NOT NULL DEFAULT 0,
    strategy            TEXT    NOT NULL DEFAULT '',
    validation_status   TEXT    NOT NULL DEFAULT 'VALID',
    validation_reason   TEXT    NOT NULL DEFAULT '',
    metadata_json       TEXT    NOT NULL DEFAULT '{}',
    received_at         TEXT    NOT NULL DEFAULT (datetime('now')),
    configuration_version TEXT  NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS ix_signals_symbol ON signals(symbol);
CREATE INDEX IF NOT EXISTS ix_signals_ts     ON signals(signal_timestamp);

-- ── Orders ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS orders (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    internal_order_id   TEXT    NOT NULL UNIQUE,
    client_order_id     TEXT    NOT NULL UNIQUE,
    exchange_order_id   TEXT,
    signal_id           TEXT    NOT NULL REFERENCES signals(signal_id),
    position_id         TEXT,
    symbol              TEXT    NOT NULL,
    side                TEXT    NOT NULL CHECK(side IN ('BUY','SELL')),
    order_type          TEXT    NOT NULL DEFAULT 'MARKET',
    requested_qty       TEXT    NOT NULL DEFAULT '0',
    requested_price     TEXT,
    executed_qty        TEXT    NOT NULL DEFAULT '0',
    average_fill_price  TEXT,
    state               TEXT    NOT NULL DEFAULT 'SIGNAL_RECEIVED',
    created_at          TEXT    NOT NULL,
    updated_at          TEXT    NOT NULL,
    submitted_at        TEXT,
    filled_at           TEXT,
    reject_reason       TEXT    NOT NULL DEFAULT '',
    raw_response_json   TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_orders_signal   ON orders(signal_id);
CREATE INDEX IF NOT EXISTS ix_orders_position ON orders(position_id);
CREATE INDEX IF NOT EXISTS ix_orders_state    ON orders(state);
CREATE INDEX IF NOT EXISTS ix_orders_symbol   ON orders(symbol);

-- ── Order Events (state machine transitions) ─────────────────────
CREATE TABLE IF NOT EXISTS order_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id        TEXT    NOT NULL,
    event_type      TEXT    NOT NULL,
    from_state      TEXT    NOT NULL DEFAULT '',
    to_state        TEXT    NOT NULL DEFAULT '',
    details_json    TEXT    NOT NULL DEFAULT '{}',
    occurred_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_order_events_order ON order_events(order_id);

-- ── Fills ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS fills (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    fill_id             TEXT    NOT NULL UNIQUE,
    order_id            TEXT    NOT NULL,
    exchange_trade_id   TEXT    NOT NULL DEFAULT '',
    symbol              TEXT    NOT NULL,
    side                TEXT    NOT NULL,
    quantity            TEXT    NOT NULL,
    price               TEXT    NOT NULL,
    commission          TEXT    NOT NULL DEFAULT '0',
    commission_asset    TEXT    NOT NULL DEFAULT 'USDT',
    is_maker            INTEGER NOT NULL DEFAULT 0,
    filled_at           TEXT    NOT NULL,
    created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_fills_order  ON fills(order_id);
CREATE INDEX IF NOT EXISTS ix_fills_symbol ON fills(symbol);

-- ── Positions ─────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS positions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id             TEXT    NOT NULL UNIQUE,
    signal_id               TEXT    NOT NULL,
    symbol                  TEXT    NOT NULL,
    side                    TEXT    NOT NULL,
    quantity                TEXT    NOT NULL DEFAULT '0',
    average_entry           TEXT,
    current_price           TEXT,
    unrealized_pnl          TEXT    NOT NULL DEFAULT '0',
    realized_pnl            TEXT    NOT NULL DEFAULT '0',
    fees_paid               TEXT    NOT NULL DEFAULT '0',
    slippage_cost           TEXT    NOT NULL DEFAULT '0',
    stop_loss               TEXT,
    stop_order_id           TEXT    NOT NULL DEFAULT '',
    take_profit_1           TEXT,
    take_profit_2           TEXT,
    take_profit_3           TEXT,
    tp_order_ids_json       TEXT    NOT NULL DEFAULT '[]',
    protection_state        TEXT    NOT NULL DEFAULT 'NONE',
    break_even_activated    INTEGER NOT NULL DEFAULT 0,
    trailing_active         INTEGER NOT NULL DEFAULT 0,
    position_state          TEXT    NOT NULL DEFAULT 'PENDING_ENTRY',
    opened_at               TEXT,
    closed_at               TEXT,
    exit_reason             TEXT    NOT NULL DEFAULT '',
    gross_pnl               TEXT    NOT NULL DEFAULT '0',
    net_pnl                 TEXT    NOT NULL DEFAULT '0',
    mae                     TEXT,
    mfe                     TEXT,
    signal_expected_return_pct REAL NOT NULL DEFAULT 0,
    configuration_version   TEXT    NOT NULL DEFAULT '',
    strategy_version        TEXT    NOT NULL DEFAULT '',
    created_at              TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at              TEXT    NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_positions_state  ON positions(position_state);
CREATE INDEX IF NOT EXISTS ix_positions_symbol ON positions(symbol);

-- ── Position Events ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS position_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    position_id     TEXT    NOT NULL,
    event_type      TEXT    NOT NULL,
    from_state      TEXT    NOT NULL DEFAULT '',
    to_state        TEXT    NOT NULL DEFAULT '',
    details_json    TEXT    NOT NULL DEFAULT '{}',
    occurred_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ── Risk Events ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS risk_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type      TEXT    NOT NULL,
    signal_id       TEXT    NOT NULL DEFAULT '',
    decision        TEXT    NOT NULL,
    reason          TEXT    NOT NULL,
    details_json    TEXT    NOT NULL DEFAULT '{}',
    occurred_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ── Reconciliation Events ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS reconciliation_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    result          TEXT    NOT NULL,
    discrepancies_json TEXT NOT NULL DEFAULT '[]',
    local_positions_json TEXT NOT NULL DEFAULT '[]',
    exchange_positions_json TEXT NOT NULL DEFAULT '[]',
    notes_json      TEXT    NOT NULL DEFAULT '[]',
    triggered_by    TEXT    NOT NULL DEFAULT 'periodic',
    occurred_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ── System Events ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS system_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type      TEXT    NOT NULL,
    severity        TEXT    NOT NULL DEFAULT 'INFO',
    component       TEXT    NOT NULL DEFAULT '',
    state_before    TEXT    NOT NULL DEFAULT '',
    state_after     TEXT    NOT NULL DEFAULT '',
    details_json    TEXT    NOT NULL DEFAULT '{}',
    message         TEXT    NOT NULL DEFAULT '',
    occurred_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ── Daily Metrics ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS daily_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT    NOT NULL UNIQUE,
    total_trades    INTEGER NOT NULL DEFAULT 0,
    wins            INTEGER NOT NULL DEFAULT 0,
    losses          INTEGER NOT NULL DEFAULT 0,
    gross_pnl       TEXT    NOT NULL DEFAULT '0',
    fees            TEXT    NOT NULL DEFAULT '0',
    slippage        TEXT    NOT NULL DEFAULT '0',
    net_pnl         TEXT    NOT NULL DEFAULT '0',
    max_drawdown    TEXT    NOT NULL DEFAULT '0',
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ── Trade Journal ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS trade_journal (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id                    TEXT    NOT NULL UNIQUE,
    signal_id                   TEXT    NOT NULL,
    signal_version              TEXT    NOT NULL DEFAULT '',
    strategy_version            TEXT    NOT NULL DEFAULT '',
    configuration_version       TEXT    NOT NULL DEFAULT '',
    symbol                      TEXT    NOT NULL,
    side                        TEXT    NOT NULL,
    signal_price                TEXT,
    reference_price             TEXT,
    requested_entry             TEXT,
    actual_entry                TEXT,
    requested_quantity          TEXT,
    executed_quantity           TEXT,
    average_fill_price          TEXT,
    stop_loss                   TEXT,
    take_profit_1               TEXT,
    fees                        TEXT    NOT NULL DEFAULT '0',
    slippage_cost               TEXT    NOT NULL DEFAULT '0',
    gross_pnl                   TEXT    NOT NULL DEFAULT '0',
    net_pnl                     TEXT    NOT NULL DEFAULT '0',
    signal_expected_return_pct  REAL    NOT NULL DEFAULT 0,
    opened_at                   TEXT,
    closed_at                   TEXT,
    holding_seconds             REAL    NOT NULL DEFAULT 0,
    mae                         TEXT,
    mfe                         TEXT,
    market_regime               TEXT    NOT NULL DEFAULT '',
    relative_bias               TEXT    NOT NULL DEFAULT '',
    confidence                  REAL    NOT NULL DEFAULT 0,
    exit_reason                 TEXT    NOT NULL DEFAULT '',
    created_at                  TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ── Health Events ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS health_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    component       TEXT    NOT NULL,
    status          TEXT    NOT NULL,
    details_json    TEXT    NOT NULL DEFAULT '{}',
    occurred_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ── Telegram Events ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS telegram_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    message_type    TEXT    NOT NULL,
    chat_id         TEXT    NOT NULL DEFAULT '',
    preview         TEXT    NOT NULL DEFAULT '',
    status          TEXT    NOT NULL DEFAULT 'sent',
    occurred_at     TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ── Configuration Versions ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS configuration_versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    version         TEXT    NOT NULL UNIQUE,
    config_json     TEXT    NOT NULL,
    activated_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RichDB:
    """
    Thread-safe SQLite wrapper for Bot5.

    Uses WAL mode for concurrent reads.
    All mutations use explicit transactions.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        """Return a per-thread connection."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def _init_db(self) -> None:
        conn = self._conn()
        conn.executescript(_SCHEMA_SQL)
        # Check/set schema version
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version == 0:
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.commit()
            logger.info(f"[DB] Initialized rich.db schema v{SCHEMA_VERSION}")
        elif version < SCHEMA_VERSION:
            self._migrate(conn, version)
        logger.info(f"[DB] rich.db ready at {self._db_path}")

    def _migrate(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Forward-only migrations. Extend when SCHEMA_VERSION bumps."""
        logger.warning(f"[DB] Schema migration needed: v{from_version} → v{SCHEMA_VERSION}")
        # Future migrations go here as elif from_version == N: blocks
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()

    # ── Signal Operations ─────────────────────────────────────────────────────

    def save_signal(self, sig: dict) -> bool:
        """Persist a signal. Returns True if inserted (new), False if duplicate."""
        conn = self._conn()
        try:
            conn.execute("""
                INSERT INTO signals
                  (signal_id, signal_version, strategy_version, source,
                   symbol, side, signal_timestamp, confidence,
                   predicted_change_pct, expected_vol_pct, relative_bias,
                   market_regime, entry_reference, stop_reference,
                   target_reference, strategy, validation_status,
                   validation_reason, metadata_json, received_at,
                   configuration_version)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                sig["signal_id"], sig.get("signal_version", "1.0"),
                sig.get("strategy_version", "unknown"), sig.get("source", "BOT4_MASTER"),
                sig["symbol"], sig["side"], sig["signal_timestamp"],
                sig.get("confidence", 0), sig.get("predicted_change_pct", 0),
                sig.get("expected_vol_pct", 0), sig.get("relative_bias", ""),
                sig.get("market_regime", "UNKNOWN"), sig.get("entry_reference", 0),
                sig.get("stop_reference", 0), sig.get("target_reference", 0),
                sig.get("strategy", ""), sig.get("validation_status", "VALID"),
                sig.get("validation_reason", ""),
                json.dumps(sig.get("metadata", {}), default=str),
                _now_iso(), sig.get("configuration_version", ""),
            ))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # duplicate signal_id

    def signal_exists(self, signal_id: str) -> bool:
        conn = self._conn()
        row = conn.execute(
            "SELECT 1 FROM signals WHERE signal_id=?", (signal_id,)
        ).fetchone()
        return row is not None

    # ── Order Operations ──────────────────────────────────────────────────────

    def save_order(self, order: dict) -> None:
        conn = self._conn()
        now = _now_iso()
        conn.execute("""
            INSERT INTO orders
              (internal_order_id, client_order_id, exchange_order_id,
               signal_id, position_id, symbol, side, order_type,
               requested_qty, requested_price, executed_qty,
               average_fill_price, state, created_at, updated_at,
               submitted_at, filled_at, reject_reason, raw_response_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(internal_order_id) DO UPDATE SET
               exchange_order_id=excluded.exchange_order_id,
               executed_qty=excluded.executed_qty,
               average_fill_price=excluded.average_fill_price,
               state=excluded.state,
               updated_at=excluded.updated_at,
               submitted_at=excluded.submitted_at,
               filled_at=excluded.filled_at,
               reject_reason=excluded.reject_reason,
               raw_response_json=excluded.raw_response_json
        """, (
            order["internal_order_id"], order["client_order_id"],
            order.get("exchange_order_id", ""),
            order["signal_id"], order.get("position_id", ""),
            order["symbol"], order["side"], order.get("order_type", "MARKET"),
            str(order.get("requested_qty", "0")), str(order.get("requested_price")) if order.get("requested_price") else None,
            str(order.get("executed_qty", "0")),
            str(order.get("average_fill_price")) if order.get("average_fill_price") else None,
            order.get("state", "SIGNAL_RECEIVED"), order.get("created_at", now), now,
            order.get("submitted_at"), order.get("filled_at"),
            order.get("reject_reason", ""),
            json.dumps(order.get("raw_response", {}), default=str),
        ))
        conn.commit()

    def update_order_state(self, internal_order_id: str, new_state: str,
                           details: Optional[dict] = None) -> None:
        conn = self._conn()
        now = _now_iso()
        # Get current state
        row = conn.execute(
            "SELECT state FROM orders WHERE internal_order_id=?", (internal_order_id,)
        ).fetchone()
        old_state = row["state"] if row else ""
        conn.execute(
            "UPDATE orders SET state=?, updated_at=? WHERE internal_order_id=?",
            (new_state, now, internal_order_id)
        )
        # Record transition
        conn.execute("""
            INSERT INTO order_events (order_id, event_type, from_state, to_state, details_json, occurred_at)
            VALUES (?, 'STATE_TRANSITION', ?, ?, ?, ?)
        """, (internal_order_id, old_state, new_state,
              json.dumps(details or {}, default=str), now))
        conn.commit()

    def get_order_by_client_id(self, client_order_id: str) -> Optional[dict]:
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM orders WHERE client_order_id=?", (client_order_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_open_orders(self) -> List[dict]:
        conn = self._conn()
        rows = conn.execute("""
            SELECT * FROM orders WHERE state NOT IN
            ('FILLED','REJECTED','CANCELLED','EXPIRED','CLOSED')
        """).fetchall()
        return [dict(r) for r in rows]

    # ── Fill Operations ───────────────────────────────────────────────────────

    def save_fill(self, fill: dict) -> None:
        conn = self._conn()
        conn.execute("""
            INSERT OR IGNORE INTO fills
              (fill_id, order_id, exchange_trade_id, symbol, side,
               quantity, price, commission, commission_asset, is_maker, filled_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            fill["fill_id"], fill["order_id"], fill.get("exchange_trade_id", ""),
            fill["symbol"], fill["side"],
            str(fill["quantity"]), str(fill["price"]),
            str(fill.get("commission", "0")), fill.get("commission_asset", "USDT"),
            int(fill.get("is_maker", False)), fill["filled_at"],
        ))
        conn.commit()

    # ── Position Operations ───────────────────────────────────────────────────

    def save_position(self, pos: dict) -> None:
        conn = self._conn()
        now = _now_iso()
        conn.execute("""
            INSERT INTO positions
              (position_id, signal_id, symbol, side, quantity, average_entry,
               current_price, unrealized_pnl, realized_pnl, fees_paid,
               slippage_cost, stop_loss, stop_order_id, take_profit_1,
               take_profit_2, take_profit_3, tp_order_ids_json,
               protection_state, break_even_activated, trailing_active,
               position_state, opened_at, closed_at, exit_reason,
               gross_pnl, net_pnl, mae, mfe, signal_expected_return_pct,
               configuration_version, strategy_version, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(position_id) DO UPDATE SET
               quantity=excluded.quantity,
               average_entry=excluded.average_entry,
               current_price=excluded.current_price,
               unrealized_pnl=excluded.unrealized_pnl,
               realized_pnl=excluded.realized_pnl,
               fees_paid=excluded.fees_paid,
               slippage_cost=excluded.slippage_cost,
               stop_loss=excluded.stop_loss,
               stop_order_id=excluded.stop_order_id,
               take_profit_1=excluded.take_profit_1,
               take_profit_2=excluded.take_profit_2,
               take_profit_3=excluded.take_profit_3,
               tp_order_ids_json=excluded.tp_order_ids_json,
               protection_state=excluded.protection_state,
               break_even_activated=excluded.break_even_activated,
               trailing_active=excluded.trailing_active,
               position_state=excluded.position_state,
               closed_at=excluded.closed_at,
               exit_reason=excluded.exit_reason,
               gross_pnl=excluded.gross_pnl,
               net_pnl=excluded.net_pnl,
               mae=excluded.mae,
               mfe=excluded.mfe,
               updated_at=excluded.updated_at
        """, (
            pos["position_id"], pos["signal_id"], pos["symbol"], pos["side"],
            str(pos.get("quantity", "0")), str(pos.get("average_entry")) if pos.get("average_entry") else None,
            str(pos.get("current_price")) if pos.get("current_price") else None,
            str(pos.get("unrealized_pnl", "0")), str(pos.get("realized_pnl", "0")),
            str(pos.get("fees_paid", "0")), str(pos.get("slippage_cost", "0")),
            str(pos.get("stop_loss")) if pos.get("stop_loss") else None,
            pos.get("stop_order_id", ""),
            str(pos.get("take_profit_1")) if pos.get("take_profit_1") else None,
            str(pos.get("take_profit_2")) if pos.get("take_profit_2") else None,
            str(pos.get("take_profit_3")) if pos.get("take_profit_3") else None,
            json.dumps(pos.get("tp_order_ids", []), default=str),
            pos.get("protection_state", "NONE"),
            int(pos.get("break_even_activated", False)),
            int(pos.get("trailing_active", False)),
            pos.get("position_state", "PENDING_ENTRY"),
            pos.get("opened_at"), pos.get("closed_at"), pos.get("exit_reason", ""),
            str(pos.get("gross_pnl", "0")), str(pos.get("net_pnl", "0")),
            str(pos.get("mae")) if pos.get("mae") else None,
            str(pos.get("mfe")) if pos.get("mfe") else None,
            float(pos.get("signal_expected_return_pct", 0)),
            pos.get("configuration_version", ""), pos.get("strategy_version", ""), now,
        ))
        conn.commit()

    def get_open_positions(self) -> List[dict]:
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM positions WHERE position_state NOT IN ('CLOSED')"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_position(self, position_id: str) -> Optional[dict]:
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM positions WHERE position_id=?", (position_id,)
        ).fetchone()
        return dict(row) if row else None

    # ── System Events ─────────────────────────────────────────────────────────

    def log_system_event(self, event_type: str, message: str, severity: str = "INFO",
                         component: str = "", state_before: str = "",
                         state_after: str = "", details: Optional[dict] = None) -> None:
        conn = self._conn()
        conn.execute("""
            INSERT INTO system_events
              (event_type, severity, component, state_before, state_after, details_json, message)
            VALUES (?,?,?,?,?,?,?)
        """, (event_type, severity, component, state_before, state_after,
              json.dumps(details or {}, default=str), message))
        conn.commit()

    # ── Risk Events ───────────────────────────────────────────────────────────

    def log_risk_event(self, event_type: str, decision: str, reason: str,
                       signal_id: str = "", details: Optional[dict] = None) -> None:
        conn = self._conn()
        conn.execute("""
            INSERT INTO risk_events (event_type, signal_id, decision, reason, details_json)
            VALUES (?,?,?,?,?)
        """, (event_type, signal_id, decision, reason,
              json.dumps(details or {}, default=str)))
        conn.commit()

    # ── Reconciliation Events ─────────────────────────────────────────────────

    def log_reconciliation(self, result: str, discrepancies: list,
                           local_positions: list, exchange_positions: list,
                           notes: list, triggered_by: str = "periodic") -> None:
        conn = self._conn()
        conn.execute("""
            INSERT INTO reconciliation_events
              (result, discrepancies_json, local_positions_json,
               exchange_positions_json, notes_json, triggered_by)
            VALUES (?,?,?,?,?,?)
        """, (result, json.dumps(discrepancies, default=str),
              json.dumps(local_positions, default=str),
              json.dumps(exchange_positions, default=str),
              json.dumps(notes, default=str), triggered_by))
        conn.commit()

    # ── Trade Journal ─────────────────────────────────────────────────────────

    def save_trade_journal(self, tj: dict) -> None:
        conn = self._conn()
        conn.execute("""
            INSERT OR REPLACE INTO trade_journal
              (trade_id, signal_id, signal_version, strategy_version,
               configuration_version, symbol, side, signal_price,
               reference_price, requested_entry, actual_entry,
               requested_quantity, executed_quantity, average_fill_price,
               stop_loss, take_profit_1, fees, slippage_cost,
               gross_pnl, net_pnl, signal_expected_return_pct,
               opened_at, closed_at, holding_seconds, mae, mfe,
               market_regime, relative_bias, confidence, exit_reason)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            tj["trade_id"], tj["signal_id"], tj.get("signal_version", ""),
            tj.get("strategy_version", ""), tj.get("configuration_version", ""),
            tj["symbol"], tj["side"],
            str(tj.get("signal_price")) if tj.get("signal_price") else None,
            str(tj.get("reference_price")) if tj.get("reference_price") else None,
            str(tj.get("requested_entry")) if tj.get("requested_entry") else None,
            str(tj.get("actual_entry")) if tj.get("actual_entry") else None,
            str(tj.get("requested_quantity")) if tj.get("requested_quantity") else None,
            str(tj.get("executed_quantity")) if tj.get("executed_quantity") else None,
            str(tj.get("average_fill_price")) if tj.get("average_fill_price") else None,
            str(tj.get("stop_loss")) if tj.get("stop_loss") else None,
            str(tj.get("take_profit_1")) if tj.get("take_profit_1") else None,
            str(tj.get("fees", "0")), str(tj.get("slippage_cost", "0")),
            str(tj.get("gross_pnl", "0")), str(tj.get("net_pnl", "0")),
            float(tj.get("signal_expected_return_pct", 0)),
            tj.get("opened_at"), tj.get("closed_at"), float(tj.get("holding_seconds", 0)),
            str(tj.get("mae")) if tj.get("mae") else None,
            str(tj.get("mfe")) if tj.get("mfe") else None,
            tj.get("market_regime", ""), tj.get("relative_bias", ""),
            float(tj.get("confidence", 0)), tj.get("exit_reason", ""),
        ))
        conn.commit()

    def get_today_pnl(self) -> float:
        """Sum of net_pnl for trades closed today (UTC)."""
        conn = self._conn()
        today = datetime.now(timezone.utc).date().isoformat()
        row = conn.execute("""
            SELECT COALESCE(SUM(CAST(net_pnl AS REAL)), 0) as total
            FROM trade_journal
            WHERE DATE(closed_at) = ?
        """, (today,)).fetchone()
        return float(row["total"]) if row else 0.0

    # ── Configuration Versioning ──────────────────────────────────────────────

    def save_config_version(self, version: str, config_dict: dict) -> None:
        conn = self._conn()
        conn.execute("""
            INSERT OR IGNORE INTO configuration_versions (version, config_json)
            VALUES (?,?)
        """, (version, json.dumps(config_dict, default=str)))
        conn.commit()

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None
