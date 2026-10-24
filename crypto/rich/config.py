"""
rich/config.py — Central configuration for Bot5 (الغني).

ALL thresholds, limits, and tunable parameters live here.
No magic numbers scattered in code.

Environment Variables (set in .env):
  TRADING_MODE          = paper | shadow | live_locked | live
  TRADING_CAPITAL_LIMIT = 50.0
  REINVEST_PROFITS      = false
  MAX_RISK_PER_TRADE    = 0.01
  MAX_DAILY_LOSS        = 0.03
  MAX_OPEN_POSITIONS    = 1
  MAX_SIGNAL_AGE_SECONDS= 300
  ...

On startup, if any REQUIRED critical parameter is missing or invalid,
RichConfig.validate() raises ConfigurationError and the process exits.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional

try:
    from dotenv import load_dotenv
    _BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_BASE, ".env"), override=False)
    load_dotenv(os.path.join(os.path.dirname(_BASE), ".env"), override=False)
except ImportError:
    _BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for env_path in [os.path.join(_BASE, ".env"), os.path.join(os.path.dirname(_BASE), ".env")]:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if k not in os.environ:
                            os.environ[k] = v



class TradingMode(str, Enum):
    PAPER = "paper"          # no real orders; pure simulation
    SHADOW = "shadow"        # reads live market; computes virtual fills; no orders
    LIVE_LOCKED = "live_locked"  # connected to live; reads only; no new orders
    LIVE = "live"            # full live execution


class ConfigurationError(Exception):
    """Raised when the configuration is invalid at startup."""


def _env(name: str, default: Optional[str] = None, required: bool = False) -> str:
    val = os.environ.get(name, default)
    if required and not val:
        raise ConfigurationError(f"Required environment variable {name!r} is not set.")
    return val or ""


def _env_float(name: str, default: float, *, min_val: float = None, max_val: float = None) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        v = float(raw)
    except ValueError:
        raise ConfigurationError(f"Environment variable {name!r} must be a float; got {raw!r}")
    if min_val is not None and v < min_val:
        raise ConfigurationError(f"{name}={v} is below minimum allowed {min_val}")
    if max_val is not None and v > max_val:
        raise ConfigurationError(f"{name}={v} exceeds maximum allowed {max_val}")
    return v


def _env_int(name: str, default: int, *, min_val: int = None) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        v = int(raw)
    except ValueError:
        raise ConfigurationError(f"Environment variable {name!r} must be an integer; got {raw!r}")
    if min_val is not None and v < min_val:
        raise ConfigurationError(f"{name}={v} is below minimum allowed {min_val}")
    return v


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "")
    if not raw:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class RichConfig:
    """Single source of truth for all Bot5 parameters.

    Immutable after validate(). Any runtime change requires a new config
    version, which is logged to the DB and attached to every trade opened
    under it.
    """

    # ── Trading Mode ────────────────────────────────────────────────
    trading_mode: TradingMode = TradingMode.PAPER

    # ── Capital ─────────────────────────────────────────────────────
    trading_capital_limit: float = 50.0   # USDT ceiling
    reinvest_profits: bool = False

    # ── Risk Limits ──────────────────────────────────────────────────
    max_risk_per_trade: float = 0.01      # 1% of capital per trade
    max_daily_loss: float = 0.03          # 3% daily drawdown limit
    max_open_positions: int = 1           # max concurrent positions
    max_account_exposure: float = 0.10    # 10% of capital ceiling
    max_symbol_exposure: float = 0.10     # 10% in any single symbol
    max_slippage_bps: float = 50.0        # 50 basis points max slippage
    min_expected_edge_bps: float = 20.0   # 20 bps minimum expected edge

    # ── Signal Freshness ─────────────────────────────────────────────
    max_signal_age_seconds: int = 300     # 5 minutes

    # ── Expected Edge Gate ───────────────────────────────────────────
    min_expected_edge_after_costs_bps: float = 5.0  # must be >0 after all costs

    # ── Entry Engine ─────────────────────────────────────────────────
    entry_order_type: str = "MARKET"      # MARKET | LIMIT
    limit_order_timeout_seconds: int = 30

    # ── Protection ───────────────────────────────────────────────────
    stop_loss_type: str = "exchange"      # exchange | local
    take_profit_enabled: bool = True
    break_even_enabled: bool = True
    trailing_stop_enabled: bool = False

    # ── Fees (used for edge calculation before fill; actual from exchange) ──
    estimated_taker_fee_bps: float = 10.0  # 0.1% Binance spot taker
    estimated_maker_fee_bps: float = 10.0  # 0.1% Binance spot maker
    safety_margin_bps: float = 5.0

    # ── Reconciliation ───────────────────────────────────────────────
    reconciliation_interval_seconds: int = 120
    reconciliation_tolerance_usdt: float = 0.05   # dust tolerance

    # ── WebSocket ────────────────────────────────────────────────────
    ws_heartbeat_interval_seconds: int = 30
    ws_reconnect_backoff_max_seconds: int = 60

    # ── Health Monitor ───────────────────────────────────────────────
    health_check_interval_seconds: int = 60
    api_latency_warning_ms: float = 2000.0
    server_time_max_offset_ms: float = 1000.0

    # ── API ──────────────────────────────────────────────────────────
    binance_recv_window: int = 5000
    binance_timeout_seconds: int = 10
    max_order_retry_attempts: int = 0    # ZERO blind retries for order placement

    # ── DB ───────────────────────────────────────────────────────────
    db_path: str = ""                    # resolved at validate()

    # ── Versions (set by load()) ─────────────────────────────────────
    configuration_version: str = "0.1.0"
    schema_version: str = "1"

    # ── Telegram ─────────────────────────────────────────────────────
    telegram_rich_bot_token: str = ""
    telegram_admin_chat_id: str = ""

    # ── Binance API ──────────────────────────────────────────────────
    api_key: str = ""
    api_secret: str = ""
    use_testnet: bool = False

    @property
    def telegram_bot_token(self) -> str:
        return self.telegram_rich_bot_token

    @property
    def telegram_chat_id(self) -> str:
        return self.telegram_admin_chat_id


    @classmethod
    def load(cls) -> "RichConfig":
        """Build a RichConfig from environment variables."""
        mode_str = _env("TRADING_MODE", "paper").strip().lower()
        try:
            mode = TradingMode(mode_str)
        except ValueError:
            raise ConfigurationError(
                f"TRADING_MODE={mode_str!r} is invalid. Must be one of: "
                f"{[m.value for m in TradingMode]}"
            )

        cfg = cls(
            trading_mode=mode,
            trading_capital_limit=_env_float("TRADING_CAPITAL_LIMIT", 50.0, min_val=1.0),
            reinvest_profits=_env_bool("REINVEST_PROFITS", False),
            max_risk_per_trade=_env_float("MAX_RISK_PER_TRADE", 0.01, min_val=0.0001, max_val=0.10),
            max_daily_loss=_env_float("MAX_DAILY_LOSS", 0.03, min_val=0.001, max_val=1.0),
            max_open_positions=_env_int("MAX_OPEN_POSITIONS", 1, min_val=1),
            max_account_exposure=_env_float("MAX_ACCOUNT_EXPOSURE", 0.10, min_val=0.001, max_val=1.0),
            max_symbol_exposure=_env_float("MAX_SYMBOL_EXPOSURE", 0.10, min_val=0.001, max_val=1.0),
            max_slippage_bps=_env_float("MAX_SLIPPAGE_BPS", 50.0, min_val=0.0),
            min_expected_edge_bps=_env_float("MIN_EXPECTED_EDGE_BPS", 20.0),
            max_signal_age_seconds=_env_int("MAX_SIGNAL_AGE_SECONDS", 300, min_val=10),
            min_expected_edge_after_costs_bps=_env_float("MIN_EXPECTED_EDGE_AFTER_COSTS_BPS", 5.0),
            entry_order_type=_env("ENTRY_ORDER_TYPE", "MARKET").upper(),
            limit_order_timeout_seconds=_env_int("LIMIT_ORDER_TIMEOUT_SECONDS", 30, min_val=5),
            stop_loss_type=_env("STOP_LOSS_TYPE", "exchange").lower(),
            take_profit_enabled=_env_bool("TAKE_PROFIT_ENABLED", True),
            break_even_enabled=_env_bool("BREAK_EVEN_ENABLED", True),
            trailing_stop_enabled=_env_bool("TRAILING_STOP_ENABLED", False),
            estimated_taker_fee_bps=_env_float("ESTIMATED_TAKER_FEE_BPS", 10.0, min_val=0.0),
            estimated_maker_fee_bps=_env_float("ESTIMATED_MAKER_FEE_BPS", 10.0, min_val=0.0),
            safety_margin_bps=_env_float("SAFETY_MARGIN_BPS", 5.0, min_val=0.0),
            reconciliation_interval_seconds=_env_int("RECONCILIATION_INTERVAL_SECONDS", 120, min_val=30),
            reconciliation_tolerance_usdt=_env_float("RECONCILIATION_TOLERANCE_USDT", 0.05, min_val=0.0),
            ws_heartbeat_interval_seconds=_env_int("WS_HEARTBEAT_INTERVAL_SECONDS", 30, min_val=10),
            ws_reconnect_backoff_max_seconds=_env_int("WS_RECONNECT_BACKOFF_MAX_SECONDS", 60, min_val=5),
            health_check_interval_seconds=_env_int("HEALTH_CHECK_INTERVAL_SECONDS", 60, min_val=15),
            api_latency_warning_ms=_env_float("API_LATENCY_WARNING_MS", 2000.0, min_val=0.0),
            server_time_max_offset_ms=_env_float("SERVER_TIME_MAX_OFFSET_MS", 1000.0, min_val=0.0),
            binance_recv_window=_env_int("BINANCE_RECV_WINDOW", 5000, min_val=1000),
            binance_timeout_seconds=_env_int("BINANCE_TIMEOUT_SECONDS", 10, min_val=3),
            max_order_retry_attempts=0,   # HARD ZERO — never blind retry orders
            configuration_version=_env("RICH_CONFIG_VERSION", "0.1.0"),
            schema_version="1",
            # Secrets — loaded but never logged
            telegram_rich_bot_token=_env("TELEGRAM_RICH_BOT_TOKEN", _env("BOT5_TOKEN", "")),
            telegram_admin_chat_id=_env("TELEGRAM_ADMIN_CHAT_ID", _env("TELEGRAM_CHAT_ID", "-1004373560971")),

        )

        # DB path — default to crypto/data/rich.db
        _data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        os.makedirs(_data_dir, exist_ok=True)
        cfg.db_path = _env("RICH_DB_PATH", os.path.join(_data_dir, "rich.db"))

        return cfg

    def validate(self) -> None:
        """Raise ConfigurationError if any constraint is violated."""
        if self.entry_order_type not in ("MARKET", "LIMIT"):
            raise ConfigurationError(f"ENTRY_ORDER_TYPE must be MARKET or LIMIT; got {self.entry_order_type!r}")
        if self.stop_loss_type not in ("exchange", "local"):
            raise ConfigurationError(f"STOP_LOSS_TYPE must be 'exchange' or 'local'; got {self.stop_loss_type!r}")
        if self.max_risk_per_trade > self.max_account_exposure:
            raise ConfigurationError(
                f"MAX_RISK_PER_TRADE ({self.max_risk_per_trade}) cannot exceed "
                f"MAX_ACCOUNT_EXPOSURE ({self.max_account_exposure})"
            )
        # LIVE mode requires proper secrets
        if self.trading_mode == TradingMode.LIVE:
            trading_key = _env("BINANCE_TRADING_API_KEY", "")
            trading_secret = _env("BINANCE_TRADING_API_SECRET", "")
            if not trading_key or not trading_secret:
                raise ConfigurationError(
                    "TRADING_MODE=live requires BINANCE_TRADING_API_KEY and "
                    "BINANCE_TRADING_API_SECRET to be set."
                )

    def is_live(self) -> bool:
        return self.trading_mode == TradingMode.LIVE

    def can_place_orders(self) -> bool:
        return self.trading_mode in (TradingMode.LIVE, TradingMode.PAPER, TradingMode.SHADOW)

    def is_real_exchange(self) -> bool:
        """True when connected to the real Binance API (LIVE or LIVE_LOCKED)."""
        return self.trading_mode in (TradingMode.LIVE, TradingMode.LIVE_LOCKED)

    def to_safe_dict(self) -> dict:
        """Return a dict safe to log (no secrets)."""
        return {
            "trading_mode": self.trading_mode.value,
            "trading_capital_limit": self.trading_capital_limit,
            "reinvest_profits": self.reinvest_profits,
            "max_risk_per_trade": self.max_risk_per_trade,
            "max_daily_loss": self.max_daily_loss,
            "max_open_positions": self.max_open_positions,
            "max_account_exposure": self.max_account_exposure,
            "max_symbol_exposure": self.max_symbol_exposure,
            "max_slippage_bps": self.max_slippage_bps,
            "min_expected_edge_bps": self.min_expected_edge_bps,
            "max_signal_age_seconds": self.max_signal_age_seconds,
            "min_expected_edge_after_costs_bps": self.min_expected_edge_after_costs_bps,
            "entry_order_type": self.entry_order_type,
            "estimated_taker_fee_bps": self.estimated_taker_fee_bps,
            "estimated_maker_fee_bps": self.estimated_maker_fee_bps,
            "safety_margin_bps": self.safety_margin_bps,
            "configuration_version": self.configuration_version,
            "schema_version": self.schema_version,
        }
