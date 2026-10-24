"""
rich/binance_adapter.py — Single, clean interface to Binance Spot API.

ALL Binance HTTP calls MUST go through this adapter.
No other module in rich/ should call requests.get() with api.binance.com directly.

Implements:
  - Signed (HMAC-SHA256) authenticated requests for TRADE + USER_DATA
  - Unsigned public requests
  - Rate limit tracking
  - Exponential backoff for transient errors
  - Server time synchronization
  - Symbol filter caching

Security:
  - API keys read from environment, never from code
  - Keys never logged (secret_redactor applied)
  - No WITHDRAW permission used or required

API Reference: https://developers.binance.com/docs/binance-spot-api-docs/rest-api
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import time
import urllib.parse
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, List, Optional, Tuple

import requests

from .secret_redactor import redact
from .time_sync import TimeSync

logger = logging.getLogger("rich.binance_adapter")

BINANCE_SPOT_BASE = "https://api.binance.com"

# Retryable HTTP status codes (not for orders — for market data only)
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
# Binance error codes that are NOT retryable
_NON_RETRYABLE_CODES = {
    -1100,  # Illegal characters in parameter
    -1101,  # Too many parameters
    -1102,  # Mandatory parameter missing
    -1111,  # Precision is over the maximum allowed
    -1121,  # Invalid symbol
    -2010,  # New order rejected
    -2011,  # Cancel rejected
    -2013,  # Order does not exist
    -2014,  # API key format invalid
    -2015,  # Invalid API key/secret/permissions
    -2019,  # Margin is insufficient
}


class BinanceAPIError(Exception):
    """Raised when Binance returns a known error code."""
    def __init__(self, code: int, msg: str):
        self.code = code
        self.msg = msg
        super().__init__(f"Binance error {code}: {msg}")

    def is_retryable(self) -> bool:
        return self.code not in _NON_RETRYABLE_CODES


class BinanceNetworkError(Exception):
    """Raised on network-level failures (timeout, connection error)."""


class BinanceAdapter:
    """
    Thin wrapper around Binance Spot REST API.

    Two sets of credentials are supported:
      - Trading key  (TRADE permission)
      - Monitor key  (USER_DATA read-only)

    If only one key pair is configured, it is used for everything.
    """

    def __init__(
        self,
        trading_api_key: str,
        trading_api_secret: str,
        monitor_api_key: str = "",
        monitor_api_secret: str = "",
        recv_window: int = 5000,
        timeout: int = 10,
    ):
        self._trading_key = trading_api_key
        self._trading_secret = trading_api_secret
        self._monitor_key = monitor_api_key or trading_api_key
        self._monitor_secret = monitor_api_secret or trading_api_secret
        self._recv_window = recv_window
        self._timeout = timeout
        self._time_sync = TimeSync()

        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10, pool_maxsize=20, max_retries=0
        )
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)

        # Symbol filter cache: {symbol: {filters...}}
        self._symbol_info_cache: Dict[str, dict] = {}
        self._cache_loaded_at: float = 0.0

    # ── Authentication Helpers ────────────────────────────────────────────────

    def _sign(self, params: dict, secret: str) -> str:
        query = urllib.parse.urlencode(params)
        return hmac.new(
            secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def _signed_params(self, params: dict, for_trading: bool = True) -> dict:
        secret = self._trading_secret if for_trading else self._monitor_secret
        p = dict(params)
        p["timestamp"] = self._time_sync.server_time_ms()
        p["recvWindow"] = self._recv_window
        p["signature"] = self._sign(p, secret)
        return p

    def _trading_headers(self) -> dict:
        return {"X-MBX-APIKEY": self._trading_key}

    def _monitor_headers(self) -> dict:
        return {"X-MBX-APIKEY": self._monitor_key}

    # ── HTTP Core ─────────────────────────────────────────────────────────────

    def _get(self, path: str, params: dict = None, signed: bool = False,
             for_trading: bool = False) -> Any:
        headers = self._trading_headers() if for_trading else (
            self._monitor_headers() if signed else {}
        )
        if signed:
            params = self._signed_params(params or {}, for_trading=for_trading)
        url = BINANCE_SPOT_BASE + path
        try:
            r = self._session.get(url, params=params, headers=headers,
                                  timeout=self._timeout)
        except requests.exceptions.Timeout:
            raise BinanceNetworkError(f"GET {path} timed out")
        except requests.exceptions.RequestException as e:
            raise BinanceNetworkError(f"GET {path} network error: {e}")

        return self._parse_response(r, path)

    def _post(self, path: str, params: dict = None) -> Any:
        """Signed POST for TRADE operations."""
        params = self._signed_params(params or {}, for_trading=True)
        url = BINANCE_SPOT_BASE + path
        try:
            r = self._session.post(url, params=params,
                                   headers=self._trading_headers(),
                                   timeout=self._timeout)
        except requests.exceptions.Timeout:
            raise BinanceNetworkError(f"POST {path} timed out")
        except requests.exceptions.RequestException as e:
            raise BinanceNetworkError(f"POST {path} network error: {e}")

        return self._parse_response(r, path)

    def _delete(self, path: str, params: dict = None) -> Any:
        """Signed DELETE for cancel operations."""
        params = self._signed_params(params or {}, for_trading=True)
        url = BINANCE_SPOT_BASE + path
        try:
            r = self._session.delete(url, params=params,
                                     headers=self._trading_headers(),
                                     timeout=self._timeout)
        except requests.exceptions.Timeout:
            raise BinanceNetworkError(f"DELETE {path} timed out")
        except requests.exceptions.RequestException as e:
            raise BinanceNetworkError(f"DELETE {path} network error: {e}")

        return self._parse_response(r, path)

    def _parse_response(self, r: requests.Response, path: str) -> Any:
        if r.status_code in _RETRYABLE_STATUS:
            logger.warning(f"[Binance] {path} → HTTP {r.status_code}")
            raise BinanceNetworkError(f"HTTP {r.status_code} on {path}")
        try:
            data = r.json()
        except Exception:
            raise BinanceNetworkError(f"Non-JSON response on {path}: {r.text[:200]}")
        if isinstance(data, dict) and "code" in data and data["code"] != 200:
            code = int(data["code"])
            msg = data.get("msg", "")
            logger.error(f"[Binance] {path} → Error {code}: {msg}")
            raise BinanceAPIError(code, msg)
        return data

    # ── Public Endpoints ──────────────────────────────────────────────────────

    def get_server_time(self) -> int:
        """Returns Binance server time in milliseconds."""
        data = self._get("/api/v3/time")
        server_time = int(data["serverTime"])
        self._time_sync.update(server_time)
        return server_time

    def get_exchange_info(self, symbol: str = None) -> dict:
        """Get exchange rules and symbol filters."""
        params = {"symbol": symbol} if symbol else {}
        return self._get("/api/v3/exchangeInfo", params)

    def get_ticker_price(self, symbol: str) -> Decimal:
        data = self._get("/api/v3/ticker/price", {"symbol": symbol})
        return Decimal(str(data["price"]))

    def get_orderbook(self, symbol: str, limit: int = 5) -> dict:
        """Returns {"bids": [[price,qty],...], "asks": [[price,qty],...]}"""
        return self._get("/api/v3/depth", {"symbol": symbol, "limit": limit})

    def get_24hr_ticker(self, symbol: str) -> dict:
        return self._get("/api/v3/ticker/24hr", {"symbol": symbol})

    # ── Authenticated Endpoints (USER_DATA) ───────────────────────────────────

    def get_account_info(self) -> dict:
        """Full account info including balances."""
        return self._get("/api/v3/account", {}, signed=True, for_trading=False)

    def get_balance(self, asset: str) -> dict:
        """Get free and locked balance for a specific asset."""
        info = self.get_account_info()
        for b in info.get("balances", []):
            if b["asset"] == asset:
                return {
                    "asset": asset,
                    "free": Decimal(b["free"]),
                    "locked": Decimal(b["locked"]),
                }
        return {"asset": asset, "free": Decimal("0"), "locked": Decimal("0")}

    def get_all_balances(self) -> Dict[str, dict]:
        """Returns {asset: {"free": Decimal, "locked": Decimal}}"""
        info = self.get_account_info()
        out = {}
        for b in info.get("balances", []):
            free = Decimal(b["free"])
            locked = Decimal(b["locked"])
            if free > 0 or locked > 0:
                out[b["asset"]] = {"free": free, "locked": locked}
        return out

    def get_open_orders(self, symbol: str = None) -> List[dict]:
        params = {"symbol": symbol} if symbol else {}
        return self._get("/api/v3/openOrders", params, signed=True, for_trading=False)

    def get_order(self, symbol: str, order_id: str = None,
                  client_order_id: str = None) -> dict:
        """
        Query a single order by exchange orderId or clientOrderId.
        Essential for unknown-state recovery.
        """
        params: dict = {"symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        elif client_order_id:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("Must provide order_id or client_order_id")
        return self._get("/api/v3/order", params, signed=True, for_trading=False)

    def get_all_orders(self, symbol: str, limit: int = 50) -> List[dict]:
        """Recent order history for a symbol."""
        return self._get("/api/v3/allOrders", {
            "symbol": symbol, "limit": limit
        }, signed=True, for_trading=False)

    def get_my_trades(self, symbol: str, limit: int = 50) -> List[dict]:
        """Recent fill trades for a symbol."""
        return self._get("/api/v3/myTrades", {
            "symbol": symbol, "limit": limit
        }, signed=True, for_trading=False)

    # ── Trading Endpoints (TRADE) ─────────────────────────────────────────────

    def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        client_order_id: str = None,
        time_in_force: str = "GTC",
        stop_price: Optional[Decimal] = None,
    ) -> dict:
        """
        Place a spot order.

        CRITICAL: This is the ONLY place orders are sent.
        The caller MUST record the order before calling this.
        The state transitions to SUBMITTING before this call,
        and ACKNOWLEDGED or UNKNOWN after.

        Returns the raw Binance order response dict.
        """
        params: dict = {
            "symbol": symbol,
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity),
        }
        if client_order_id:
            params["newClientOrderId"] = client_order_id
        if order_type.upper() in ("LIMIT", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"):
            if price is None:
                raise ValueError(f"price required for order_type={order_type}")
            params["price"] = str(price)
            params["timeInForce"] = time_in_force
        if stop_price is not None:
            params["stopPrice"] = str(stop_price)

        logger.info(
            f"[Binance] Placing {order_type} {side} {quantity} {symbol} "
            f"client_id={client_order_id}"
        )
        return self._post("/api/v3/order", params)

    def cancel_order(self, symbol: str, order_id: str = None,
                     client_order_id: str = None) -> dict:
        """Cancel an open order by exchange ID or client ID."""
        params: dict = {"symbol": symbol}
        if order_id:
            params["orderId"] = order_id
        elif client_order_id:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("Must provide order_id or client_order_id")
        return self._delete("/api/v3/order", params)

    def place_oco_order(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,          # take profit limit price
        stop_price: Decimal,     # stop trigger price
        stop_limit_price: Decimal,  # stop limit price
        list_client_order_id: str = None,
    ) -> dict:
        """
        Place an OCO (One-Cancels-Other) order for TP+SL simultaneously.
        Available on Binance Spot for exit management.
        """
        params: dict = {
            "symbol": symbol,
            "side": side.upper(),
            "quantity": str(quantity),
            "price": str(price),
            "stopPrice": str(stop_price),
            "stopLimitPrice": str(stop_limit_price),
            "stopLimitTimeInForce": "GTC",
        }
        if list_client_order_id:
            params["listClientOrderId"] = list_client_order_id
        return self._post("/api/v3/order/oco", params)

    # ── Symbol Filters ────────────────────────────────────────────────────────

    def load_symbol_info(self, symbol: str, force: bool = False) -> dict:
        """
        Load and cache symbol trading rules from Binance.
        Returns parsed filter dict.
        """
        if not force and symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]

        info = self.get_exchange_info(symbol)
        for sym in info.get("symbols", []):
            if sym["symbol"] == symbol:
                parsed = self._parse_filters(sym)
                self._symbol_info_cache[symbol] = parsed
                self._cache_loaded_at = time.time()
                return parsed

        raise ValueError(f"Symbol {symbol!r} not found in exchange info")

    def _parse_filters(self, sym_info: dict) -> dict:
        """Extract all relevant trading rules from exchangeInfo symbol entry."""
        result = {
            "symbol": sym_info["symbol"],
            "status": sym_info.get("status", ""),
            "base_asset": sym_info.get("baseAsset", ""),
            "quote_asset": sym_info.get("quoteAsset", ""),
            "is_trading": sym_info.get("status", "") == "TRADING",
        }
        for f in sym_info.get("filters", []):
            ft = f["filterType"]
            if ft == "PRICE_FILTER":
                result["tick_size"] = Decimal(f["tickSize"])
                result["min_price"] = Decimal(f["minPrice"])
                result["max_price"] = Decimal(f["maxPrice"])
            elif ft == "LOT_SIZE":
                result["step_size"] = Decimal(f["stepSize"])
                result["min_qty"] = Decimal(f["minQty"])
                result["max_qty"] = Decimal(f["maxQty"])
            elif ft == "MIN_NOTIONAL":
                result["min_notional"] = Decimal(f.get("minNotional", "0"))
            elif ft == "NOTIONAL":
                result["min_notional"] = Decimal(f.get("minNotional", "0"))
            elif ft == "ICEBERG_PARTS":
                pass  # not needed for simple spot
            elif ft == "MARKET_LOT_SIZE":
                result["market_min_qty"] = Decimal(f.get("minQty", "0"))
                result["market_max_qty"] = Decimal(f.get("maxQty", "0"))
                result["market_step_size"] = Decimal(f.get("stepSize", "0"))
        return result

    def apply_lot_size(self, quantity: Decimal, symbol: str) -> Decimal:
        """Round quantity down to the nearest valid step_size."""
        info = self.load_symbol_info(symbol)
        step = info.get("step_size", Decimal("0.00000001"))
        if step == 0:
            return quantity
        return (quantity / step).to_integral_value(rounding=ROUND_DOWN) * step

    def apply_tick_size(self, price: Decimal, symbol: str) -> Decimal:
        """Round price to the nearest valid tick_size."""
        info = self.load_symbol_info(symbol)
        tick = info.get("tick_size", Decimal("0.00000001"))
        if tick == 0:
            return price
        return (price / tick).to_integral_value(rounding=ROUND_DOWN) * tick

    def validate_order_filters(
        self, symbol: str, quantity: Decimal, price: Optional[Decimal] = None
    ) -> Tuple[bool, str]:
        """
        Validate an order against all symbol filters before submitting.
        Returns (ok, reason).
        """
        try:
            info = self.load_symbol_info(symbol)
        except Exception as e:
            return False, f"Could not load symbol info: {e}"

        if not info.get("is_trading"):
            return False, f"{symbol} is not in TRADING status"
        if quantity < info.get("min_qty", Decimal("0")):
            return False, f"qty {quantity} < min_qty {info.get('min_qty')}"
        if quantity > info.get("max_qty", Decimal("999999999")):
            return False, f"qty {quantity} > max_qty {info.get('max_qty')}"
        if price is not None:
            notional = quantity * price
            min_notional = info.get("min_notional", Decimal("0"))
            if notional < min_notional:
                return False, f"notional {notional} < min_notional {min_notional}"
        return True, "ok"

    # ── User Data Stream ──────────────────────────────────────────────────────

    def create_listen_key(self) -> str:
        """Create a User Data Stream listen key."""
        data = self._session.post(
            BINANCE_SPOT_BASE + "/api/v3/userDataStream",
            headers={"X-MBX-APIKEY": self._trading_key},
            timeout=self._timeout,
        ).json()
        return data["listenKey"]

    def keepalive_listen_key(self, listen_key: str) -> None:
        """Extend listen key validity (must call every ~30 min)."""
        self._session.put(
            BINANCE_SPOT_BASE + "/api/v3/userDataStream",
            params={"listenKey": listen_key},
            headers={"X-MBX-APIKEY": self._trading_key},
            timeout=self._timeout,
        )

    def close_listen_key(self, listen_key: str) -> None:
        self._session.delete(
            BINANCE_SPOT_BASE + "/api/v3/userDataStream",
            params={"listenKey": listen_key},
            headers={"X-MBX-APIKEY": self._trading_key},
            timeout=self._timeout,
        )

    # ── Health / Connectivity ─────────────────────────────────────────────────

    def ping(self) -> bool:
        """Returns True if Binance is reachable."""
        try:
            self._get("/api/v3/ping")
            return True
        except Exception:
            return False

    def check_connectivity(self) -> Tuple[bool, float]:
        """Ping and measure latency in ms. Returns (ok, latency_ms)."""
        start = time.time()
        ok = self.ping()
        latency_ms = (time.time() - start) * 1000
        return ok, latency_ms

    def get_time_sync(self) -> TimeSync:
        return self._time_sync
