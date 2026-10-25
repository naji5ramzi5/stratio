"""
╔══════════════════════════════════════════════════════════════════╗
║     STRATOCRYPTO v3.0 — Institutional Crypto Analysis Suite    ║
║     AI Signals · Liquidity Intel · Price Prediction             ║
║     Telegram Tri-Bot System · Iraq Time (UTC+3)                 ║
╚══════════════════════════════════════════════════════════════════╝

Bots:
  Bot 1 — Signals: AI-scored trade setups (score >= 65)
  Bot 2 — Liquidity: Order-book anomalies, spoofing, walls
  Bot 3 — Predictions: Price forecasts 2h/6h/8h/24h (rise > 3%)
"""

import os, sys, json, time, logging, threading
import requests
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from prediction_tracker import prediction_tracker

TELEGRAM_PROXY = os.getenv("TELEGRAM_PROXY", "")
TELEGRAM_TIMEOUT = 5

try:
    import pytz
    BAGHDAD_TZ = pytz.timezone("Asia/Baghdad")
except ImportError:
    BAGHDAD_TZ = None

def fmt_symbol(sym):
    if sym.endswith("USDT"):
        return sym[:-4] + "/USDT"
    if sym.endswith("USD"):
        return sym[:-3] + "/USD"
    if sym.endswith("BTC"):
        return sym[:-3] + "/BTC"
    return sym

def now_baghdad():
    if BAGHDAD_TZ:
        return datetime.now(BAGHDAD_TZ)
    return datetime.utcnow() + timedelta(hours=3)

def fmt_baghdad(dt=None):
    dt = dt or now_baghdad()
    return dt.strftime("%Y-%m-%d %H:%M AST")

# Load .env from the script's own folder FIRST, then CWD, then project root.
# (On a server the CWD is often /root or the repo root, so relying on CWD alone
# silently left every token empty and no bot could send.)
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, ".env"))
load_dotenv()
load_dotenv(os.path.join(_HERE, "..", ".env"))

BINANCE_BASE     = os.getenv("BINANCE_BASE", "https://data-api.binance.vision")
BINANCE_FAPI     = "https://fapi.binance.com"
BINANCE_API_KEY  = os.getenv("BINANCE_API_KEY", "")
_admin_ids = os.getenv("ADMIN_CHAT_ID", "")
# Support comma-separated IDs, including negative channel IDs (e.g. "-1004373560971")
AUTHORIZED_USERS = []
for _id in _admin_ids.split(","):
    _id = _id.strip()
    try:
        if _id:
            AUTHORIZED_USERS.append(int(_id))
    except ValueError:
        pass
if not AUTHORIZED_USERS:
    AUTHORIZED_USERS = [-1004373560971]

MASTER_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
BOT1_TOKEN = os.getenv("TELEGRAM_PREDICTION_TOKEN", MASTER_TOKEN)
BOT2_TOKEN = os.getenv("TELEGRAM_LIQUIDITY_TOKEN", MASTER_TOKEN)
BOT3_TOKEN = os.getenv("TELEGRAM_PREDICTOR_TOKEN", MASTER_TOKEN)
BOT4_TOKEN = os.getenv("BOT4_TOKEN", MASTER_TOKEN)

MIN_24H_VOLUME_USDT       = float(os.getenv("MIN_24H_VOLUME_USDT", "5000000"))
LIQUIDITY_IMBALANCE_THRESH = float(os.getenv("LIQUIDITY_IMBALANCE_THRESHOLD", "15"))
SPREAD_THRESH              = float(os.getenv("SPREAD_CHANGE_THRESHOLD", "0.5"))
PREDICTION_THRESH          = float(os.getenv("PREDICTION_INCREASE_THRESHOLD", "3"))
ORDER_BOOK_LEVELS          = 100
SCAN_INTERVAL_MINUTES      = 20
AI_SCORE_THRESHOLD         = 50

# Default universe for the weekly pairs rescan (high-volume spot USDT pairs).
_DEFAULT_UNIVERSE = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "LTCUSDT", "LINKUSDT", "DOTUSDT", "AVAXUSDT", "TRXUSDT",
    "FILUSDT", "MATICUSDT", "ATOMUSDT", "ETCUSDT", "NEARUSDT", "APTUSDT",
    "ARBUSDT", "OPUSDT", "SUIUSDT", "TONUSDT", "PEPEUSDT", "SHIBUSDT",
    "BCHUSDT", "UNIUSDT",
]

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(BASE_DIR, "liquidity_state.json")
LOG_FILE   = os.path.join(BASE_DIR, "smart_money_log.json")
PRED_LOG   = os.path.join(BASE_DIR, "predictions.json")
PRED_HIST  = os.path.join(BASE_DIR, "prediction_history.json")

from logging.handlers import RotatingFileHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        RotatingFileHandler(os.path.join(BASE_DIR, "advanced_bot.log"), maxBytes=5*1024*1024, backupCount=3, encoding="utf-8")
    ]
)
logger = logging.getLogger("StratoCrypto")

try:
    with open(STATE_FILE, "r", encoding="utf-8") as f:
        liquidity_state = json.load(f)
except Exception:
    liquidity_state = {}

try:
    with open(PRED_HIST, "r", encoding="utf-8") as f:
        prediction_history = json.load(f)
except Exception:
    prediction_history = []

monitoring_active = True
state_lock = threading.Lock()
pred_lock = threading.Lock()
last_daily_report_date = None
last_training_date = None

# ─── TELEGRAM ────────────────────────────────────────────────────
_session = None
def _get_session():
    global _session
    if _session is not None:
        return _session
    _session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=20, pool_maxsize=30, max_retries=2)
    _session.mount('http://', adapter)
    _session.mount('https://', adapter)
    if TELEGRAM_PROXY:
        _session.proxies = {"https": TELEGRAM_PROXY, "http": TELEGRAM_PROXY}
    return _session

def send_telegram(token: str, message: str, parse_mode: str = "Markdown"):
    if not token:
        return
    from telegram_mute import suppress
    if suppress(logger, "advanced_bot", message):
        return True
    if len(message) > 4096:
        message = message[:4050] + "\n\n... (truncated)"
    sess = _get_session()
    for uid in AUTHORIZED_USERS:
        # STRICT RULE: ONLY BOT4_TOKEN (@bossbtc2_bot) is allowed to send to the channel (uid < 0)!
        if uid < 0 and token != BOT4_TOKEN:
            continue
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        try:
            resp = sess.post(url, json={
                "chat_id": uid, "text": message, "parse_mode": parse_mode
            }, timeout=TELEGRAM_TIMEOUT)
            if resp.status_code == 400 and "chat not found" in resp.text:
                pass  # user didn't start bot, skip silently
            elif resp.status_code == 400 and "parse" in resp.text.lower():
                # A malformed Markdown entity must never silence a signal.
                # Resend the exact same text without parse_mode.
                r2 = sess.post(url, json={"chat_id": uid, "text": message},
                               timeout=TELEGRAM_TIMEOUT)
                if r2.status_code != 200:
                    logger.warning(f"Telegram {uid} plain-text retry failed: {r2.text[:80]}")
                else:
                    logger.info(f"Telegram {uid}: Markdown rejected, sent as plain text")
            elif resp.status_code != 200:
                logger.warning(f"Telegram {uid}: {resp.text[:80]}")
        except Exception as e:
            logger.warning(f"Telegram {uid} send failed: {e}")
        time.sleep(0.1)

# ─── MARKET DATA ────────────────────────────────────────────────
_binance_session = None
def _get_binance_session():
    global _binance_session
    if _binance_session is not None:
        return _binance_session
    _binance_session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=30, pool_maxsize=50, max_retries=2)
    _binance_session.mount('http://', adapter)
    _binance_session.mount('https://', adapter)
    if TELEGRAM_PROXY:
        _binance_session.proxies = {"https": TELEGRAM_PROXY, "http": TELEGRAM_PROXY}
    _binance_session.headers.update({"X-MBX-APIKEY": BINANCE_API_KEY} if BINANCE_API_KEY else {})
    return _binance_session

def get_binance(path: str, params: dict = None, futures: bool = False) -> dict:
    base = BINANCE_FAPI if futures else BINANCE_BASE
    sess = _get_binance_session()
    try:
        r = sess.get(base + path, params=params, timeout=8)
        if r.status_code == 451 or (r.status_code != 200 and not futures):
            r = sess.get("https://data-api.binance.vision" + path, params=params, timeout=8)
        return r.json()
    except Exception:
        return {}

def get_ticker_24h(symbol: str) -> dict:
    return get_binance("/api/v3/ticker/24hr", {"symbol": symbol})

def get_all_prices() -> dict:
    """One-call snapshot of every spot price: {SYMBOL: lastPrice}.

    Uses the shared market_data cache so concurrent bots do not each hit
    Binance separately."""
    try:
        from market_data import get_all_prices as _md_prices
        prices = _md_prices()
        if prices:
            return prices
    except Exception:
        pass
    # fallback to direct Binance call
    data = get_binance("/api/v3/ticker/price")
    if not isinstance(data, list):
        return {}
    out = {}
    for t in data:
        try:
            p = float(t.get("price", 0))
        except (TypeError, ValueError):
            continue
        if p > 0 and t.get("symbol"):
            out[t["symbol"]] = p
    return out


def get_klines(symbol: str, interval: str = "1h", limit: int = 200) -> list:
    """Klines via the shared cache (falls back to a direct call)."""
    try:
        from market_data import get_klines as _md_klines
        k = _md_klines(symbol, interval, limit)
        if k:
            return k
    except Exception:
        pass
    data = get_binance("/api/v3/klines", {"symbol": symbol, "interval": interval, "limit": limit})
    return data if isinstance(data, list) else []

def get_order_book(symbol: str, levels: int = ORDER_BOOK_LEVELS) -> dict:
    return get_binance("/api/v3/depth", {"symbol": symbol, "limit": levels})

def get_recent_trades(symbol: str, limit: int = 500) -> list:
    data = get_binance("/api/v3/trades", {"symbol": symbol, "limit": limit})
    return data if isinstance(data, list) else []

def get_open_interest(symbol: str) -> dict:
    return get_binance("/fapi/v1/openInterest", {"symbol": symbol}, futures=True)

def get_funding_rate(symbol: str) -> dict:
    data = get_binance("/fapi/v1/fundingRate", {"symbol": symbol, "limit": 1}, futures=True)
    return data[0] if isinstance(data, list) and data else {}

def get_long_short_ratio(symbol: str) -> dict:
    return get_binance("/futures/data/globalLongShortAccountRatio",
                       {"symbol": symbol, "period": "1h", "limit": 1}, futures=True)

# Pairs that cannot produce a meaningful directional forecast: stablecoins,
# tokenised gold and other pegs.  "USDC +3% in 24h" is noise that wastes a
# prediction slot and can never become a trade.
_NON_FORECASTABLE_BASES = {
    "USDC", "USDT", "BUSD", "TUSD", "FDUSD", "USDP", "DAI", "USD1",
    "BFUSD", "EUR", "EURI", "AEUR", "PAXG", "XAUT",
}


def _is_forecastable(sym: str) -> bool:
    """False for stablecoin/gold pairs that should never be scanned."""
    base = sym[:-4] if sym.endswith("USDT") else sym
    return base.upper() not in _NON_FORECASTABLE_BASES


def get_qualified_symbols(min_volume: float = MIN_24H_VOLUME_USDT) -> list:
    data = get_binance("/api/v3/ticker/24hr")
    if not isinstance(data, list):
        logger.warning(f"[FILTER] Binance ticker unavailable (got {type(data).__name__}); "
                       f"cannot build symbol list")
        return []
    qualified = []
    for t in data:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT") or not _is_forecastable(sym):
            continue
        try:
            if float(t.get("quoteVolume", 0)) >= min_volume:
                qualified.append(sym)
        except Exception:
            continue
    logger.info(f"[FILTER] {len(qualified)} symbols >= ${min_volume/1e6:.0f}M vol")
    
    # Display USDT pairs for user
    usdt_pairs = [sym for sym in qualified if sym.endswith("USDT")]
    if not usdt_pairs:
        logger.info("[INFO] No USDT pairs found in qualifying symbols")
        return qualified
    
    logger.info("[INFO] USDT Pairs (qualified):")
    for sym in usdt_pairs[:20]:
        logger.info(f"  • {sym}")
    if len(usdt_pairs) > 20:
        logger.info(f"  ... and {len(usdt_pairs) - 20} more USDT pairs")
    
    return sorted(qualified)


def get_symbols_by_volume(limit: int = 30,
                          min_volume: float = MIN_24H_VOLUME_USDT) -> list:
    """Top-N USDT pairs by 24h quote volume.

    The old code did ``get_qualified_symbols()[:15]``, which is alphabetical:
    it always predicted AAVE..BTC and never ETH, SOL, XRP, DOGE, etc.  Ranking
    by volume keeps the majors in every scan.
    """
    data = get_binance("/api/v3/ticker/24hr")
    if not isinstance(data, list):
        logger.warning("[FILTER] Binance ticker unavailable; cannot rank symbols by volume")
        return []
    rows = []
    for t in data:
        sym = t.get("symbol", "")
        if not sym.endswith("USDT") or not _is_forecastable(sym):
            continue
        try:
            vol = float(t.get("quoteVolume", 0) or 0)
        except (TypeError, ValueError):
            continue
        if vol >= min_volume:
            rows.append((vol, sym))
    rows.sort(reverse=True)
    out = [sym for _, sym in rows[:limit]]
    logger.info(f"[FILTER] top {len(out)} by 24h volume: {', '.join(out[:8])}"
                + (" ..." if len(out) > 8 else ""))
    return out

# ─── TECHNICAL ANALYSIS ─────────────────────────────────────────
def compute_ema(prices: np.ndarray, period: int) -> np.ndarray:
    ema = np.zeros_like(prices)
    ema[period - 1] = np.mean(prices[:period])
    k = 2 / (period + 1)
    for i in range(period, len(prices)):
        ema[i] = prices[i] * k + ema[i - 1] * (1 - k)
    return ema

def compute_rsi(prices: np.ndarray, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

def compute_macd(prices: np.ndarray) -> dict:
    if len(prices) < 26:
        return {"macd": 0, "signal": 0, "hist": 0, "trend": "Neutral"}
    ema12 = compute_ema(prices, 12)
    ema26 = compute_ema(prices, 26)
    macd_line = ema12 - ema26
    signal_line = compute_ema(macd_line[25:], 9)
    hist = macd_line[-1] - signal_line[-1]
    trend = "Bullish 🟢" if hist > 0 else "Bearish 🔴"
    return {"macd": round(macd_line[-1], 6), "signal": round(signal_line[-1], 6),
            "hist": round(hist, 6), "trend": trend}

def compute_vwap(klines: list) -> float:
    if not klines:
        return 0.0
    tp_vol = sum(((float(k[2]) + float(k[3]) + float(k[4])) / 3) * float(k[5]) for k in klines)
    total_vol = sum(float(k[5]) for k in klines)
    return round(tp_vol / total_vol, 8) if total_vol > 0 else 0.0

def compute_atr(klines: list, period: int = 14) -> float:
    if len(klines) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(klines)):
        h, l, pc = float(klines[i][2]), float(klines[i][3]), float(klines[i-1][4])
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    return round(np.mean(trs[-period:]), 8)

def compute_bollinger(prices: np.ndarray, period: int = 20) -> dict:
    if len(prices) < period:
        return {"upper": 0, "middle": 0, "lower": 0, "position": "middle"}
    sma = np.mean(prices[-period:])
    std = np.std(prices[-period:])
    upper, lower = sma + 2 * std, sma - 2 * std
    current = prices[-1]
    if current > upper:
        pos = "Above Upper (Overbought)"
    elif current < lower:
        pos = "Below Lower (Oversold)"
    else:
        pct = (current - lower) / (upper - lower) * 100
        pos = f"{pct:.0f}% of Band"
    return {"upper": round(upper, 8), "middle": round(sma, 8),
            "lower": round(lower, 8), "position": pos}

def classify_trend(closes: np.ndarray) -> str:
    if len(closes) < 200:
        return "Insufficient Data"
    ema20 = compute_ema(closes, 20)[-1]
    ema50 = compute_ema(closes, 50)[-1]
    ema200 = compute_ema(closes, 200)[-1]
    price = closes[-1]
    score = sum([price > ema20, price > ema50, price > ema200,
                 ema20 > ema50, ema50 > ema200])
    return {5: "Strong Bullish 🚀", 4: "Bullish 📈", 3: "Neutral ➡️",
            2: "Bearish 📉", 1: "Strong Bearish 🩸", 0: "Extreme Bear 💀"}.get(score, "Neutral ➡️")

def get_technical_analysis(symbol: str, interval: str = "1h") -> dict:
    klines = get_klines(symbol, interval, 250)
    if len(klines) < 30:
        return {}
    closes = np.array([float(k[4]) for k in klines])
    current_price = closes[-1]
    rsi = compute_rsi(closes)
    macd = compute_macd(closes)
    vwap = compute_vwap(klines[-24:])
    atr = compute_atr(klines)
    bb = compute_bollinger(closes)
    trend = classify_trend(closes)
    ema20 = round(compute_ema(closes, 20)[-1], 8)
    ema50 = round(compute_ema(closes, 50)[-1], 8) if len(closes) >= 50 else 0
    ema200 = round(compute_ema(closes, 200)[-1], 8) if len(closes) >= 200 else 0
    vwap_pos = "Above VWAP 🟢" if current_price > vwap else "Below VWAP 🔴"
    if rsi >= 70:   rsi_label = f"{rsi} (Overbought 🔴)"
    elif rsi <= 30: rsi_label = f"{rsi} (Oversold 🟢)"
    else:           rsi_label = f"{rsi} (Neutral ⚪)"
    return {
        "interval": interval, "current_price": round(current_price, 8),
        "trend": trend, "rsi": rsi, "rsi_label": rsi_label,
        "macd": macd, "vwap": vwap, "vwap_position": vwap_pos,
        "atr": atr, "bollinger": bb,
        "ema20": ema20, "ema50": ema50, "ema200": ema200,
    }

# ─── ORDER FLOW ─────────────────────────────────────────────────
def analyze_order_flow(symbol: str) -> dict:
    trades = get_recent_trades(symbol, limit=500)
    if not trades:
        return {}
    buy_vol  = sum(float(t["qty"]) for t in trades if not t.get("isBuyerMaker", True))
    sell_vol = sum(float(t["qty"]) for t in trades if t.get("isBuyerMaker", True))
    total = buy_vol + sell_vol
    if total == 0:
        return {}
    cvd = buy_vol - sell_vol
    buy_pct = round(buy_vol / total * 100, 1)
    sell_pct = round(sell_vol / total * 100, 1)
    avg_trade = total / len(trades)
    whale_trades = [t for t in trades if float(t["qty"]) > avg_trade * 5]
    whale_count = len(whale_trades)
    whale_buy_vol = sum(float(t["qty"]) for t in whale_trades if not t.get("isBuyerMaker", True))
    whale_sell_vol = sum(float(t["qty"]) for t in whale_trades if t.get("isBuyerMaker", True))
    cvd_label = "Positive 🟢" if cvd > 0 else "Negative 🔴"
    return {
        "buy_vol": round(buy_vol, 2), "sell_vol": round(sell_vol, 2),
        "buy_pct": buy_pct, "sell_pct": sell_pct,
        "cvd": round(cvd, 2), "cvd_label": cvd_label,
        "whale_count": whale_count,
        "whale_buy_vol": round(whale_buy_vol, 2),
        "whale_sell_vol": round(whale_sell_vol, 2),
        "whale_detected": whale_count >= 3,
        "dominant": "Buyers 🟢" if buy_pct > 55 else "Sellers 🔴" if sell_pct > 55 else "Balanced ⚪"
    }

# ─── DERIVATIVES ────────────────────────────────────────────────
def analyze_derivatives(symbol: str) -> dict:
    result = {}
    oi = get_open_interest(symbol)
    if oi and "openInterest" in oi:
        result["open_interest"] = float(oi["openInterest"])
    fr = get_funding_rate(symbol)
    if fr and "fundingRate" in fr:
        rate = float(fr["fundingRate"]) * 100
        result["funding_rate"] = round(rate, 4)
        if rate > 0.1:
            result["funding_label"] = f"{rate:.4f}% (Longs Pay 🔴)"
        elif rate < -0.05:
            result["funding_label"] = f"{rate:.4f}% (Shorts Pay 🟢)"
        else:
            result["funding_label"] = f"{rate:.4f}% (Neutral ⚪)"
    ls = get_long_short_ratio(symbol)
    if isinstance(ls, list) and ls:
        ls = ls[0]
    if ls and "longShortRatio" in ls:
        ratio = float(ls["longShortRatio"])
        result["ls_ratio"] = round(ratio, 3)
        result["ls_label"] = "Longs 📈" if ratio > 1.2 else "Shorts 📉" if ratio < 0.8 else "Balanced ⚪"
    return result

# ─── LIQUIDITY (100-LEVEL ORDER BOOK) ──────────────────────────
def analyze_liquidity(symbol: str) -> dict:
    ob = get_order_book(symbol, ORDER_BOOK_LEVELS)
    bids = ob.get("bids", [])
    asks = ob.get("asks", [])
    if not bids or not asks:
        return {"error": "No order book data"}
    best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
    spread_pct = (best_ask - best_bid) / best_ask * 100
    bid_vols = [float(b[1]) for b in bids]
    ask_vols = [float(a[1]) for a in asks]
    total_bid, total_ask = sum(bid_vols), sum(ask_vols)
    total = total_bid + total_ask
    imbalance = (total_bid - total_ask) / total * 100 if total > 0 else 0
    # Spoofing (100-level)
    avg_bid_sz = np.mean(bid_vols[:20]) if bid_vols else 1
    avg_ask_sz = np.mean(ask_vols[:20]) if ask_vols else 1
    spoof_bids = [b for b in bids[:30] if float(b[1]) > avg_bid_sz * 5]
    spoof_asks = [a for a in asks[:30] if float(a[1]) > avg_ask_sz * 5]
    spoofing_detected = len(spoof_bids) >= 2 or len(spoof_asks) >= 2
    # Walls
    bid_walls = [(float(b[0]), float(b[1])) for b in bids[:50] if float(b[1]) > total_bid * 0.15]
    ask_walls = [(float(a[0]), float(a[1])) for a in asks[:50] if float(a[1]) > total_ask * 0.15]
    # Iceberg
    bid_prices_rounded = [round(float(b[0]), 2) for b in bids[:30]]
    ask_prices_rounded = [round(float(a[0]), 2) for a in asks[:30]]
    iceberg_suspected = len(set(bid_prices_rounded)) < len(bid_prices_rounded) * 0.7
    # Concentration
    top5_bid_pct = sum(bid_vols[:5]) / total_bid * 100 if total_bid > 0 else 0
    # Signal
    if imbalance > LIQUIDITY_IMBALANCE_THRESH:
        signal = "Strong Buy 🟢"
    elif imbalance > 5:
        signal = "Buy 🟩"
    elif imbalance < -LIQUIDITY_IMBALANCE_THRESH:
        signal = "Strong Sell 🔴"
    elif imbalance < -5:
        signal = "Sell 🟥"
    else:
        signal = "Neutral ⚪"
    # Smart Money Score
    sm_score = 0
    if abs(imbalance) > 20: sm_score += 30
    elif abs(imbalance) > 10: sm_score += 15
    if not spoofing_detected: sm_score += 20
    if spread_pct < 0.1: sm_score += 20
    if top5_bid_pct < 60: sm_score += 15
    if not iceberg_suspected: sm_score += 15
    return {
        "best_bid": best_bid, "best_ask": best_ask,
        "spread_pct": round(spread_pct, 4),
        "imbalance_pct": round(imbalance, 2), "signal": signal,
        "bid_depth": round(total_bid, 2), "ask_depth": round(total_ask, 2),
        "spoofing_detected": spoofing_detected,
        "bid_walls": bid_walls[:3], "ask_walls": ask_walls[:3],
        "iceberg_suspected": iceberg_suspected,
        "smart_money_score": sm_score,
    }

# ─── MULTI-TIMEFRAME ────────────────────────────────────────────
def multi_timeframe_analysis(symbol: str) -> dict:
    timeframes = ["5m", "15m", "1h", "4h", "1d"]
    results = {}
    bullish, bearish, neutral = 0, 0, 0
    for tf in timeframes:
        ta = get_technical_analysis(symbol, tf)
        if not ta:
            continue
        results[tf] = {
            "trend": ta.get("trend", "N/A"),
            "rsi": ta.get("rsi", 50),
            "macd_trend": ta.get("macd", {}).get("trend", "Neutral"),
        }
        t = ta.get("trend", "")
        if "Bullish" in t or "Bull" in t:
            bullish += 1
        elif "Bearish" in t or "Bear" in t:
            bearish += 1
        else:
            neutral += 1
    if bullish >= 4:
        bias = "Strong Bullish Confluence 🚀"
    elif bullish >= 3:
        bias = "Bullish Bias 📈"
    elif bearish >= 4:
        bias = "Strong Bearish Confluence 🩸"
    elif bearish >= 3:
        bias = "Bearish Bias 📉"
    elif bullish > bearish:
        bias = "Slightly Bullish ↗️"
    elif bearish > bullish:
        bias = "Slightly Bearish ↘️"
    else:
        bias = "Mixed / Neutral ➡️"
    return {
        "timeframes": results,
        "bullish_count": bullish, "bearish_count": bearish, "neutral_count": neutral,
        "unified_bias": bias,
    }

# ─── AI SCORING ─────────────────────────────────────────────────
def compute_risk_level(ta: dict, liq: dict, flow: dict, deriv: dict) -> dict:
    risk = 0
    factors = []
    rsi = ta.get("rsi", 50)
    if rsi and rsi > 80: risk += 30; factors.append("RSI overbought")
    elif rsi and rsi < 20: risk += 25; factors.append("RSI oversold")
    spread = liq.get("spread_pct", 0)
    if spread and spread > 0.5: risk += 15; factors.append("Wide spread")
    if liq.get("spoofing_detected"): risk += 25; factors.append("Spoofing detected")
    fr = deriv.get("funding_rate", 0) if deriv else 0
    if fr and abs(fr) > 0.2: risk += 15; factors.append("Extreme funding")
    if flow and flow.get("whale_detected") and flow.get("whale_buy_vol", 0) < flow.get("whale_sell_vol", 0):
        risk += 10; factors.append("Whale distribution")
    level = "Low 🟢"
    if risk >= 50: level = "High 🔴"
    elif risk >= 25: level = "Medium 🟡"
    return {"risk_score": risk, "risk_level": level, "risk_factors": factors}

def compute_ai_score(liq: dict, ta: dict, flow: dict, deriv: dict, mtf: dict) -> dict:
    score = 0; reasons = []; breakdown = {}
    # Liquidity (25)
    liq_s = 0
    imb = abs(liq.get("imbalance_pct", 0))
    if imb > 20: liq_s += 12
    elif imb > 10: liq_s += 7
    elif imb > 5: liq_s += 3
    sm = liq.get("smart_money_score", 0)
    liq_s += min(sm // 10, 10)
    if not liq.get("spoofing_detected", False): liq_s += 3
    liq_s = min(liq_s, 25)
    breakdown["liquidity"] = liq_s
    if liq_s >= 18: reasons.append("✅ Strong liquidity imbalance")
    elif liq_s >= 12: reasons.append("⚠️ Moderate liquidity signal")
    # Technical (25)
    ta_s = 0
    rsi = ta.get("rsi", 50)
    if rsi < 40: ta_s += 8
    elif rsi > 60: ta_s += 3
    else: ta_s += 5
    macd_t = ta.get("macd", {}).get("trend", "")
    if "Bullish" in macd_t: ta_s += 7
    elif "Bearish" in macd_t: ta_s += 2
    trend_t = ta.get("trend", "")
    if "Strong Bullish" in trend_t: ta_s += 10
    elif "Bullish" in trend_t: ta_s += 7
    elif "Neutral" in trend_t: ta_s += 3
    ta_s = min(ta_s, 25)
    breakdown["technical"] = ta_s
    if ta_s >= 18: reasons.append("✅ Strong technical alignment")
    elif ta_s >= 12: reasons.append("⚠️ Mixed technical indicators")
    # Order Flow (25)
    fl_s = 0
    buy_pct = flow.get("buy_pct", 50) if flow else 50
    if buy_pct >= 65: fl_s += 12
    elif buy_pct >= 55: fl_s += 7
    elif buy_pct >= 45: fl_s += 3
    if flow and flow.get("whale_detected"):
        fl_s += 8 if flow.get("whale_buy_vol", 0) > flow.get("whale_sell_vol", 0) else -3
        if fl_s > 0: reasons.append("🐋 Whale accumulation")
    cvd = flow.get("cvd", 0) if flow else 0
    if cvd > 0: fl_s += 5
    fl_s = max(0, min(fl_s, 25))
    breakdown["order_flow"] = fl_s
    if fl_s >= 18: reasons.append("✅ Strong buying pressure")
    # Derivatives (25)
    dv_s = 0
    fr = deriv.get("funding_rate", 0) if deriv else 0
    if -0.05 <= fr <= 0.05: dv_s += 8
    elif fr < -0.05: dv_s += 12
    ls = deriv.get("ls_ratio", 1) if deriv else 1
    if ls < 0.9: dv_s += 10
    elif 0.9 <= ls <= 1.1: dv_s += 5
    dv_s = min(dv_s, 25)
    breakdown["derivatives"] = dv_s
    if dv_s >= 15: reasons.append("✅ Favorable derivatives positioning")
    # MTF bonus
    mtf_bonus = 0
    if mtf.get("bullish_count", 0) >= 4: mtf_bonus = 5
    elif mtf.get("bullish_count", 0) >= 3: mtf_bonus = 3
    score = liq_s + ta_s + fl_s + dv_s + mtf_bonus
    if score >= 90: grade = "⭐⭐⭐ ELITE SETUP"
    elif score >= 80: grade = "⭐⭐ STRONG BUY"
    elif score >= 70: grade = "⭐ BUY"
    elif score >= 50: grade = "➡️ NEUTRAL"
    else: grade = "❌ AVOID"
    return {
        "total": min(score, 100), "grade": grade,
        "breakdown": breakdown, "reasons": reasons,
        "confidence": min(round(score * 0.95, 1), 99),
    }

# ─── TRADE SETUP ────────────────────────────────────────────────
def compute_trade_setup(ta: dict, liq: dict) -> dict:
    price = ta.get("current_price", 0)
    atr = ta.get("atr", price * 0.01)
    if not price or not atr:
        return {}
    stop_loss = round(price - atr * 1.5, 8)
    tp1 = round(price + atr * 1.5, 8)
    tp2 = round(price + atr * 3.0, 8)
    tp3 = round(price + atr * 5.0, 8)
    rr = round((tp2 - price) / (price - stop_loss), 2) if price > stop_loss else 0
    entry_low = round(price - atr * 0.3, 8)
    entry_high = round(price + atr * 0.3, 8)
    return {
        "entry_zone": f"{entry_low} – {entry_high}",
        "stop_loss": stop_loss, "tp1": tp1, "tp2": tp2, "tp3": tp3,
        "risk_reward": rr,
    }

# ─── PREDICTION ENGINE ──────────────────────────────────────────
def _rsi(prices, period=14):
    if len(prices) < period + 1: return 50.0
    d = np.diff(prices)
    g = np.where(d > 0, d, 0)
    ls = np.where(d < 0, -d, 0)
    ag = np.mean(g[-period:])
    al = np.mean(ls[-period:])
    return 100.0 if al == 0 else round(100 - (100 / (1 + ag / al)), 2)

def _macd_trend(prices):
    if len(prices) < 26: return "Neutral"
    k12 = 2 / 13; k26 = 2 / 27
    e12 = np.zeros_like(prices); e26 = np.zeros_like(prices)
    e12[11] = np.mean(prices[:12]); e26[25] = np.mean(prices[:26])
    for i in range(12, len(prices)): e12[i] = prices[i] * k12 + e12[i-1] * (1 - k12)
    for i in range(26, len(prices)): e26[i] = prices[i] * k26 + e26[i-1] * (1 - k26)
    macd = e12 - e26
    sig = np.zeros_like(macd)
    sig[25] = np.mean(macd[25:34])
    k9 = 2 / 10
    for i in range(26, len(macd)): sig[i] = macd[i] * k9 + sig[i-1] * (1 - k9)
    return "Bullish" if (macd[-1] - sig[-1]) > 0 else "Bearish"

def predict_movement(symbol, tf_hours):
    """Predict price movement using the v4.0 ML-Augmented engine from predictor.py."""
    try:
        from predictor import predict_price_movement
        result = predict_price_movement(symbol, tf_hours)
        if result and "error" not in result:
            return result
        if result and "error" in result:
            return {"symbol": symbol, "timeframe_hours": tf_hours, "error": result["error"]}
        return {"symbol": symbol, "timeframe_hours": tf_hours, "error": "empty result"}
    except Exception as e:
        return {"symbol": symbol, "timeframe_hours": tf_hours, "error": str(e)}

def save_prediction(pred):
    with pred_lock:
        pred["_id"] = f"{pred['symbol']}_{pred['timeframe_hours']}h_{now_baghdad().timestamp():.0f}"
        pred["_created_at"] = now_baghdad().isoformat()
        pred["_verified"] = False
        pred["_actual_change_pct"] = None
        pred["_correct"] = None
        prediction_history.append(pred)
        try:
            with open(PRED_HIST, "w", encoding="utf-8") as f:
                json.dump(prediction_history[-5000:], f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass

def verify_predictions():
    verified = 0
    now = now_baghdad()
    for p in prediction_history:
        if p.get("_verified"):
            continue
        tf = p.get("timeframe_hours", 0)
        created = p.get("_created_at", "")
        if not created:
            continue
        try:
            age = (now - datetime.fromisoformat(created)).total_seconds() / 3600
        except Exception:
            continue
        if age < tf:
            continue
        symbol = p["symbol"]
        try:
            klines = get_klines(symbol, "1h", 5)
            if not klines:
                continue
            current_price = float(klines[-1][4])
            predicted_price = p.get("predicted_price", 0)
            if current_price and predicted_price:
                actual_pct = ((current_price - (predicted_price / (1 + p.get("predicted_change_pct", 0) / 100))) / (predicted_price / (1 + p.get("predicted_change_pct", 0) / 100))) * 100
                pred_pct = p.get("predicted_change_pct", 0)
                p["_actual_change_pct"] = round(actual_pct, 2)
                if (pred_pct > 0 and actual_pct > 0) or (pred_pct < 0 and actual_pct < 0):
                    p["_correct"] = True
                    p["_accuracy"] = round(min(abs(actual_pct) / abs(pred_pct), 2) * 100, 1) if pred_pct != 0 else 0
                else:
                    p["_correct"] = False
                    p["_accuracy"] = 0
                p["_verified"] = True
                p["_verified_at"] = now_baghdad().isoformat()
                verified += 1
        except Exception:
            continue
    # Second pass: market-neutral excess return vs the same-scan cohort.
    # Directional accuracy is contaminated by market beta (a bull week makes
    # every UP call "correct"); excess vs same-timestamp median removes beta
    # and is the metric that showed OOS-stable structure (eval_quartile_*).
    try:
        import statistics as _stats
        done = []
        for x in prediction_history:
            if not x.get("_verified") or x.get("_actual_change_pct") is None:
                continue
            if "_excess_pct" in x:
                continue
            try:
                xt = datetime.fromisoformat(x.get("_created_at", ""))
            except Exception:
                x["_excess_pct"] = None
                continue
            done.append((xt, x))
        if done:
            all_pts = [(datetime.fromisoformat(x["_created_at"]), x["_actual_change_pct"])
                       for x in prediction_history
                       if x.get("_verified") and x.get("_actual_change_pct") is not None
                       and x.get("_created_at")]
            for xt, x in done:
                cohort = [v for t, v in all_pts
                          if abs((t - xt).total_seconds()) <= 600]
                if len(cohort) >= 5:
                    x["_excess_pct"] = round(x["_actual_change_pct"] - _stats.median(cohort), 2)
                else:
                    x["_excess_pct"] = None
    except Exception:
        pass
    # Track B -- live measurement of the market-neutral relative-value edge.
    # One cumulative line per cycle: overall + per relative_bias bucket.
    try:
        _byb = {}
        for x in prediction_history:
            if x.get("_excess_pct") is None:
                continue
            _byb.setdefault(x.get("relative_bias", "?"), []).append(x["_excess_pct"])
        _n_all = sum(len(v) for v in _byb.values())
        if _n_all >= 30:
            _all = [e for v in _byb.values() for e in v]
            _line = (f"[RELVAL] excess n={_n_all} mean={sum(_all)/len(_all):+.3f}%"
                     + "".join(f" | {b}:{sum(v)/len(v):+.3f}%(n={len(v)})"
                               for b, v in sorted(_byb.items()) if len(v) >= 10))
            logger.info(_line)
    except Exception:
        pass
    if verified:
        with pred_lock:
            try:
                with open(PRED_HIST, "w", encoding="utf-8") as f:
                    json.dump(prediction_history[-5000:], f, ensure_ascii=False, indent=2, default=str)
            except Exception:
                pass
        # Online learning: update models with verified outcomes
        try:
            from ml_trainer import online_learn
            online_learn()
        except Exception:
            pass
    return verified

def get_prediction_stats():
    total = len(prediction_history)
    verified_count = sum(1 for p in prediction_history if p.get("_verified"))
    correct = sum(1 for p in prediction_history if p.get("_correct") is True)
    wrong = sum(1 for p in prediction_history if p.get("_correct") is False)
    by_tf = {}
    for p in prediction_history:
        if not p.get("_verified"):
            continue
        tf = p.get("timeframe_hours", 0)
        if tf not in by_tf:
            by_tf[tf] = {"total": 0, "correct": 0}
        by_tf[tf]["total"] += 1
        if p.get("_correct"):
            by_tf[tf]["correct"] += 1
    by_symbol = {}
    for p in prediction_history:
        if not p.get("_verified"):
            continue
        sym = p["symbol"]
        if sym not in by_symbol:
            by_symbol[sym] = {"total": 0, "correct": 0}
        by_symbol[sym]["total"] += 1
        if p.get("_correct"):
            by_symbol[sym]["correct"] += 1
    # Market-neutral view: mean excess vs same-scan median, overall and by
    # relative_bias. This is the metric that survives regime shifts; raw
    # directional win-rate mostly reflects which way the market went.
    exc = [p["_excess_pct"] for p in prediction_history
           if p.get("_verified") and p.get("_excess_pct") is not None]
    by_bias = {}
    for p in prediction_history:
        if not p.get("_verified") or p.get("_excess_pct") is None:
            continue
        b = p.get("relative_bias", "?")
        by_bias.setdefault(b, []).append(p["_excess_pct"])
    return {
        "total": total,
        "verified": verified_count,
        "correct": correct,
        "wrong": wrong,
        "win_rate": round(correct / verified_count * 100, 1) if verified_count else 0,
        "mean_excess_pct": round(sum(exc) / len(exc), 3) if exc else None,
        "excess_samples": len(exc),
        "excess_by_bias": {b: {"n": len(v), "mean_excess_pct": round(sum(v) / len(v), 3)}
                           for b, v in sorted(by_bias.items())},
        "by_timeframe": {str(k): v for k, v in sorted(by_tf.items())},
        "by_symbol": dict(sorted(by_symbol.items(), key=lambda x: x[1]["total"], reverse=True)[:10])
    }

def format_daily_report():
    today = now_baghdad().strftime("%Y-%m-%d")
    todays_preds = [p for p in prediction_history if p.get("_created_at", "").startswith(today)]
    verified_today = [p for p in todays_preds if p.get("_verified")]
    if not verified_today:
        return None
    correct = [p for p in verified_today if p.get("_correct") is True]
    wrong = [p for p in verified_today if p.get("_correct") is False]
    perfect_coins = {}
    for p in correct:
        s = p["symbol"]
        if s not in perfect_coins:
            perfect_coins[s] = {"total": 0, "correct": 0}
        perfect_coins[s]["total"] += 1
        perfect_coins[s]["correct"] += 1
    for p in wrong:
        s = p["symbol"]
        if s not in perfect_coins:
            perfect_coins[s] = {"total": 0, "correct": 0}
        perfect_coins[s]["total"] += 1
    perfect_100 = [s for s, v in perfect_coins.items() if v["correct"] == v["total"] and v["total"] >= 2]
    msg  = f"📊 *تقرير نهاية اليوم – {today}*\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n\n"
    msg += f"📈 *التوقعات اليوم:*\n"
    msg += f"  صحيح: `{len(correct)}` | خاطئ: `{len(wrong)}` | الدقة: `{round(len(correct)/len(verified_today)*100,1)}%`\n"
    _exc = [p["_excess_pct"] for p in verified_today if p.get("_excess_pct") is not None]
    if _exc:
        msg += f"  📐 مقابل السوق: `{sum(_exc)/len(_exc):+.2f}%` (متوسط فائض {len(_exc)} عينة)\n"
    # Track B/C -- honest reporting: state the OOS baseline explicitly and
    # surface the cumulative market-neutral edge per relative_bias bucket.
    msg += f"  🎯 الأساس العشوائي (OOS): `51%` — الفائض/الدقة فوق هذا الأساس فقط هي المحتوى\n"
    try:
        _parts = [f"{b}: `{v['mean_excess_pct']:+.3f}%` (n={v['n']})"
                  for b, v in (get_prediction_stats().get("excess_by_bias") or {}).items()
                  if v.get("n", 0) >= 10]
        if _parts:
            msg += f"  📐 حسب الانحياز (تراكمي):\n"
            for _p in _parts:
                msg += f"    • {_p}\n"
    except Exception:
        pass
    msg += "\n"
    if correct:
        msg += f"✅ *التوقعات الصحيحة ({len(correct)}):*\n"
        for p in sorted(correct, key=lambda x: abs(x.get("_accuracy", 0) or 0), reverse=True)[:30]:
            sym = p["symbol"]
            tf = p.get("timeframe_hours", 0)
            acc = p.get("_accuracy", 0) or 0
            pred_pct = p.get("predicted_change_pct", 0)
            actual_pct = p.get("_actual_change_pct", 0)
            msg += f"  • `{sym}` {tf}h: توقع {pred_pct:+.2f}% → فعلي {actual_pct:+.2f}% (دقة {acc}%)\n"
    if wrong:
        msg += f"\n❌ *التوقعات الخاطئة ({len(wrong)}):*\n"
        for p in sorted(wrong, key=lambda x: abs(x.get("predicted_change_pct", 0)), reverse=True)[:10]:
            sym = p["symbol"]
            tf = p.get("timeframe_hours", 0)
            pred_pct = p.get("predicted_change_pct", 0)
            actual_pct = p.get("_actual_change_pct", 0)
            msg += f"  • `{sym}` {tf}h: توقع {pred_pct:+.2f}% → فعلي {actual_pct:+.2f}%\n"
    if perfect_100:
        msg += f"\n🏆 *عملات دقة 100% (≥2 توقعات):*\n"
        for s in perfect_100[:10]:
            msg += f"  • `{s}` – {perfect_coins[s]['correct']}/{perfect_coins[s]['total']} صحيح\n"
    try:
        from paper_trading import paper_engine
        ps = paper_engine.get_stats()
        if ps.get("total_closed", 0) > 0:
            profit_usd = ps['total_return_usd']
            profit_str = f"+${profit_usd:.0f}" if profit_usd >= 0 else f"-${abs(profit_usd):.0f}"
            msg += "\n\n📊 *Paper Trading*\n"
            msg += f"  💰 رأس المال: `${ps['current_capital']:.0f}`\n"
            msg += f"  💵 الربح: `{profit_str}` | `{ps['total_return_pct']:+.1f}%`\n"
            msg += f"  🎯 الصفقات: `{ps['wins']} ربح / {ps['losses']} خسارة` ({ps['win_rate_pct']}%)\n"
            msg += f"  📊 Profit Factor: `{ps['profit_factor']}` | Drawdown: `{ps['drawdown_pct']}%`\n"
    except Exception:
        pass
    msg += f"\n⏰ {fmt_baghdad()} (Baghdad)"
    return msg

def get_daily_report():
    today = now_baghdad().strftime("%Y-%m-%d")
    todays_preds = [p for p in prediction_history if p.get("_created_at", "").startswith(today)]
    verified_today = [p for p in todays_preds if p.get("_verified")]
    return len(verified_today) > 0

def get_market_context():
    try:
        from market_context import get_full_market_context
        return get_full_market_context()
    except Exception:
        return {}

# ─── BOT 1 MESSAGE (AI Signals) ─────────────────────────────────
def format_bot1_message(symbol: str, ta: dict, liq: dict,
                         flow: dict, deriv: dict, mtf: dict,
                         ai: dict, setup: dict, risk: dict = None) -> str:
    price = ta.get("current_price", 0)
    signal = liq.get("signal", "N/A")
    score = ai.get("total", 0)
    grade = ai.get("grade", "")
    msg  = f"📊 *{fmt_symbol(symbol)}*\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"💰 السعر: `{price}`\n"
    msg += f"📈 الاتجاه: {ta.get('trend', 'N/A')}\n"
    msg += f"🎯 الإشارة: {signal}\n\n"
    if risk:
        msg += f"🛡️ المخاطرة: {risk.get('risk_level', 'N/A')}\n\n"
    msg += f"📚 *دفتر الأوامر (أفضل {ORDER_BOOK_LEVELS}):*\n"
    msg += f"  عمق الشراء: {liq.get('bid_depth', 0):>14,.0f}\n"
    msg += f"  عمق البيع : {liq.get('ask_depth', 0):>14,.0f}\n"
    msg += f"  imbalance : {liq.get('imbalance_pct', 0):>10.2f}%\n"
    msg += f"  Spread    : {liq.get('spread_pct', 0):>10.4f}%\n\n"
    if flow:
        msg += f"⚡ *تدفق الأوامر:*\n"
        msg += f"  مشترين: `{flow.get('buy_pct', 0)}%` | بائعين: `{flow.get('sell_pct', 0)}%`\n"
        msg += f"  CVD: {flow.get('cvd_label', 'N/A')} | {flow.get('dominant', '')}\n"
        if flow.get("whale_detected"):
            msg += f"  🐋 نشاط حوت!\n"
        msg += "\n"
    msg += f"📉 *التحليل الفني (1h):*\n"
    msg += f"  RSI: `{ta.get('rsi_label', 'N/A')}`\n"
    msg += f"  MACD: `{ta.get('macd', {}).get('trend', 'N/A')}`\n"
    msg += f"  VWAP: `{ta.get('vwap_position', 'N/A')}`\n"
    msg += f"  EMA20: `{ta.get('ema20', 0)}` | EMA50: `{ta.get('ema50', 0)}`\n"
    msg += f"  BB: `{ta.get('bollinger', {}).get('position', 'N/A')}`\n\n"
    msg += f"🏦 *الذكية (Smart Money):*\n"
    msg += f"  Spoofing: {'⚠️ نعم' if liq.get('spoofing_detected') else '✅ لا'}\n"
    msg += f"  Iceberg: {'⚠️ محتمل' if liq.get('iceberg_suspected') else '✅ واضح'}\n"
    msg += f"  درجة SM: `{liq.get('smart_money_score', 0)}/100`\n\n"
    if deriv:
        msg += f"📊 *المشتقات:*\n"
        if "funding_label" in deriv: msg += f"  التمويل: `{deriv['funding_label']}`\n"
        if "ls_ratio" in deriv: msg += f"  L/S: `{deriv.get('ls_ratio')}` {deriv.get('ls_label', '')}\n"
        if "open_interest" in deriv: msg += f"  الفائدة المفتوحة: `{deriv.get('open_interest'):,.0f}`\n"
        msg += "\n"
    msg += f"🌐 *الأطر المتعددة:* {mtf.get('unified_bias', 'N/A')}\n"
    msg += f"  صاعد: `{mtf.get('bullish_count', 0)}/5` | هابط: `{mtf.get('bearish_count', 0)}/5`\n\n"
    msg += f"🤖 *الذكاء الاصطناعي: `{score}/100`* | {grade}\n"
    msg += f"📊 الثقة: `{ai.get('confidence', 0)}%`\n"
    if ai.get("reasons"):
        msg += f"\n🧠 *أسباب التحليل:*\n"
        for r in ai["reasons"]:
            msg += f"  {r}\n"
    if setup:
        msg += f"\n🎯 *إعداد الصفقة:*\n"
        msg += f"  الدخول: `{setup.get('entry_zone', 'N/A')}`\n"
        msg += f"  وقف الخسارة: `{setup.get('stop_loss', 'N/A')}`\n"
        msg += f"  TP1: `{setup.get('tp1', 'N/A')}` | TP2: `{setup.get('tp2', 'N/A')}` | TP3: `{setup.get('tp3', 'N/A')}`\n"
        msg += f"  R/R: `{setup.get('risk_reward', 0)}x`\n"
    msg += f"\n⏰ {fmt_baghdad()} (بغداد)"
    return msg

# ─── BOT 2 MESSAGE (Liquidity Alert) ────────────────────────────
def format_bot2_message(symbol: str, liq: dict, flow: dict, ta: dict) -> str:
    msg  = f"📊 *تنبيه سيولة: {fmt_symbol(symbol)}*\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"💰 السعر: `{ta.get('current_price', 'N/A')}`\n"
    msg += f"📈 الإشارة: {liq.get('signal', 'N/A')}\n\n"
    msg += f"📚 *دفتر الأوامر:*\n"
    msg += f"  imbalance: `{liq.get('imbalance_pct', 0)}%`\n"
    msg += f"  Spread: `{liq.get('spread_pct', 0)}%`\n"
    msg += f"  عمق الشراء: `{liq.get('bid_depth', 0):,.0f}`\n"
    msg += f"  عمق البيع: `{liq.get('ask_depth', 0):,.0f}`\n\n"
    if liq.get("spoofing_detected"):
        msg += f"⚠️ *تم اكتشاف SPOOFING!*\n\n"
    walls_b = liq.get("bid_walls", [])
    walls_a = liq.get("ask_walls", [])
    if walls_b or walls_a:
        msg += f"🧱 *جدران السيولة:*\n"
        for w in walls_b[:2]: msg += f"  🟢 جدار شراء @ `{w[0]}` حجم `{w[1]:,.0f}`\n"
        for w in walls_a[:2]: msg += f"  🔴 جدار بيع @ `{w[0]}` حجم `{w[1]:,.0f}`\n"
        msg += "\n"
    if flow:
        msg += f"⚡ *تدفق الأوامر:*\n"
        msg += f"  مشترين: `{flow.get('buy_pct', 0)}%` | بائعين: `{flow.get('sell_pct', 0)}%`\n"
        msg += f"  CVD: {flow.get('cvd_label', 'N/A')}\n"
        if flow.get("whale_detected"):
            msg += f"  🐋 *تم اكتشاف حوت!*\n"
    msg += f"\n🕒 {fmt_baghdad()} (بغداد)"
    return msg

# ─── BOT 3 MESSAGE (Price Predictions) ──────────────────────────
PREDICTION_TIMEFRAMES = [24]  # only 24h for speed; add 2,6,8 if needed

def prediction_confidence_label(prediction: dict) -> str:
    """Do not present an uncalibrated model score as a confidence percentage."""
    if prediction.get("confidence_available"):
        return f"{prediction.get('confidence', 0)}%"
    return "غير معايرة"

def format_bot3_message(predictions: list, ctx: dict) -> str:
    if not predictions:
        return ""
    sym = predictions[0].get("symbol", "Unknown")
    price = predictions[0].get("current_price", 0)
    msg  = f"🔮 *{fmt_symbol(sym)} – توقع السعر*\n"
    msg += f"━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"💰 السعر الحالي: `{price}`\n"
    meta_prob = predictions[0].get("meta_labeling_prob")
    if meta_prob is not None:
        reliable = predictions[0].get("meta_labeling_reliable", False)
        msg += f"🔬 Meta-Labeling: `{meta_prob:.0%}` {'✅ موثوق' if reliable else '❌ غير موثوق'}\n"
    market_sent = predictions[0].get("market_sentiment")
    if market_sent:
        msg += f"🌐 السوق: `{market_sent}`"
        fr = predictions[0].get("funding_rate_pct", 0)
        if fr: msg += f" | Funding: `{fr:+.4f}%`"
        oi = predictions[0].get("open_interest_change_1h_pct")
        if oi: msg += f" | OI: `{oi:+.1f}%`"
        msg += "\n"
    risk = predictions[0].get("risk", {})
    if risk:
        sl = risk.get("suggested_stop_loss_pct", 0)
        tp = risk.get("suggested_take_profit_pct", 0)
        pos = risk.get("suggested_position_size_pct", 0)
        rr = risk.get("risk_reward_ratio", 0)
        msg += f"🎯 إدارة المخاطر: SL `{sl}%` TP `{tp}%` | R:R `{rr}` | حجم `{pos}%`\n"
    msg += "\n"
    if ctx:
        btc = ctx.get("btc", {})
        fng = ctx.get("fear_greed", {})
        msg += f"฿ BTC: `${btc.get('btc_price', '?')}` ({btc.get('btc_change_24h', 0):+.2f}%)\n"
        msg += f"📊 هيمنة BTC: `{btc.get('btc_dominance', 0):.1f}%`"
        if fng:
            msg += f" | F&G: `{fng.get('value', '?')}` ({fng.get('classification', '')})"
        msg += "\n\n"
    msg += f"📈 *الحركة المتوقعة (> {PREDICTION_THRESH}%):*\n"
    header_shown = False
    for p in predictions:
        if "error" in p:
            continue
        pct = p.get("predicted_change_pct", 0)
        conf = prediction_confidence_label(p)
        if pct < PREDICTION_THRESH:
            continue
        tf = p.get("timeframe_hours", 0)
        timing = p.get("entry_timing", "N/A")
        if not header_shown:
            msg += "```\n"
            msg += f"  {'TF':>5}  {'التغير%':>8}  {'P↑/P↓/P→':>14}  {'ثقة':>10}  الدخول\n"
            msg += f"  {'─'*5}  {'─'*8}  {'─'*14}  {'─'*10}  ─────\n"
            header_shown = True
        probs = f"{p.get('prob_up', 0):.0%}/{p.get('prob_down', 0):.0%}/{p.get('prob_neutral', 0):.0%}"
        msg += f"  {tf:>3}h  {pct:>+6.2f}%  {probs:>14}  {conf:>10}  {timing[:12]:>12}\n"
    if header_shown:
        msg += "```\n"
    else:
        msg += "  ⚠️ لا توجد عملات بارتفاع متوقع > 3%\n\n"
    above_3 = sorted(
        [p for p in predictions if "error" not in p and p.get("predicted_change_pct", 0) >= PREDICTION_THRESH],
        key=lambda x: x.get("predicted_change_pct", 0), reverse=True
    )
    if above_3:
        msg += f"\n🏆 *أفضل التوقعات (> {PREDICTION_THRESH}%):*\n"
        msg += "```\n"
        msg += f"  {'#':>2}  {'العملة':>12}  {'TF':>4}  {'التغير%':>8}  {'الثقة':>10}  {'التقييم':>20}\n"
        msg += f"  {'─'*2}  {'─'*12}  {'─'*4}  {'─'*8}  {'─'*10}  {'─'*20}\n"
        for i, p in enumerate(above_3[:20], 1):
            msg += f"  {i:>2}  {fmt_symbol(p.get('symbol', '?')):>12}  {p.get('timeframe_hours', 0):>2}h  {p.get('predicted_change_pct', 0):>+6.2f}%  {prediction_confidence_label(p):>10}  {p.get('grade', ''):>20}\n"
        msg += "```\n"
    msg += f"\n⏰ {fmt_baghdad()} (بغداد)\n"
    msg += f"_المسح التالي كل {SCAN_INTERVAL_MINUTES} دقيقة_"
    return msg


# ─── ANALYZE SYMBOL ────────────────────────────────────────────
def analyze_symbol(symbol: str, lite_mode: bool = True) -> dict:
    logger.info(f"  → {symbol}")
    result = {"symbol": symbol}
    try:
        ticker = get_ticker_24h(symbol)
        result["price"] = float(ticker.get("lastPrice", 0))
        result["change24"] = float(ticker.get("priceChangePercent", 0))
        result["vol24"] = float(ticker.get("quoteVolume", 0))
        ta = get_technical_analysis(symbol, "1h")
        result["ta"] = ta
        flow = analyze_order_flow(symbol)
        result["flow"] = flow
        if not lite_mode:
            deriv = analyze_derivatives(symbol)
            result["deriv"] = deriv
        else:
            result["deriv"] = {}
        liq = analyze_liquidity(symbol)
        result["liq"] = liq
        if not lite_mode:
            mtf = multi_timeframe_analysis(symbol)
            result["mtf"] = mtf
        else:
            result["mtf"] = {}
        ai = compute_ai_score(liq, ta, flow, result.get("deriv", {}), result.get("mtf", {}))
        result["ai"] = ai
        risk = compute_risk_level(ta, liq, flow, result.get("deriv", {}))
        result["risk"] = risk
        setup = compute_trade_setup(ta, liq)
        result["setup"] = setup
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        result["error"] = str(e)
    return result


# ─── BOT 3 PREDICTION SCAN ─────────────────────────────────────
def run_prediction_scan():
    logger.info("=" * 60)
    logger.info(f"🔮 Bot 3 – Prediction scan at {fmt_baghdad()}")
    logger.info("=" * 60)
    # Verify past predictions
    v = verify_predictions()
    if v:
        logger.info(f"[BOT3] Verified {v} predictions")
    # Clear ML feature cache for fresh data
    try:
        from ml_trainer import clear_feature_cache
        clear_feature_cache()
    except Exception:
        pass
    # Rank by 24h volume, not alphabetically: the old [:15] slice always picked
    # AAVE..BTC and never reached ETH, SOL, XRP or DOGE.
    symbols = get_symbols_by_volume(15)
    if not symbols:
        logger.warning("[BOT3] No symbols to predict (Binance ticker unavailable?)")
        return
    logger.info(f"[BOT3] Predicting {len(symbols)} symbols x {len(PREDICTION_TIMEFRAMES)} TFs...")
    ctx = get_market_context()
    all_predictions = []
    for i, sym in enumerate(symbols):
        for tf in PREDICTION_TIMEFRAMES:
            try:
                r = predict_movement(sym, tf)
                all_predictions.append(r)
                if "error" in r:
                    logger.warning(f"[BOT3] {sym} {tf}h failed: {r['error']}")
            except Exception as e:
                logger.warning(f"[BOT3] {sym} {tf}h crashed: {e}")
        if (i + 1) % 10 == 0:
            logger.info(f"[BOT3] {i+1}/{len(symbols)} symbols predicted")
    n_err = sum(1 for p in all_predictions if "error" in p)
    if n_err:
        logger.warning(f"[BOT3] {n_err}/{len(all_predictions)} predictions failed "
                       f"(first: {next(p.get('error') for p in all_predictions if 'error' in p)})")
    # Save each prediction with metadata for future verification
    for p in all_predictions:
        if "error" not in p:
            save_prediction(p)
    above_threshold = [p for p in all_predictions
                        if "error" not in p and p.get("predicted_change_pct", 0) >= PREDICTION_THRESH]
    # Directional signals are REPORTED but NOT auto-traded. Evidence: the 24h
    # directional ML edge is dead (48.9% over 6784 verified; <50% at 6/12/24/48h),
    # so auto-opening on it would burn capital on noise. Live trading leans on
    # the relative-value (pairs) edge via run_pairs_scan() instead.
    if above_threshold:
        logger.info(f"[BOT3] {len(above_threshold)} directional signals reported (not auto-traded; directional edge is dead)")
    # Accuracy stats
    stats = get_prediction_stats()
    if above_threshold:
        by_symbol = {}
        for p in above_threshold:
            s = p["symbol"]
            if s not in by_symbol:
                by_symbol[s] = []
            by_symbol[s].append(p)
        for sym, preds in sorted(by_symbol.items()):
            msg = format_bot3_message(preds, ctx)
            if len(msg) > 4000:
                msg = msg[:3900] + "\n... (truncated)"
            send_telegram(BOT3_TOKEN, msg)
            time.sleep(1.5)
        summary = sorted(above_threshold, key=lambda x: x["predicted_change_pct"], reverse=True)
        summary_msg = f"📊 *ملخص التوقعات (> {PREDICTION_THRESH}%)*\n"
        summary_msg += f"━━━━━━━━━━━━━━━━━━━━━━\n"
        summary_msg += f"إجمالي الإشارات: `{len(above_threshold)}`\n"
        if stats["verified"]:
            summary_msg += f"📈 نسبة الربح: `{stats['win_rate']}%` ({stats['correct']}/{stats['verified']})\n\n"
        else:
            summary_msg += "\n"
        top_20 = {}
        for p in summary:
            s = p["symbol"]
            if s not in top_20 or p["predicted_change_pct"] > top_20[s]["predicted_change_pct"]:
                top_20[s] = p
        top_20_list = sorted(top_20.values(), key=lambda x: x["predicted_change_pct"], reverse=True)[:20]
        summary_msg += f"🏆 *أفضل 20 عملة:*\n"
        summary_msg += "```\n"
        summary_msg += f"  {'#':>2}  {'العملة':>12}  {'TF':>4}  {'التغير%':>8}  {'الثقة':>10}  {'الدخول':>12}\n"
        summary_msg += f"  {'─'*2}  {'─'*12}  {'─'*4}  {'─'*8}  {'─'*10}  {'─'*12}\n"
        for i, p in enumerate(top_20_list, 1):
            summary_msg += f"  {i:>2}  {fmt_symbol(p['symbol']):>12}  {p['timeframe_hours']:>2}h  {p['predicted_change_pct']:>+6.2f}%  {prediction_confidence_label(p):>10}  {p.get('entry_timing', 'N/A')[:12]:>12}\n"
        summary_msg += "```\n"
        if stats["verified"]:
            summary_msg += f"\n📈 *إحصائيات الدقة:*\n"
            summary_msg += f"  نسبة الربح: `{stats['win_rate']}%` ({stats['correct']}/{stats['verified']})\n"
            tf_stats = stats.get("by_timeframe", {})
            if tf_stats:
                summary_msg += f"  حسب الإطار:\n"
                for tf, s_ in tf_stats.items():
                    wr = round(s_["correct"] / s_["total"] * 100, 1) if s_["total"] else 0
                    summary_msg += f"    {tf}h: `{wr}%` ({s_['correct']}/{s_['total']})\n"
            sym_stats = stats.get("by_symbol", {})
            if sym_stats:
                best_sym = max(sym_stats, key=lambda x: sym_stats[x]["correct"] / max(sym_stats[x]["total"], 1))
                summary_msg += f"  أفضل عملة: `{best_sym}` ({sym_stats[best_sym]['correct']}/{sym_stats[best_sym]['total']})\n"
        summary_msg += f"\n⏰ {fmt_baghdad()} (بغداد)"
        send_telegram(BOT3_TOKEN, summary_msg)
    else:
        msg = f"🔮 *مسح التوقعات*\nلا توجد عملات بارتفاع متوقع > {PREDICTION_THRESH}%."
        if stats["verified"]:
            msg += f"\n\n📈 الدقة: `{stats['win_rate']}%` ({stats['correct']}/{stats['verified']})"
        msg += f"\n\n⏰ {fmt_baghdad()} (بغداد)"
        send_telegram(BOT3_TOKEN, msg)
    # Paper Trading summary after each scan
    try:
        from paper_trading import paper_engine
        ps = paper_engine.get_stats()
        if ps.get("total_closed", 0) > 0 or ps.get("active_trades", 0) > 0:
            profit_usd = ps['total_return_usd']
            profit_str = f"+${profit_usd:.0f}" if profit_usd >= 0 else f"-${abs(profit_usd):.0f}"
            pt_msg = "📊 *Paper Trading*\n━━━━━━━━━━━━━━━━━━\n"
            pt_msg += f"💰 رأس المال: `${ps['current_capital']:.0f}`\n"
            pt_msg += f"📈 الربح: `{profit_str}` | `{ps['total_return_pct']:+.1f}%`\n"
            pt_msg += f"🎯 Win Rate: `{ps['win_rate_pct']}%` ({ps['wins']}/{ps['total_closed']})\n"
            pt_msg += f"📊 Profit Factor: `{ps['profit_factor']}`\n"
            pt_msg += f"📉 Drawdown: `{ps['drawdown_pct']}%`\n"
            if ps['active_trades']:
                pt_msg += f"\n🟢 صفقات مفتوحة: `{ps['active_trades']}`"
            pt_msg += f"\n\n⏰ {fmt_baghdad()} (بغداد)"
            send_telegram(BOT3_TOKEN, pt_msg)
    except Exception:
        pass
    try:
        with open(PRED_LOG, "w", encoding="utf-8") as f:
            json.dump(all_predictions, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass
    logger.info(f"[BOT3] Complete | {len(above_threshold)} coins > {PREDICTION_THRESH}% | WR: {stats['win_rate']}%")

# ─── PAIRS SCAN (relative-value / stat-arb) ────────────────────
# ─── Track A: weekly pairs universe refilter ─────────────────────────────
# pairs_robust.json was generated once (Aug 8) and never re-verified; every
# Monday >= 06:00 Baghdad we re-run the same robust scan in a daemon thread
# and atomically rewrite pairs_robust/pairs_active. Empty results are
# discarded so a failed refilter can never wipe a working universe.
PAIRS_REFILTER_STATE = os.path.join(BASE_DIR, "pairs_refilter_state.json")
_pairs_refilter_thread = None

_REFILTER_UNIVERSE = sorted(set([
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "LTCUSDT", "LINKUSDT", "DOTUSDT", "AVAXUSDT", "TRXUSDT",
    "FILUSDT", "MATICUSDT", "ATOMUSDT", "ETCUSDT", "NEARUSDT", "APTUSDT",
    "ARBUSDT", "OPUSDT", "SUIUSDT", "TONUSDT", "PEPEUSDT", "SHIBUSDT",
    "BCHUSDT", "UNIUSDT", "AAVEUSDT", "MKRUSDT", "INJUSDT", "1000SATSUSDT",
    "LDOUSDT", "RUNEUSDT",
]))
_REFILTER_WINDOWS = [(2500, 1000), (3000, 1000), (3500, 1000)]


def _refilter_state_read():
    try:
        with open(PAIRS_REFILTER_STATE, encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _weekly_pairs_refilter():
    import time as _time
    try:
        import stat_arb as _sa
        import numpy as _np
        t0 = _time.time()
        logger.info(f"[PAIRS-REFILTER] start universe={len(_REFILTER_UNIVERSE)}")
        screen = _sa.scan_pairs(_REFILTER_UNIVERSE, "1h", 3000,
                                p_max=0.05, hl_min=4, hl_max=800)
        short = [r for r in screen if r.get("status") == "shortlisted"]
        logger.info(f"[PAIRS-REFILTER] stage-1 screened={len(screen)} "
                    f"shortlisted={len(short)}")
        robust = []
        for r in short:
            a, b = r["pair"].split("/", 1)
            wins = {}
            for lb, tb in _REFILTER_WINDOWS:
                try:
                    df = _sa.fetch_aligned_pair(a, b, "1h", lb)
                    if df is None or len(df) < tb + 50:
                        continue
                    m = _sa.walk_forward(df, "1h", train_bars=tb, step=tb // 4,
                                         z_entry=2.0, z_exit=0.5, cost_bps=25)
                    if m["cointegrated"] and m["n_trades"] > 0 and m["sharpe"] > 0:
                        wins[f"w{lb}"] = (round(m["sharpe"], 2), m["n_trades"])
                except Exception:
                    pass
            if len(wins) >= 2:
                robust.append({"pair": r["pair"], "windows": wins,
                               "mean_sharpe": round(
                                   float(_np.mean([s for s, _ in wins.values()])), 2)})
        if not robust:
            logger.warning("[PAIRS-REFILTER] no robust pairs — keeping current universe")
            return
        active = [p for p in robust if p["mean_sharpe"] >= 1.0 and len(p["windows"]) >= 2]
        if not active:
            logger.warning("[PAIRS-REFILTER] no pair >=1.0 mean_sharpe — keeping current universe")
            return
        active.sort(key=lambda x: -x["mean_sharpe"])
        # never orphan an open paper position
        try:
            with open(os.path.join(BASE_DIR, "paper_pairs.json"), encoding="utf-8") as f:
                _open = {p.get("pair") for p in json.load(f).get("positions", [])
                         if p.get("pair") and not p.get("exit_reason")}
        except Exception:
            _open = set()
        kept = sorted(_open - {p["pair"] for p in active})
        if kept:
            active += [{"pair": k, "windows": {}, "mean_sharpe": 0.0,
                        "kept_for_open_position": True} for k in kept]
        gen = now_baghdad().isoformat(timespec="seconds")
        robust_out = {"generated": gen, "universe_size": len(_REFILTER_UNIVERSE),
                      "windows": [f"{lb}/{tb}" for lb, tb in _REFILTER_WINDOWS],
                      "source": "weekly_refilter", "pairs": robust}
        active_out = {"generated": gen, "min_sharpe": 1.0, "min_windows": 2,
                      "source": "pairs_robust.json", "pairs": active}
        for name, payload in (("pairs_robust.json", robust_out),
                              ("pairs_active.json", active_out)):
            tmp = os.path.join(BASE_DIR, name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, os.path.join(BASE_DIR, name))
        st = _refilter_state_read()
        st.update({"last_ok_ts": _time.time(), "generated": gen,
                   "n_robust": len(robust), "n_active": len(active)})
        with open(PAIRS_REFILTER_STATE, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=2)
        logger.info(f"[PAIRS-REFILTER] done robust={len(robust)} "
                    f"active={len(active)} kept_open={len(kept)} "
                    f"elapsed={_time.time() - t0:.0f}s")
        send_telegram(BOT1_TOKEN,
                      f"🔗 *إعادة فلترة أسبوعية للبائرز*\n"
                      f"مرنة: `{len(robust)}` | نشط: `{len(active)}`"
                      + (f" | مفتوحة محفوظة: `{len(kept)}`" if kept else ""))
    except Exception as e:
        logger.warning(f"[PAIRS-REFILTER] failed: {e}")


def _maybe_weekly_pairs_refilter():
    """Trigger Track A once per Monday >= 06:00 Baghdad (max one launch/day)."""
    global _pairs_refilter_thread
    try:
        if _pairs_refilter_thread is not None and _pairs_refilter_thread.is_alive():
            return
        n = now_baghdad()
        if n.weekday() != 0 or n.hour < 6:
            return
        import time as _time
        st = _refilter_state_read()
        today = n.strftime("%Y-%m-%d")
        if today in (st.get("attempt_days") or []):
            return
        last_ok = float(st.get("last_ok_ts", 0.0) or 0.0)
        if last_ok and _time.time() - last_ok < 6.5 * 86400:
            return
        st["attempt_days"] = ((st.get("attempt_days") or []) + [today])[-8:]
        with open(PAIRS_REFILTER_STATE, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=2)
        import threading as _threading
        _pairs_refilter_thread = _threading.Thread(target=_weekly_pairs_refilter,
                                                   daemon=True, name="pairs-refilter")
        _pairs_refilter_thread.start()
        logger.info("[PAIRS-REFILTER] scheduled (weekly Monday refilter)")
    except Exception:
        pass


def run_pairs_scan():
    """Open/close two-leg paper positions for the OOS-approved universe.

    The approved pairs live in ``pairs_universe.json`` (produced by
    ``stat_arb --rescan``). We trade ONLY those — every other pair is noise.
    ``pair_paper`` keeps its own persistent book in ``paper_pairs.json``.
    """
    logger.info("=" * 60)
    logger.info(f"🔗 Pairs scan at {fmt_baghdad()}")
    logger.info("=" * 60)
    _maybe_weekly_pairs_refilter()
    univ_file = os.path.join(BASE_DIR, "pairs_active.json")
    if not os.path.exists(univ_file):
        univ_file = os.path.join(BASE_DIR, "pairs_universe.json")
    if not os.path.exists(univ_file):
        logger.info("[PAIRS] no pairs_universe.json (run stat_arb --rescan first)")
        return
    try:
        with open(univ_file, "r", encoding="utf-8") as f:
            univ = json.load(f)
    except Exception as e:
        logger.warning(f"[PAIRS] could not read universe: {e}")
        return
    approved = univ.get("approved", univ.get("pairs", []))
    if not approved:
        logger.info("[PAIRS] universe empty — no pairs approved")
        return
    try:
        from pair_paper import PairPaperBook
    except Exception as e:
        logger.warning(f"[PAIRS] pair_paper import failed: {e}")
        return
    book = PairPaperBook()
    events = []
    # Regime gate: only open new pairs when market conditions are favorable
    try:
        from regime_predictor import predict_regime
        from data_loader.binance_ohlcv import fetch_klines
        regime_ok = True
        regime_note = ""
        # Use BTC as the regime proxy (fast)
        btc_k = fetch_klines("BTCUSDT", "1h", 200)
        reg = predict_regime(btc_k)
        if reg.get("status") == "ok":
            regime_ok = reg["score"] >= 60
            regime_note = f"regime={reg['regime']} score={reg['score']} action={reg['action']}"
            logger.info(f"[PAIRS] {regime_note}")
    except Exception as e:
        regime_ok = True  # if regime check fails, don't block trading
        regime_note = f"regime check failed: {e}"
        logger.warning(f"[PAIRS] {regime_note}")

    for rec in approved:
        pair = rec.get("pair", "")
        if "/" not in pair:
            continue
        a, b = pair.split("/", 1)
        # If regime is unfavorable, only manage (close) existing positions
        if not regime_ok and book._position_index(pair) < 0:
            continue
        try:
            ev = book.step(a, b, interval="1h", lookback=1500, train_bars=1000,
                            z_entry=2.0, z_exit=0.5, z_stop=3.5)
            events.extend(ev)
        except Exception as e:
            logger.warning(f"[PAIRS] step failed for {pair}: {e}")
    for ev in events:
        t = ev.get("type")
        if t == "open":
            msg = (f"🔗 *Pairs OPEN* {ev.get('direction')} `{ev.get('pair')}`\n"
                   f"z={ev.get('z')} hedge={ev.get('hedge')} size={ev.get('size_pct',0):.0%}")
            logger.info(f"[PAIRS] OPEN {ev.get('direction')} {ev.get('pair')} z={ev.get('z')}")
            send_telegram(BOT1_TOKEN, msg)
        elif t == "close":
            pnl = ev.get("pnl_pct", 0)
            emoji = "✅" if pnl and pnl > 0 else "❌"
            msg = (f"{emoji} *Pairs CLOSE* `{ev.get('pair')}` {ev.get('reason')}\n"
                   f"PnL: `{pnl:+.2f}%` bars={ev.get('bars_held')}")
            logger.info(f"[PAIRS] CLOSE {ev.get('pair')} {ev.get('reason')} pnl={pnl}%")
            send_telegram(BOT1_TOKEN, msg)
    logger.info(f"[PAIRS] scanned {len(approved)} | open={len(book.open_positions)} | events={len(events)}")


# ─── FULL SCAN (Bot 1 + Bot 2) ─────────────────────────────────
def run_full_scan(lite_mode=True):
    """
    Full market scan for Bot 1 (signals) and Bot 2 (liquidity).
    lite_mode=True: fast scan (top 30 symbols, skips heavy analyses)
    lite_mode=False: deep scan (all symbols, full analyses) - for scheduled deep scans
    """
    global monitoring_active
    if not monitoring_active:
        return
    logger.info("=" * 60)
    logger.info(f"🔍 Market scan at {fmt_baghdad()} (lite={lite_mode})")
    logger.info("=" * 60)
    
    # Get symbols - limit to top 30 by volume in lite mode for speed.
    # (Alphabetical order used to hide ETH/SOL/XRP/DOGE from every scan.)
    if lite_mode:
        symbols = get_symbols_by_volume(30)
        all_symbols = symbols
    else:
        all_symbols = get_qualified_symbols()
        symbols = all_symbols
    if not all_symbols:
        return
    logger.info(f"Scanning {len(symbols)} symbols (of {len(all_symbols)} qualified)...")
    
    p1_signals, p2_signals = [], []
    max_workers = 10 if lite_mode else 8
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(analyze_symbol, sym, lite_mode): sym for sym in symbols}
        for f in as_completed(futures):
            sym = futures[f]
            try:
                r = f.result()
                ai_score = r.get("ai", {}).get("total", 0)
                liq = r.get("liq", {})
                if ai_score >= AI_SCORE_THRESHOLD:
                    p1_signals.append(r)
                    logger.info(f"  🎯 [{sym}] AI: {ai_score}")
                prev = liquidity_state.get(sym, {})
                imb = liq.get("imbalance_pct", 0)
                prev_imb = prev.get("imbalance_pct", 0)
                is_liq = (abs(imb) > LIQUIDITY_IMBALANCE_THRESH and abs(prev_imb) <= LIQUIDITY_IMBALANCE_THRESH) \
                         or liq.get("spoofing_detected") or liq.get("bid_walls") or liq.get("ask_walls")
                if is_liq:
                    p2_signals.append(r)
                with state_lock:
                    liquidity_state[sym] = {"imbalance_pct": imb, "spread_pct": liq.get("spread_pct", 0), "timestamp": now_baghdad().isoformat()}
                log_entry = {"timestamp": now_baghdad().isoformat(), "symbol": sym, "ai_score": ai_score, "imbalance": imb, "spread": liq.get("spread_pct", 0), "signal": liq.get("signal", "")}
                with open(LOG_FILE, "a", encoding="utf-8") as lf:
                    lf.write(json.dumps(log_entry) + "\n")
            except Exception as e:
                logger.error(f"Error {sym}: {e}")
            time.sleep(0.01)
    # Bot 1
    p1_signals.sort(key=lambda x: x.get("ai", {}).get("total", 0), reverse=True)
    for r in p1_signals[:10]:
        try:
            sym = r["symbol"]
            msg = format_bot1_message(sym, r.get("ta", {}), r.get("liq", {}),
                                       r.get("flow", {}), r.get("deriv", {}),
                                       r.get("mtf", {}), r.get("ai", {}),
                                       r.get("setup", {}), r.get("risk", {}))
            send_telegram(BOT1_TOKEN, msg)
            time.sleep(1)
        except Exception as e:
            logger.error(f"Bot1 send failed for {sym}: {e}")
    # Bot 2
    for r in p2_signals[:15]:
        try:
            msg = format_bot2_message(r["symbol"], r.get("liq", {}), r.get("flow", {}), r.get("ta", {}))
            send_telegram(BOT2_TOKEN, msg)
            time.sleep(0.8)
        except Exception as e:
            logger.error(f"Bot2 send failed: {e}")
    # Persist
    with state_lock:
        try:
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(liquidity_state, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    logger.info(f"✅ Scan | Bot1: {len(p1_signals)} signals | Bot2: {len(p2_signals)} alerts")


# ─── BOT 4 MASTER SIGNAL (3/3 Agreement) ──────────────────────

def run_master_scan():
    """Find coins where ALL 3 bots agree → send one strong signal."""
    logger.info("=" * 60)
    logger.info(f"🏆 Bot 4 – Master scan at {fmt_baghdad()}")
    logger.info("=" * 60)
    symbols = get_symbols_by_volume(15)
    if not symbols:
        logger.warning("[BOT4] No symbols available (Binance ticker unavailable?)")
        return
    logger.info(f"[BOT4] Scanning {len(symbols)} for audited convergence...")
    try:
        from ml_trainer import clear_feature_cache
        clear_feature_cache()
    except Exception:
        pass

    # Get Bot 3 predictions for all symbols — load cached results from
    # prediction_history.json (computed by Bot3 scan) instead of re-running
    # the slow foundation models (83s each on CPU).  If no cached prediction
    # exists for a symbol, fall back to live compute (throttled).
    preds_by_sym = {}
    cached_preds = {}
    try:
        hist = []
        if os.path.exists(PRED_HIST):
            with open(PRED_HIST, "r", encoding="utf-8") as f:
                hist = json.load(f)
        if not hist and hasattr(prediction_tracker, 'data'):
            hist = prediction_tracker.data.get("predictions", [])
        cutoff = time.time() - 7200  # last 2 hours only
        for p in hist:
            if p.get("_verified") or p.get("error"):
                continue
            created = p.get("_created_at") or p.get("created_at") or ""
            try:
                ct = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
                if ct < cutoff:
                    continue
            except Exception:
                continue
            sym = p.get("symbol", "")
            if sym not in cached_preds:
                cached_preds[sym] = []
            cached_preds[sym].append(p)
        logger.info(f"[BOT4] Loaded {len(cached_preds)} cached predictions from history")
    except Exception as e:
        logger.warning(f"[BOT4] Could not load prediction cache: {e}")

    # Bot 4 only analyses symbols that Bot 3 actually predicted.  The volume
    # ranking can shift between the two scans, so keep any cached symbol even
    # if it dropped out of today's top list (capped so the scan stays fast).
    scan_syms = [s for s in symbols if s in cached_preds]
    for sym in cached_preds:
        if sym not in scan_syms and len(scan_syms) < 18:
            scan_syms.append(sym)
    for sym in scan_syms:
        preds_by_sym[sym] = cached_preds[sym]
    skipped = [s for s in symbols if s not in cached_preds]
    if skipped:
        logger.info("[BOT4] No cached pred for %d symbol(s) (Bot3 covers the top 15): %s",
                    len(skipped), ", ".join(skipped[:10]))
    logger.info(f"[BOT4] Usable predictions for {len(preds_by_sym)} symbols")

    if not preds_by_sym:
        # Last-resort fallback so Boss is never blind.  The old comment above
        # promised "fall back to live compute" but the code just returned, which
        # is exactly why Boss went completely silent when Bot 3 produced nothing.
        live_n = min(5, len(symbols))
        logger.warning(f"[BOT4] No cached predictions in the last 2h; "
                       f"computing {live_n} live fallback predictions")
        for sym in symbols[:live_n]:
            try:
                r = predict_movement(sym, PREDICTION_TIMEFRAMES[0])
                if r and "error" not in r:
                    preds_by_sym[sym] = [r]
                else:
                    logger.warning(f"[BOT4] live fallback failed for {sym}: {(r or {}).get('error')}")
            except Exception as e:
                logger.warning(f"[BOT4] live fallback crashed for {sym}: {e}")

    if not preds_by_sym:
        logger.warning("[BOT4] Still no predictions; sending heartbeat instead of silence.")
        send_telegram(BOT4_TOKEN,
                      "📡 *Boss: مسح ماسح*\n"
                      "Bot 3 لم ينتج توقعات جديدة ولم ينجح الحساب المباشر، لذلك لا يوجد ما يُفحص الآن.\n"
                      "سأعيد المحاولة في الدورة القادمة.\n\n"
                      f"🕒 {fmt_baghdad()} (بغداد)")
        return

    ctx = get_market_context()
    master_signals = []

    for sym, preds in sorted(preds_by_sym.items()):
        try:
            ta = get_technical_analysis(sym, "1h")
            liq = analyze_liquidity(sym)
            flow = analyze_order_flow(sym)
            deriv = analyze_derivatives(sym)
            mtf = multi_timeframe_analysis(sym)
            ai = compute_ai_score(liq, ta, flow, deriv, mtf)
            risk = compute_risk_level(ta, liq, flow, deriv)
            setup = compute_trade_setup(ta, liq)

            ai_score = ai.get("total", 0)
            imb = liq.get("imbalance_pct", 0)
            best_pred = max(preds, key=lambda x: abs(x.get("predicted_change_pct", 0)))
            from decision_engine import evaluate_signal
            decision = evaluate_signal(
                best_pred, liq, flow, ai_score,
                ai_threshold=AI_SCORE_THRESHOLD,
                min_probability=0.55,
                min_probability_gap=0.10,
            )
            if decision.decision != "TRADE_READY":
                logger.info("[BOT4] %s -> %s (%s)", sym, decision.decision,
                            ", ".join(decision.reasons + decision.conflicts + decision.notes))
                continue

            master_signals.append({
                "symbol": sym,
                "price": float(ta.get("current_price", 0)),
                "ai_score": ai_score,
                "ai_grade": ai.get("grade", ""),
                "imbalance": imb,
                "liquidity_signal": liq.get("signal", ""),
                "prediction": best_pred,
                "setup": setup,
                "risk": risk.get("risk_level", "N/A"),
                "walls_b": liq.get("bid_walls", []),
                "walls_a": liq.get("ask_walls", []),
                "spoofing": liq.get("spoofing_detected", False),
                "flow_buy_pct": flow.get("buy_pct", 50) if flow else 50,
                "meta_labeling_prob": best_pred.get("meta_labeling_prob"),
                "meta_labeling_reliable": best_pred.get("meta_labeling_reliable", False),
                "market_sentiment": best_pred.get("market_sentiment"),
                "funding_rate_pct": best_pred.get("funding_rate_pct"),
                "long_short_ratio": best_pred.get("long_short_ratio"),
                "decision": decision.as_dict(),
            })
        except Exception:
            continue

    if not master_signals:
        logger.info("[BOT4] No trade-ready convergence found")
        send_telegram(BOT4_TOKEN, f"🏆 *مسح ماستر*\nلا توجد إشارات مكتملة المعايرة والتوافق حالياً.\n\n⏰ {fmt_baghdad()} (بغداد)")
        return

    master_signals.sort(key=lambda x: x["prediction"]["predicted_change_pct"], reverse=True)
    logger.info(f"[BOT4] {len(master_signals)} master signals found!")

    # Local audit trail: every signal we are about to send is persisted so
    # its later _excess_pct/_actual_change_pct can be joined offline.
    try:
        with open(os.path.join(BASE_DIR, "master_signals.json"), "a",
                  encoding="utf-8") as sf:
            for s in master_signals:
                rec = {
                    "ts": now_baghdad().isoformat(),
                    "symbol": s["symbol"],
                    "price": s["price"],
                    "pred_change": s["prediction"].get("predicted_change_pct"),
                    "pred_created_at": s["prediction"].get("created_at"),
                    "ai_score": s["ai_score"],
                    "bias": s["prediction"].get("relative_bias"),
                    "prob_up": s["prediction"].get("prob_up"),
                    "prob_up_raw": s["prediction"].get("prob_up_raw"),
                    "decision": s["decision"],
                }
                sf.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                logger.info("[BOT4] TRADE_READY %s pred=%s%% ai=%s bias=%s",
                            s["symbol"], rec["pred_change"], s["ai_score"], rec["bias"])
    except Exception as e:
        logger.warning(f"[BOT4] could not persist master signals: {e}")

    # Send individual master signals
    for s in master_signals[:5]:
        msg = format_bot4_message(s, ctx)
        if len(msg) > 4000:
            msg = msg[:3900] + "\n... (مختصر)"
        send_telegram(BOT4_TOKEN, msg)
        time.sleep(1.5)

    # Summary of all master signals
    if len(master_signals) > 1:
        summ = f"🏆 *ملخص إشارات ماستر ({len(master_signals)})*\n"
        summ += "━━━━━━━━━━━━━━━━━━━━━━\n\n"
        for s in master_signals:
            pct = s["prediction"]["predicted_change_pct"]
            tf = s["prediction"]["timeframe_hours"]
            conf = prediction_confidence_label(s["prediction"])
            direction = "LONG 🟢" if pct > 0 else "SHORT 🔴"
            summ += f"• `{fmt_symbol(s['symbol'])}` {direction} | AI: `{s['ai_score']}` | {pct:+.1f}% ({tf}h) | ثقة `{conf}`\n"
        summ += f"\n⏰ {fmt_baghdad()} (بغداد)"
        send_telegram(BOT4_TOKEN, summ)

    logger.info(f"[BOT4] Complete | {len(master_signals)} master signals")

def format_bot4_message(s: dict, ctx: dict) -> str:
    sym = s["symbol"]
    price = s["price"]
    p = s["prediction"]
    setup = s["setup"]
    pct = p.get("predicted_change_pct", 0)
    conf = prediction_confidence_label(p)
    tf = p.get("timeframe_hours", 0)
    grade = p.get("grade", "")

    direction = "LONG 🟢" if pct > 0 else "SHORT 🔴"
    entry = setup.get("entry_zone", "—") if setup else "—"
    sl = setup.get("stop_loss", "—") if setup else "—"
    tp1 = setup.get("tp1", "—") if setup else "—"

    msg = f"🏆 *{fmt_symbol(sym)} – إشارة ماستر*\n"
    msg += "━━━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"💰 السعر: `{price}`\n\n"
    msg += f"🤖 *بوت 1 (AI):* `{s['ai_score']}/100` – {s['ai_grade']}\n"
    msg += f"📊 *بوت 2 (سيولة):* imbalance `{s['imbalance']:+.1f}%`"
    if s.get("spoofing"):
        msg += " ⚠️ Spoofing"
    msg += "\n"
    for w in s.get("walls_b", [])[:1]:
        msg += f"     🟢 جدار شراء @ `{w[0]}` حجم `{w[1]:,.0f}`\n"
    for w in s.get("walls_a", [])[:1]:
        msg += f"     🔴 جدار بيع @ `{w[0]}` حجم `{w[1]:,.0f}`\n"
    msg += f"🔮 *بوت 3 (توقع):* `{pct:+.2f}%` ({tf}h) | ثقة `{conf}` | {grade}\n"
    bias_txt = {"LEAN_LONG": "📈 تفوق متوقع مقابل السوق",
                "FADE_PUMP": "📉 تأخر متوقع مقابل السوق (مضخّمة)",
                "NEUTRAL": "⚖️ محايد"}.get(p.get("relative_bias", ""), "")
    if bias_txt:
        msg += f"   📐 القيمة النسبية: {bias_txt}\n"
    meta_prob = s.get("meta_labeling_prob")
    if meta_prob is not None:
        reliable = s.get("meta_labeling_reliable", False)
        msg += f"   🔬 Meta: `{meta_prob:.0%}` {'✅' if reliable else '❌'}\n"
    market_sent = p.get("market_sentiment")
    if market_sent:
        msg += f"   🌐 سوق: {market_sent}"
        fr = p.get("funding_rate_pct", 0)
        if fr: msg += f" | Funding: `{fr:+.4f}%`"
        ls = p.get("long_short_ratio", 0)
        if ls: msg += f" | L/S: `{ls:.2f}`"
        msg += "\n"
    risk = p.get("risk", {})
    if risk:
        sl_val = risk.get("suggested_stop_loss_pct", 0)
        tp_val = risk.get("suggested_take_profit_pct", 0)
        pos = risk.get("suggested_position_size_pct", 0)
        rr = risk.get("risk_reward_ratio", 0)
        msg += f"   🎯 SL: `{sl_val}%` TP: `{tp_val}%` (R:R `{rr}`) حجم `{pos}%`\n"
    msg += "\n"
    msg += f"⚡ *القرار: {direction}*\n"
    msg += f"🎯 الدخول: `{entry}`\n"
    msg += f"🛑 وقف: `{sl}`\n"
    msg += f"✅ الهدف: `{tp1}`\n"
    decision = s.get("decision", {})
    msg += f"⭐ قرار الماستر: `{decision.get('decision', 'N/A')}` | درجة `{decision.get('score', 0)}/100`\n"
    if ctx:
        btc = ctx.get("btc", {})
        msg += f"\n฿ BTC: `${btc.get('btc_price', '?')}` ({btc.get('btc_change_24h', 0):+.2f}%)"
    msg += f"\n\n⏰ {fmt_baghdad()} (بغداد)"
    return msg

# ─── STARTUP ────────────────────────────────────────────────────
def send_startup_report():
    # Disabled per user request (no welcome/online messages)
    return


# ─── MAIN ───────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║       StratoCrypto v3.0 – Tri-Bot System            ║")
    logger.info("╚══════════════════════════════════════════════════════╝")
    logger.info(f"  Bot 1 (Signals)   : {'✅' if BOT1_TOKEN else '❌'}")
    logger.info(f"  Bot 2 (Liquidity) : {'✅' if BOT2_TOKEN else '❌'}")
    logger.info(f"  Bot 3 (Predict)   : {'✅' if BOT3_TOKEN else '❌'}")
    logger.info(f"  Bot 4 (Master)    : {'✅' if BOT4_TOKEN else '❌'}")
    logger.info(f"  Order Book        : {ORDER_BOOK_LEVELS} levels")
    logger.info(f"  Min Volume        : ${MIN_24H_VOLUME_USDT/1e6:.0f}M")
    logger.info(f"  Scan Interval     : {SCAN_INTERVAL_MINUTES} min")
    logger.info(f"  Timezone          : Asia/Baghdad (UTC+3)")

    last_daily_report_date = now_baghdad().strftime("%Y-%m-%d")
    # Train ML ensemble if needed (background thread — don't block startup)
    try:
        import os
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ensemble_v2.pkl")
        if not os.path.exists(model_path):
            logger.info("[ML] No ensemble found → training in background (MLP+RF+GBR+XGB+LR)...")
            import threading
            def _train_bg():
                try:
                    from ml_trainer import train_all
                    train_all()
                    import datetime
                    global last_training_date
                    last_training_date = datetime.datetime.now().isoformat()
                except Exception as e:
                    logger.warning(f"[ML] Background training failed: {e}")
            threading.Thread(target=_train_bg, daemon=True).start()
        else:
            logger.info("[ML] Multi-model ensemble loaded (MLP+RF+GBR+XGB+LR)")
    except Exception as e:
        logger.warning(f"[ML] Ensemble init skipped: {e}")
    send_startup_report()
    # Run all scans immediately (lite mode for speed)
    run_full_scan(lite_mode=True)
    if BOT3_TOKEN:
        try:
            logger.info("[MAIN] Initial prediction scan...")
            run_prediction_scan()
            logger.info("[MAIN] Initial prediction scan done")
        except Exception as e:
            logger.error(f"Prediction scan failed: {e}")
    if BOT4_TOKEN:
        try:
            logger.info("[MAIN] Initial master scan...")
            run_master_scan()
            logger.info("[MAIN] Initial master scan done")
        except Exception as e:
            logger.error(f"Master scan failed: {e}")

    while True:
        logger.info(f"[MAIN] Sleeping {SCAN_INTERVAL_MINUTES} min until next scan...")
        time.sleep(SCAN_INTERVAL_MINUTES * 60)
        if not monitoring_active:
            continue
        try:
            logger.info("[MAIN] Starting scheduled scan cycle")
            # Clear ML feature cache for fresh data every scan
            try:
                from ml_trainer import clear_feature_cache
                clear_feature_cache()
                logger.info("[MAIN] Feature cache cleared")
            except Exception as e:
                logger.error(f"[MAIN] Feature cache clear failed: {e}")
            try:
                logger.info("[MAIN] Running pairs scan...")
                run_pairs_scan()
                logger.info("[MAIN] Pairs scan done")
            except Exception as e:
                logger.error(f"[MAIN] Pairs scan failed: {e}")
            try:
                logger.info("[MAIN] Running full scan...")
                run_full_scan(lite_mode=True)
                logger.info("[MAIN] Full scan done")
            except Exception as e:
                logger.error(f"[MAIN] Full scan failed: {e}")
            if BOT3_TOKEN:
                try:
                    logger.info("[MAIN] Running prediction scan...")
                    run_prediction_scan()
                    logger.info("[MAIN] Prediction scan done")
                except Exception as e:
                    logger.error(f"[MAIN] Prediction scan failed: {e}")
            if BOT4_TOKEN:
                try:
                    logger.info("[MAIN] Running master scan...")
                    run_master_scan()
                    logger.info("[MAIN] Master scan done")
                except Exception as e:
                    logger.error(f"[MAIN] Master scan failed: {e}")
            logger.info("[MAIN] Scheduled scan cycle complete")
        except Exception as e:
            logger.error(f"Main loop error: {e}")

