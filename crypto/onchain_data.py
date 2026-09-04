"""
On-Chain Data Integration v1.0
مصادر: WhaleAlert, CoinGecko, Blockchain.com, mempool.space
"""
import requests
import logging
from datetime import datetime, timedelta
import json
import os

logger = logging.getLogger(__name__)


def get_whale_alerts(min_value_usd=500000):
    try:
        resp = requests.get(
            "https://api.whale-alert.io/v1/transactions",
            params={"api_key": os.getenv("WHALEALERT_API_KEY", ""), "min_value": min_value_usd},
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            txns = data.get("transactions", [])
            result = []
            for t in txns[:20]:
                result.append({
                    "symbol": t.get("symbol", "unknown"),
                    "amount": t.get("amount", 0),
                    "amount_usd": t.get("amount_usd", 0),
                    "from": t.get("from", {}).get("address", "unknown")[:12],
                    "to": t.get("to", {}).get("address", "unknown")[:12],
                    "hash": t.get("hash", "")[:16],
                    "time": datetime.fromtimestamp(t.get("timestamp", 0)).isoformat() if t.get("timestamp") else "",
                })
            return result
    except Exception as e:
        logger.debug(f"[ONCHAIN] WhaleAlert: {e}")
    return []


def get_exchange_netflow(asset="bitcoin", days=7):
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{asset}/market_chart"
        resp = requests.get(url, params={"vs_currency": "usd", "days": days}, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            prices = data.get("prices", [])
            if prices:
                return {
                    "current_price": prices[-1][1] if prices else 0,
                    "price_change_24h_pct": round((prices[-1][1] / prices[-2][1] - 1) * 100, 2) if len(prices) >= 2 else 0,
                    "high_24h": max(p[1] for p in prices[-48:]) if len(prices) >= 48 else 0,
                    "low_24h": min(p[1] for p in prices[-48:]) if len(prices) >= 48 else 0,
                }
    except Exception as e:
        logger.debug(f"[ONCHAIN] CoinGecko: {e}")
    return {}


def get_mempool_fees():
    try:
        resp = requests.get("https://mempool.space/api/v1/fees/recommended", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "fastest": data.get("fastestFee", 0),
                "half_hour": data.get("halfHourFee", 0),
                "hour": data.get("hourFee", 0),
                "minimum": data.get("minimumFee", 0),
            }
    except Exception as e:
        logger.debug(f"[ONCHAIN] Mempool: {e}")
    return {}


def get_btc_dominance_and_metrics():
    try:
        resp = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        if resp.status_code == 200:
            data = resp.json().get("data", {})
            return {
                "btc_dominance_pct": round(data.get("market_cap_percentage", {}).get("btc", 0), 1),
                "eth_dominance_pct": round(data.get("market_cap_percentage", {}).get("eth", 0), 1),
                "total_market_cap": round(data.get("total_market_cap", {}).get("usd", 0) / 1e12, 2),
                "total_volume_24h": round(data.get("total_volume", {}).get("usd", 0) / 1e9, 2),
                "active_cryptocurrencies": data.get("active_cryptocurrencies", 0),
            }
    except Exception as e:
        logger.debug(f"[ONCHAIN] CoinGecko Global: {e}")
    return {}


def get_aggregated_onchain_signal(symbol):
    signals = []
    signal_value = 0

    whales = get_whale_alerts()
    btc_whales = [w for w in whales if "BTC" in w.get("symbol", "").upper()]
    if btc_whales:
        total_whale_vol = sum(w.get("amount_usd", 0) for w in btc_whales)
        if total_whale_vol > 50_000_000:
            signals.append(("large_whale_movement", -0.3))
            signal_value -= 0.2

    btc_data = get_exchange_netflow("bitcoin")
    if btc_data.get("price_change_24h_pct", 0) < -3:
        signals.append(("btc_drop", -0.4))
        signal_value -= 0.3
    elif btc_data.get("price_change_24h_pct", 0) > 3:
        signals.append(("btc_surge", 0.3))
        signal_value += 0.2

    dominance = get_btc_dominance_and_metrics()
    if dominance:
        if dominance.get("btc_dominance_pct", 50) > 55:
            signals.append(("high_btc_dominance", -0.2))
            signal_value -= 0.15
        elif dominance.get("btc_dominance_pct", 50) < 40:
            signals.append(("low_btc_dominance", 0.3))
            signal_value += 0.2

    signal_value = max(-1, min(1, signal_value))
    if signal_value > 0.3:
        sentiment = "bullish"
    elif signal_value < -0.3:
        sentiment = "bearish"
    else:
        sentiment = "neutral"

    mempool = get_mempool_fees()

    return {
        "sentiment": sentiment,
        "signal_value": round(signal_value, 3),
        "whale_alerts_24h": len(whales),
        "btc_data": btc_data,
        "dominance": dominance,
        "mempool_fees": mempool,
        "signals": signals,
    }