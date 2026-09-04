"""
Crypto-specific market data: Funding Rate, Open Interest, Long/Short Ratio
These are critical features missing from the current pipeline.
"""
import requests
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

BINANCE_FAPI = "https://fapi.binance.com"


def _req(url, params=None, timeout=5):
    try:
        r = requests.get(url, params=params, timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        logger.debug(f"Request failed: {url} - {e}")
        return None


def get_funding_rate(symbol, limit=50):
    data = _req(f"{BINANCE_FAPI}/fapi/v1/fundingRate",
                {"symbol": symbol, "limit": limit})
    if not isinstance(data, list) or len(data) == 0:
        return {"current": 0, "mean_7d": 0, "mean_3d": 0, "trend": "neutral", "signal": 0}

    rates = [float(d["fundingRate"]) for d in data]
    current = rates[-1] if rates else 0
    mean_3d = sum(rates[-9:]) / max(len(rates[-9:]), 1) if len(rates) >= 9 else current
    mean_7d = sum(rates[-21:]) / max(len(rates[-21:]), 1) if len(rates) >= 21 else current

    if current > 0.001:
        trend = "expensive_long"
    elif current < -0.001:
        trend = "expensive_short"
    elif abs(current) < 0.0001:
        trend = "neutral"
    else:
        trend = "slight_imbalance"

    signal = 0
    if current > 0.0005 and mean_7d > 0.0003:
        signal = -1
    elif current < -0.0005 and mean_7d < -0.0003:
        signal = 1
    elif current > mean_7d * 1.5 and current > 0:
        signal = -0.5
    elif current < mean_7d * 1.5 and current < 0:
        signal = 0.5

    return {
        "current": round(current * 100, 4),
        "mean_3d_pct": round(mean_3d * 100, 4),
        "mean_7d_pct": round(mean_7d * 100, 4),
        "trend": trend,
        "signal": signal,
    }


def get_open_interest(symbol, limit=50):
    data = _req(f"{BINANCE_FAPI}/fapi/v1/openInterestHist",
                {"symbol": symbol, "period": "1h", "limit": limit})
    if not isinstance(data, list) or len(data) < 2:
        return {"current": 0, "change_1h_pct": 0, "change_24h_pct": 0, "trend": "neutral"}

    oi_values = [float(d["sumOpenInterest"]) for d in data]
    oi_usd = [float(d["sumOpenInterestValue"]) for d in data]
    current = oi_values[-1]
    oi_1h_ago = oi_values[-2] if len(oi_values) >= 2 else current
    oi_24h_ago = oi_values[0] if oi_values else current

    change_1h = ((current - oi_1h_ago) / max(oi_1h_ago, 1)) * 100
    change_24h = ((current - oi_24h_ago) / max(oi_24h_ago, 1)) * 100

    if change_1h > 5 and change_24h > 10:
        trend = "rising_sharply"
    elif change_1h > 2:
        trend = "rising"
    elif change_1h < -5 and change_24h < -10:
        trend = "falling_sharply"
    elif change_1h < -2:
        trend = "falling"
    else:
        trend = "stable"

    return {
        "current_usd": round(oi_usd[-1]) if oi_usd else 0,
        "current_contracts": round(current),
        "change_1h_pct": round(change_1h, 2),
        "change_24h_pct": round(change_24h, 2),
        "trend": trend,
    }


def get_long_short_ratio(symbol, limit=30):
    data = _req(f"{BINANCE_FAPI}/futures/data/globalLongShortAccountRatio",
                {"symbol": symbol, "period": "1h", "limit": limit})
    if not isinstance(data, list) or len(data) == 0:
        return {"long_short_ratio": 1.0, "long_pct": 50, "short_pct": 50}

    ratios = [float(d["longShortRatio"]) for d in data]
    long_pcts = [float(d["longAccount"]) for d in data]

    current_ratio = ratios[-1] if ratios else 1.0
    current_long = long_pcts[-1] if long_pcts else 50
    mean_ratio = sum(ratios) / len(ratios) if ratios else 1.0

    if current_ratio > mean_ratio * 1.2 and current_ratio > 1.2:
        sentiment = "crowded_long"
    elif current_ratio < mean_ratio * 0.8 and current_ratio < 0.8:
        sentiment = "crowded_short"
    else:
        sentiment = "balanced"

    return {
        "long_short_ratio": round(current_ratio, 3),
        "long_pct": round(current_long, 1),
        "short_pct": round(100 - current_long, 1),
        "sentiment": sentiment,
        "deviation_from_mean": round((current_ratio - mean_ratio) / mean_ratio * 100, 1),
    }


def get_aggregated_sentiment(symbol):
    fr = get_funding_rate(symbol)
    oi = get_open_interest(symbol)
    ls = get_long_short_ratio(symbol)

    signals = []
    signal_value = 0

    if fr.get("signal", 0) != 0:
        signals.append(("funding", fr["signal"]))
        signal_value += fr["signal"] * 0.4

    if ls.get("long_short_ratio", 1) > 1.3:
        signals.append(("long_short", -0.5))
        signal_value -= 0.2
    elif ls.get("long_short_ratio", 1) < 0.7:
        signals.append(("long_short", 0.5))
        signal_value += 0.2

    if oi.get("change_1h_pct", 0) > 3 and oi.get("trend") == "rising":
        signals.append(("oi_growth", 0.3))
        signal_value += 0.15
    elif oi.get("change_1h_pct", 0) < -3:
        signals.append(("oi_decline", -0.3))
        signal_value -= 0.15

    signal_value = max(-1, min(1, signal_value))

    if signal_value > 0.3:
        sentiment = "bullish"
    elif signal_value < -0.3:
        sentiment = "bearish"
    else:
        sentiment = "neutral"

    return {
        "sentiment": sentiment,
        "signal_value": round(signal_value, 3),
        "components": {
            "funding_rate": fr,
            "open_interest": oi,
            "long_short_ratio": ls,
        },
        "signals": signals,
    }