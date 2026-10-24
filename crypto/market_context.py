"""
StratoCrypto Market Context
BTC dominance, Fear & Greed Index, market regime classification.
"""

import requests
import numpy as np


def _lazy_import(name):
    import importlib
    mod = importlib.import_module("advanced_bot")
    return getattr(mod, name)


def get_btc_dominance() -> dict:
    get_binance = _lazy_import("get_binance")
    try:
        data = get_binance("/api/v3/ticker/24hr")
        if not isinstance(data, list):
            return {"btc_dominance": 0, "btc_price": 0, "btc_change_24h": 0}
        btc_vol = total_vol = btc_price = 0
        btc_change = 0.0
        for t in data:
            sym = t.get("symbol", "")
            vol = float(t.get("quoteVolume", 0) or 0)
            if sym == "BTCUSDT":
                btc_vol = vol
                btc_price = float(t.get("lastPrice", 0))
                btc_change = float(t.get("priceChangePercent", 0))
            if sym.endswith("USDT"):
                total_vol += vol
        dominance = (btc_vol / total_vol * 100) if total_vol > 0 else 0
        return {
            "btc_dominance": round(dominance, 2),
            "btc_price": btc_price,
            "btc_change_24h": round(btc_change, 2)
        }
    except Exception:
        return {"btc_dominance": 0, "btc_price": 0, "btc_change_24h": 0}


def get_fear_greed_index() -> dict:
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
        if resp.status_code == 200:
            item = resp.json().get("data", [{}])[0]
            return {
                "value": int(item.get("value", 50)),
                "classification": item.get("value_classification", "Neutral")
            }
    except Exception:
        pass
    return {"value": 50, "classification": "Neutral"}


def classify_market_regime(btc_klines_1d: list) -> str:
    if len(btc_klines_1d) < 50:
        return "Insufficient Data"
    closes = np.array([float(k[4]) for k in btc_klines_1d])
    returns = np.diff(closes) / closes[:-1] * 100
    volatility = np.std(returns[-20:])
    avg_return = np.mean(returns[-20:])
    if volatility > 4:
        return "High Volatility ⚡"
    elif avg_return > 0.5:
        return "Trending Up 🟢"
    elif avg_return < -0.5:
        return "Trending Down 🔴"
    else:
        return "Ranging ➡️"


def get_full_market_context() -> dict:
    get_klines = _lazy_import("get_klines")
    btc_dom = get_btc_dominance()
    fng = get_fear_greed_index()
    btc_klines = get_klines("BTCUSDT", "1d", 100)
    regime = classify_market_regime(btc_klines) if btc_klines else "Unknown"
    return {
        "btc": btc_dom,
        "fear_greed": fng,
        "regime": regime,
    }
