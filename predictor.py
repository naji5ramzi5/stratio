"""
StratoCrypto Prediction Engine v3.0 – Accuracy-Focused
Multi-factor ensemble with statistical confidence, divergence detection,
volume confirmation, and market regime awareness.
"""

import numpy as np
from sklearn.linear_model import LinearRegression


def _lazy(name):
    import importlib
    return getattr(importlib.import_module("advanced_bot"), name)


def _linreg(closes, n_steps):
    if len(closes) < 10:
        return float(closes[-1]) if len(closes) else 0.0
    x = np.arange(len(closes)).reshape(-1, 1)
    m = LinearRegression().fit(x, closes)
    return float(m.predict([[len(closes) + n_steps - 1]])[0])


def _rsi(prices, p=14):
    if len(prices) < p + 1:
        return 50.0
    d = np.diff(prices)
    ag = np.mean(np.where(d > 0, d, 0)[-p:])
    al = np.mean(np.where(d < 0, -d, 0)[-p:])
    if al == 0:
        return 100.0
    return 100 - 100 / (1 + ag / al)


def _divergence(price, rsi_vals):
    """Detect bullish/bearish divergence between price and RSI."""
    if len(price) < 10 or len(rsi_vals) < 10:
        return 0
    p_low = np.argmin(price[-10:])
    r_low = np.argmin(rsi_vals[-10:])
    p_high = np.argmax(price[-10:])
    r_high = np.argmax(rsi_vals[-10:])
    score = 0
    if price[-10 + p_low] < price[-10 + p_low - 1] and rsi_vals[-10 + p_low] > rsi_vals[-10 + p_low - 1]:
        score += 25
    if price[-10 + p_high] > price[-10 + p_high - 1] and rsi_vals[-10 + p_high] < rsi_vals[-10 + p_high - 1]:
        score -= 25
    return score


def _confluence_weighted(preds, weights):
    """Weighted average of multiple predictions with agreement bonus."""
    if not preds:
        return 0, 0
    weighted = sum(p * w for p, w in zip(preds, weights)) / sum(weights)
    # Agreement bonus: if 60%+ agree on direction, boost confidence
    positive = sum(1 for p in preds if p > 0)
    total = len(preds)
    agreement = max(positive, total - positive) / total
    bonus = (agreement - 0.5) * 2  # 0 to 1 scale
    return weighted, bonus


def predict_price_movement(symbol, tf_hours):
    get_klines = _lazy("get_klines")
    get_ta = _lazy("get_technical_analysis")
    get_flow = _lazy("analyze_order_flow")
    logger = _lazy("logger")
    log = logger.info

    try:
        # ── Data collection ────────────────────────────────
        if tf_hours <= 2:
            interval, lookback, n_fwd = "5m", 100, 24
        elif tf_hours <= 6:
            interval, lookback, n_fwd = "15m", 100, 24
        elif tf_hours <= 8:
            interval, lookback, n_fwd = "30m", 100, 16
        else:
            interval, lookback, n_fwd = "1h", 200, 24

        klines = get_klines(symbol, interval, lookback)
        if not klines or len(klines) < 30:
            return {"symbol": symbol, "tf": tf_hours, "error": "no data"}

        c = np.array([float(k[4]) for k in klines])  # close
        h = np.array([float(k[2]) for k in klines])  # high
        l = np.array([float(k[3]) for k in klines])  # low
        v = np.array([float(k[5]) for k in klines])  # volume
        price = c[-1]

        # ── 1. LINEAR REGRESSION ──────────────────────────
        lr_price = _linreg(c, n_fwd)
        lr_pct = ((lr_price - price) / price) * 100

        # ── 2. MOMENTUM (ROC) ─────────────────────────────
        roc_3 = (c[-1] / c[-4] - 1) * 100 if c[-4] else 0
        roc_5 = (c[-1] / c[-6] - 1) * 100 if c[-6] else 0
        roc_10 = (c[-1] / c[-11] - 1) * 100 if c[-11] else 0
        mom_score = roc_3 * 0.5 + roc_5 * 0.3 + roc_10 * 0.2

        # ── 3. VOLUME TREND ───────────────────────────────
        v_avg_5 = np.mean(v[-5:]) if len(v) >= 5 else np.mean(v)
        v_avg_20 = np.mean(v[-20:]) if len(v) >= 20 else np.mean(v)
        vol_ratio = v_avg_5 / v_avg_20 if v_avg_20 > 0 else 1
        vol_trend_score = (vol_ratio - 1) * 100

        # ── 4. ATR VOLATILITY PROJECTION ──────────────────
        trs = [max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1])) for i in range(1, len(klines))]
        atr = np.mean(trs[-14:]) if len(trs) >= 14 else np.std(c[-20:]) * 0.5 if len(c) >= 20 else price * 0.01
        atr_pct = (atr / price) * 100
        atr_direction = 1 if lr_pct > 0 else -1
        atr_proj = atr_pct * (n_fwd / 24) ** 0.5 * atr_direction

        # ── 5. TECHNICAL CONFIRMATION ─────────────────────
        ta = get_ta(symbol, interval)
        rsi_val = ta.get("rsi", 50)
        macd_t = ta.get("macd", {}).get("trend", "Neutral")
        ema20 = ta.get("ema20", price)
        ema50 = ta.get("ema50", price * 0.98)
        # RSI divergence
        rsi_series = np.array([_rsi(c[:i+1]) for i in range(max(0, len(c)-20), len(c))])
        div_score = _divergence(c[-20:], rsi_series[-20:]) if len(rsi_series) >= 10 else 0
        tech_score = (rsi_val - 50) * 0.8 + div_score
        if "Bullish" in macd_t:
            tech_score += 15
        elif "Bearish" in macd_t:
            tech_score -= 15
        # EMA position
        if price > ema20 > ema50:
            tech_score += 15
        elif price < ema20 < ema50:
            tech_score -= 15

        # ── 6. ORDER FLOW ─────────────────────────────────
        flow = get_flow(symbol)
        buy_pct = flow.get("buy_pct", 50) if flow else 50
        whale_buy = flow.get("whale_buy_vol", 0) if flow else 0
        whale_sell = flow.get("whale_sell_vol", 0) if flow else 0
        flow_score = (buy_pct - 50) * 2
        if whale_buy > whale_sell and whale_buy > 0:
            flow_score += 10

        # ── 7. SUPPORT / RESISTANCE ───────────────────────
        pivots_h, pivots_l = [], []
        for i in range(2, len(h) - 2):
            if h[i] > h[i-1] and h[i] > h[i+1]:
                pivots_h.append(h[i])
            if l[i] < l[i-1] and l[i] < l[i+1]:
                pivots_l.append(l[i])
        near_res = min([r for r in pivots_h if r > price * 0.98], default=price * 1.08)
        near_sup = max([s for s in pivots_l if s < price * 1.02], default=price * 0.92)
        dist_res = ((near_res - price) / price) * 100
        dist_sup = ((price - near_sup) / price) * 100
        room_up = dist_res > 3
        room_down = dist_sup > 3

        # ── ENSEMBLE ──────────────────────────────────────
        methods = []
        weights = []
        # LR (weight 30)
        methods.append(lr_pct)
        weights.append(30)
        # Momentum (weight 20)
        methods.append(mom_score)
        weights.append(20)
        # ATR (weight 15)
        methods.append(atr_proj)
        weights.append(15)
        # Technical (weight 20)
        methods.append(tech_score * 0.3)
        weights.append(20)
        # Flow (weight 15)
        methods.append(flow_score * 0.3)
        weights.append(15)

        pred_change, agree_bonus = _confluence_weighted(methods, weights)
        if abs(pred_change) > 1:
            log(f"  📊 {symbol} {tf_hours}h: lr={lr_pct:.1f} mom={mom_score:.1f} atr={atr_proj:.1f} tech={tech_score:.0f} flow={flow_score:.0f} => pred={pred_change:.1f}%")

        # ── ACCURACY FILTERS ──────────────────────────────
        reasons = []
        filters_passed = 0

        # Filter 1: Trend alignment
        if (lr_pct > 0 and mom_score > 0) or (lr_pct < 0 and mom_score < 0):
            filters_passed += 1
        elif abs(lr_pct) > 2 or abs(mom_score) > 2:
            filters_passed += 0.5
        else:
            reasons.append("lr/momentum disagree")

        # Filter 2: Volume confirmation
        if vol_ratio > 1.1:
            filters_passed += 1
        elif vol_ratio < 0.8:
            reasons.append("volume declining")
        else:
            filters_passed += 0.5

        # Filter 3: No extreme RSI (>85 or <15 means unreliable)
        if 15 < rsi_val < 85:
            filters_passed += 1
        else:
            reasons.append(f"rsi={rsi_val:.0f} extreme")

        # Filter 4: Room to move
        if (pred_change > 0 and room_up) or (pred_change < 0 and room_down) or (dist_res > 1.5 and dist_sup > 1.5):
            filters_passed += 1
        elif pred_change > 0 and not room_up:
            reasons.append(f"resistance {dist_res:.1f}%")

        # Filter 5: Order flow confirmation
        if (pred_change > 0 and buy_pct > 52) or (pred_change < 0 and buy_pct < 48):
            filters_passed += 1
        else:
            reasons.append("flow neutral")

        # ── CONFIDENCE ────────────────────────────────────
        raw_conf = 50 + filters_passed * 8 + agree_bonus * 5
        if abs(pred_change) < 1:
            raw_conf -= 15
        elif abs(pred_change) > 5:
            raw_conf += 5
        if len(c) >= 100:
            raw_conf += 5
        if atr_pct > 5:
            raw_conf -= 8
        confidence = max(15, min(92, raw_conf))

        # ── Confidence adjustment based on filters ─────────
        if filters_passed < 1.5:
            confidence = min(confidence, 35)
        elif filters_passed < 2.5:
            confidence = min(confidence, 65)

        predicted_price = price * (1 + pred_change / 100)

        # ── GRADE ─────────────────────────────────────────
        if pred_change >= 5.0:   grade = "⭐⭐⭐ STRONG UP"
        elif pred_change >= 3.0: grade = "⭐⭐ MODERATE UP"
        elif pred_change >= 1.5: grade = "⭐ SLIGHT UP"
        elif pred_change >= -1.5: grade = "➡️ NEUTRAL"
        elif pred_change >= -3.0: grade = "⭐ SLIGHT DOWN"
        elif pred_change >= -5.0: grade = "⭐⭐ MODERATE DOWN"
        else: grade = "⭐⭐⭐ STRONG DOWN"

        # ── ENTRY TIMING ──────────────────────────────────
        if rsi_val and rsi_val > 75:
            timing = "Wait pullback ⏳"
        elif rsi_val and rsi_val < 25:
            timing = "Oversold entry 🟢"
        elif dist_res < 1.5 and pred_change > 0:
            timing = "Breakout needed 🔄"
        elif dist_sup < 1 and pred_change < 0:
            timing = "Breakdown risk 🔴"
        elif vol_ratio > 1.3:
            timing = "Strong vol – now 🟢"
        else:
            timing = "Immediate 🟢"

        return {
            "symbol": symbol, "timeframe_hours": tf_hours,
            "current_price": round(price, 8),
            "predicted_price": round(predicted_price, 8),
            "predicted_change_pct": round(pred_change, 2),
            "confidence": int(confidence),
            "grade": grade,
            "entry_timing": timing,
            "filters_passed": int(filters_passed),
            "rsi": round(rsi_val, 1) if rsi_val else 50,
            "macd_trend": macd_trend,
            "buy_pressure_pct": round(buy_pct, 1),
            "volume_ratio": round(vol_ratio, 2),
            "atr_pct": round(atr_pct, 2),
            "nearest_support": round(near_sup, 8),
            "nearest_resistance": round(near_res, 8),
            "divergence_score": round(div_score, 1),
            "reasons": reasons[:3],
        }

    except Exception as e:
        return {"symbol": symbol, "timeframe_hours": tf_hours, "error": str(e)}
