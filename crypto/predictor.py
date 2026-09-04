"""
StratoCrypto Prediction Engine v4.0 – ML-Augmented
Multi-factor ensemble with ML model integration, historical accuracy calibration,
divergence detection, volume confirmation, and market regime awareness.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN
from datetime import datetime
import math
from prediction_tracker import prediction_tracker
from risk_metrics import compute_risk_metrics
from meta_labeling import meta_labeler
from crypto_market_data import get_aggregated_sentiment as get_market_sentiment
from onchain_data import get_aggregated_onchain_signal
from multi_exchange import get_exchange_price

PREDICTOR_VERSION = "4.0.1"


def _lazy(name):
    import importlib
    return getattr(importlib.import_module("advanced_bot"), name)


def _safe(val):
    """Convert NaN/Inf to 0 to prevent propagation."""
    try:
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return 0.0
        return v
    except (TypeError, ValueError):
        return 0.0


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
    rsi = 100 - 100 / (1 + ag / al)
    if np.isnan(rsi) or np.isinf(rsi):
        return 50.0
    return rsi


def _divergence(price, rsi_vals):
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
    if not preds:
        return 0, 0
    weighted = sum(p * w for p, w in zip(preds, weights)) / sum(weights)
    positive = sum(1 for p in preds if p > 0)
    total = len(preds)
    agreement = max(positive, total - positive) / total
    bonus = (agreement - 0.5) * 2
    return weighted, bonus


def _regime_detection(c, v):
    regime = "neutral"
    score = 0
    if len(c) >= 50:
        sma20 = np.mean(c[-20:])
        sma50 = np.mean(c[-50:])
        if sma20 > sma50 * 1.05:
            regime = "bullish"
            score = 1
        elif sma20 < sma50 * 0.95:
            regime = "bearish"
            score = -1
    if len(v) >= 20:
        v_ratio = np.mean(v[-5:]) / (np.mean(v[-20:]) + 1e-10)
        if v_ratio > 1.5:
            if regime == "bullish":
                regime = "strong_bullish"
                score = 2
            elif regime == "bearish":
                regime = "strong_bearish"
                score = -2
    return regime, score


def _ml_prediction(symbol, tf_hours):
    try:
        from ml_trainer import predict_all, save_training_sample
        ml_result, feature_vec = predict_all(symbol, tf_hours)
        if ml_result is not None:
            return ml_result, feature_vec
    except ImportError:
        pass
    except Exception:
        pass
    return None, None


def _foundation_prediction(closes, n_steps):
    """Chronos-2 + TimesFM zero-shot forecast. Returns (result_dict|None)."""
    try:
        from foundation_models import foundation_forecast
        return foundation_forecast(closes, n_steps)
    except Exception:
        return None


def _cluster_support_resistance(h, l, price):
    pivots_h, pivots_l = [], []
    for i in range(2, len(h) - 2):
        if h[i] > h[i-1] and h[i] > h[i+1]:
            pivots_h.append(float(h[i]))
        if l[i] < l[i-1] and l[i] < l[i+1]:
            pivots_l.append(float(l[i]))
    if len(pivots_h) > 3:
        pivots_h = np.array(pivots_h).reshape(-1, 1)
        cluster_h = DBSCAN(eps=price * 0.02, min_samples=2).fit(pivots_h)
        levels_h = sorted([np.mean(pivots_h[cluster_h.labels_ == i]) for i in set(cluster_h.labels_) if i >= 0] +
                          [float(p) for p, l in zip(pivots_h.flatten(), cluster_h.labels_) if l < 0])
    else:
        levels_h = sorted(set(pivots_h))
    if len(pivots_l) > 3:
        pivots_l = np.array(pivots_l).reshape(-1, 1)
        cluster_l = DBSCAN(eps=price * 0.02, min_samples=2).fit(pivots_l)
        levels_l = sorted([np.mean(pivots_l[cluster_l.labels_ == i]) for i in set(cluster_l.labels_) if i >= 0] +
                          [float(p) for p, l in zip(pivots_l.flatten(), cluster_l.labels_) if l < 0])
    else:
        levels_l = sorted(set(pivots_l))
    near_res = min([r for r in levels_h if r > price * 0.98], default=price * 1.08)
    near_sup = max([s for s in levels_l if s < price * 1.02], default=price * 0.92)
    return near_res, near_sup


def predict_price_movement(symbol, tf_hours):
    get_klines = _lazy("get_klines")
    get_ta = _lazy("get_technical_analysis")
    get_flow = _lazy("analyze_order_flow")
    logger = _lazy("logger")
    log = logger.info

    try:
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

        c = np.array([float(k[4]) for k in klines])
        h = np.array([float(k[2]) for k in klines])
        l = np.array([float(k[3]) for k in klines])
        v = np.array([float(k[5]) for k in klines])
        price = c[-1]

        lr_price = _linreg(c, n_fwd)
        lr_pct = ((lr_price - price) / price) * 100 if price > 0 else 0

        roc_3 = (c[-1] / c[-4] - 1) * 100 if c[-4] else 0
        roc_5 = (c[-1] / c[-6] - 1) * 100 if c[-6] else 0
        roc_10 = (c[-1] / c[-11] - 1) * 100 if c[-11] else 0
        mom_score = roc_3 * 0.5 + roc_5 * 0.3 + roc_10 * 0.2

        v_avg_5 = np.mean(v[-5:]) if len(v) >= 5 else np.mean(v)
        v_avg_20 = np.mean(v[-20:]) if len(v) >= 20 else np.mean(v)
        vol_ratio = v_avg_5 / v_avg_20 if v_avg_20 > 0 else 1

        trs = [max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1])) for i in range(1, len(klines))]
        atr = np.mean(trs[-14:]) if len(trs) >= 14 else np.std(c[-20:]) * 0.5 if len(c) >= 20 else price * 0.01
        atr_pct = (atr / price) * 100
        atr_direction = 1 if lr_pct > 0 else -1
        atr_proj = atr_pct * (n_fwd / 24) ** 0.5 * atr_direction

        ta = get_ta(symbol, interval)
        rsi_val = ta.get("rsi", 50)
        macd_t = ta.get("macd", {}).get("trend", "Neutral")
        ema20 = ta.get("ema20", price)
        ema50 = ta.get("ema50", price * 0.98)
        rsi_series = np.array([_rsi(c[:i+1]) for i in range(max(0, len(c)-20), len(c))])
        div_score = _divergence(c[-20:], rsi_series[-20:]) if len(rsi_series) >= 10 else 0
        tech_score = (rsi_val - 50) * 0.8 + div_score
        if "Bullish" in macd_t:
            tech_score += 15
        elif "Bearish" in macd_t:
            tech_score -= 15
        if price > ema20 > ema50:
            tech_score += 15
        elif price < ema20 < ema50:
            tech_score -= 15

        flow = get_flow(symbol)
        buy_pct = flow.get("buy_pct", 50) if flow else 50
        whale_buy = flow.get("whale_buy_vol", 0) if flow else 0
        whale_sell = flow.get("whale_sell_vol", 0) if flow else 0
        flow_score = (buy_pct - 50) * 2
        if whale_buy > whale_sell and whale_buy > 0:
            flow_score += 10

        near_res, near_sup = _cluster_support_resistance(h, l, price)
        dist_res = ((near_res - price) / price) * 100
        dist_sup = ((price - near_sup) / price) * 100
        room_up = dist_res > 3
        room_down = dist_sup > 3

        regime, regime_score = _regime_detection(c, v)

        ml_result, feature_vec = _ml_prediction(symbol, tf_hours)
        ml_pct = _safe(ml_result["predicted_change_pct"]) if ml_result else None
        ml_conf = ml_result["confidence"] if ml_result else None

        foundation = _foundation_prediction(c, n_fwd)
        fnd_pct = foundation["change_pct"] if foundation else None
        fnd_conf = foundation["confidence"] if foundation else None

        methods = []
        weights = []
        methods.append(lr_pct)
        weights.append(30)
        methods.append(mom_score)
        weights.append(20)
        methods.append(atr_proj)
        weights.append(15)
        methods.append(tech_score * 0.3)
        weights.append(20)
        methods.append(flow_score * 0.3)
        weights.append(15)
        if ml_pct is not None:
            methods.append(ml_pct * 5)
            weights.append(25)
        if fnd_pct is not None:
            methods.append(fnd_pct * 5)
            weights.append(25)

        methods = [_safe(m) for m in methods]
        pred_change, agree_bonus = _confluence_weighted(methods, weights)
        if abs(pred_change) > 1:
            log(f"  {symbol} {tf_hours}h: lr={lr_pct:.1f} mom={mom_score:.1f} atr={atr_proj:.1f} tech={tech_score:.0f} flow={flow_score:.0f} ml={ml_pct} fnd={fnd_pct} => pred={pred_change:.1f}%")

        reasons = []
        filters_passed = 0

        if (lr_pct > 0 and mom_score > 0) or (lr_pct < 0 and mom_score < 0):
            filters_passed += 1
        elif abs(lr_pct) > 2 or abs(mom_score) > 2:
            filters_passed += 0.5
        else:
            reasons.append("lr/momentum disagree")

        if ml_pct is not None and abs(ml_pct - pred_change) < 2:
            filters_passed += 0.5
        elif ml_pct is not None:
            reasons.append("ml diverges")

        if fnd_pct is not None and abs(fnd_pct - pred_change) < 2:
            filters_passed += 0.5
        elif fnd_pct is not None:
            reasons.append("foundation diverges")

        if vol_ratio > 1.1:
            filters_passed += 1
        elif vol_ratio < 0.8:
            reasons.append("volume declining")
        else:
            filters_passed += 0.5

        if 15 < rsi_val < 85:
            filters_passed += 1
        else:
            reasons.append(f"rsi={rsi_val:.0f} extreme")

        if (pred_change > 0 and room_up) or (pred_change < 0 and room_down) or (dist_res > 1.5 and dist_sup > 1.5):
            filters_passed += 1
        elif pred_change > 0 and not room_up:
            reasons.append(f"resistance {dist_res:.1f}%")

        if (pred_change > 0 and buy_pct > 52) or (pred_change < 0 and buy_pct < 48):
            filters_passed += 1
        else:
            reasons.append("flow neutral")

        if regime_score != 0 and ((pred_change > 0 and regime_score > 0) or (pred_change < 0 and regime_score < 0)):
            filters_passed += 0.5
        else:
            reasons.append(f"regime={regime}")

        stats = prediction_tracker.get_stats()
        hist_accuracy = stats.get("direction_accuracy_pct", 50) if stats else 50

        raw_conf = 50 + filters_passed * 7 + agree_bonus * 4
        if abs(pred_change) < 1:
            raw_conf -= 15
        elif abs(pred_change) > 5:
            raw_conf += 5
        if len(c) >= 100:
            raw_conf += 5
        if atr_pct > 5:
            raw_conf -= 8
        if ml_conf is not None:
            raw_conf = raw_conf * 0.6 + ml_conf * 0.4
        if fnd_conf is not None:
            raw_conf = raw_conf * 0.7 + fnd_conf * 0.3
        if hist_accuracy > 0:
            acc_factor = (hist_accuracy - 50) / 50
            raw_conf += acc_factor * 10

        confidence = max(15, min(92, raw_conf))
        if filters_passed < 1.5:
            confidence = min(confidence, 35)
        elif filters_passed < 2.5:
            confidence = min(confidence, 65)

        pred_change = _safe(pred_change)
        predicted_price = price * (1 + pred_change / 100)

        # ── Probabilistic view (model-implied, honestly labeled) ────────
        # The ensemble point forecast is treated as the mean of a normal
        # whose std is the EXPECTED move volatility (ATR projection over the
        # horizon + dispersion across the combined methods). P(up/down/
        # neutral) are model-implied and NOT independently calibrated yet —
        # calibration status is reported explicitly.
        method_disp = (float(np.std([m for m in methods]))
                       if len(methods) > 1 else (abs(pred_change) * 0.1 + 0.5))
        expected_vol_pct = max(atr_pct * (n_fwd / 24.0) ** 0.5, method_disp, 0.5)
        _sigma = max(expected_vol_pct, 0.5)
        def _norm_cdf(x, mu, sig):
            return 0.5 * (1.0 + math.erf((x - mu) / (sig * math.sqrt(2.0))))
        _p_up = 1.0 - _norm_cdf(0.0, pred_change, _sigma)
        _p_down = _norm_cdf(0.0, pred_change, _sigma)
        _p_neutral = _norm_cdf(0.5, pred_change, _sigma) - _norm_cdf(-0.5, pred_change, _sigma)
        _s = _p_up + _p_down + _p_neutral
        prob_up = round(_p_up / _s, 3)
        prob_down = round(_p_down / _s, 3)
        prob_neutral = round(_p_neutral / _s, 3)
        calibrated = False

        if pred_change >= 5.0:   grade = "STRONG UP"
        elif pred_change >= 3.0: grade = "MODERATE UP"
        elif pred_change >= 1.5: grade = "SLIGHT UP"
        elif pred_change >= -1.5: grade = "NEUTRAL"
        elif pred_change >= -3.0: grade = "SLIGHT DOWN"
        elif pred_change >= -5.0: grade = "MODERATE DOWN"
        else: grade = "STRONG DOWN"

        if rsi_val and rsi_val > 75:
            timing = "Wait pullback"
        elif rsi_val and rsi_val < 25:
            timing = "Oversold entry"
        elif dist_res < 1.5 and pred_change > 0:
            timing = "Breakout needed"
        elif dist_sup < 1 and pred_change < 0:
            timing = "Breakdown risk"
        elif vol_ratio > 1.3:
            timing = "Strong vol now"
        else:
            timing = "Immediate"

        prediction_result = {
            "symbol": symbol, "timeframe_hours": tf_hours,
            "horizon_hours": tf_hours,
            "created_at": datetime.now().isoformat(),
            "model_version": PREDICTOR_VERSION,
            "current_price": round(price, 8),
            "predicted_price": round(predicted_price, 8),
            "predicted_change_pct": round(pred_change, 2),
            "expected_return_pct": round(pred_change, 2),
            "expected_vol_pct": round(expected_vol_pct, 2),
            "prob_up": prob_up,
            "prob_down": prob_down,
            "prob_neutral": prob_neutral,
            "calibrated": calibrated,
            "confidence": int(confidence),
            "grade": grade,
            "entry_timing": timing,
            "market_regime": regime,
            "filters_passed": int(filters_passed),
            "rsi": round(rsi_val, 1) if rsi_val else 50,
            "macd_trend": macd_t,
            "buy_pressure_pct": round(buy_pct, 1),
            "volume_ratio": round(vol_ratio, 2),
            "atr_pct": round(atr_pct, 2),
            "nearest_support": round(near_sup, 8),
            "nearest_resistance": round(near_res, 8),
            "divergence_score": round(div_score, 1),
            "historical_accuracy_pct": hist_accuracy,
            "ml_prediction": ml_pct,
            "foundation_prediction": fnd_pct,
            "reasons": reasons[:3],
        }

        try:
            risk = compute_risk_metrics(prediction_result, c)
            prediction_result["risk"] = risk
        except Exception:
            prediction_result["risk"] = {}

        try:
            from anomaly_detector import detect_regime_change, volatility_regime
            regime_label, regime_trend = detect_regime_change(c)
            vol_label, vol_pct = volatility_regime(c)
            prediction_result["detailed_regime"] = regime_label
            prediction_result["regime_trend_pct"] = regime_trend
            prediction_result["volatility_regime"] = vol_label
            prediction_result["volatility_annualized_pct"] = vol_pct
        except Exception:
            pass

        try:
            meta_prob, is_reliable = meta_labeler.predict(prediction_result)
            prediction_result["meta_labeling_prob"] = round(meta_prob, 3)
            prediction_result["meta_labeling_reliable"] = is_reliable
            if not is_reliable:
                confidence = int(confidence * 0.6)
                prediction_result["confidence"] = confidence
                prediction_result["reasons"].append(f"meta_unreliable({meta_prob:.2f})")
        except Exception:
            pass

        try:
            onchain = get_aggregated_onchain_signal(symbol)
            prediction_result["onchain_sentiment"] = onchain["sentiment"]
            prediction_result["onchain_signal"] = onchain["signal_value"]
            prediction_result["onchain_whale_alerts"] = onchain["whale_alerts_24h"]
            if onchain["sentiment"] == "bullish" and pred_change > 0:
                confidence = min(95, confidence + 5)
            elif onchain["sentiment"] == "bearish" and pred_change < 0:
                confidence = min(95, confidence + 5)
        except Exception:
            pass

        try:
            exch = get_exchange_price(symbol + "USDT" if not symbol.endswith("USDT") else symbol)
            if exch and exch.get("num_exchanges", 0) > 1:
                prediction_result["multi_exchange_price"] = exch["price"]
                prediction_result["exchanges_count"] = exch["num_exchanges"]
                prediction_result["exchange_spreads"] = exch.get("spread_bps", {})
        except Exception:
            pass

        try:
            market_sent = get_market_sentiment(symbol + "USDT" if not symbol.endswith("USDT") else symbol)
            prediction_result["market_sentiment"] = market_sent["sentiment"]
            prediction_result["market_signal_value"] = market_sent["signal_value"]
            fr = market_sent["components"]["funding_rate"]
            oi = market_sent["components"]["open_interest"]
            ls = market_sent["components"]["long_short_ratio"]
            prediction_result["funding_rate_pct"] = fr["current"]
            prediction_result["funding_trend"] = fr["trend"]
            prediction_result["open_interest_change_1h_pct"] = oi["change_1h_pct"]
            prediction_result["open_interest_trend"] = oi["trend"]
            prediction_result["long_short_ratio"] = ls["long_short_ratio"]
            prediction_result["long_short_sentiment"] = ls["sentiment"]
            if market_sent["sentiment"] == "bullish" and pred_change > 0:
                confidence = min(92, confidence + 8)
            elif market_sent["sentiment"] == "bearish" and pred_change < 0:
                confidence = min(92, confidence + 8)
            elif market_sent["sentiment"] != "neutral":
                confidence = max(15, confidence - 5)
            prediction_result["confidence"] = confidence
        except Exception:
            pass

        # ── Honest confidence anchoring ─────────────────────────────
        # Confidence must reflect the ACTUAL verified track record, not a
        # heuristic filter count. With enough verified samples we anchor to
        # realised direction accuracy; otherwise fall back to calibrated ML
        # confidence and flag as uncalibrated. Low agreement still caps it.
        _n_ver = int((stats or {}).get("total_verified", 0) or 0)
        _hist_acc = (stats or {}).get("direction_accuracy_pct", None)
        if _n_ver >= 30 and _hist_acc is not None:
            confidence = max(15, min(90, float(_hist_acc)))
            calibrated = True
            confidence_reason = f"verified_direction_accuracy_n={_n_ver}"
        else:
            # A model score is not calibrated confidence. Keep the probability
            # fields visible, but suppress the confidence number until the live
            # tracker has enough verified outcomes to support it.
            confidence = 0
            confidence_reason = f"uncalibrated_insufficient_verified_samples_n={_n_ver}"
        if filters_passed < 1.5:
            confidence = min(confidence, 35)
        elif filters_passed < 2.5:
            confidence = min(confidence, 65)
        prediction_result["confidence"] = int(confidence)
        prediction_result["confidence_available"] = calibrated
        prediction_result["calibrated"] = calibrated
        prediction_result["confidence_reason"] = confidence_reason
        prediction_result["verified_sample_count"] = _n_ver

        # Record AFTER every confidence adjustment so the tracked confidence
        # matches the confidence actually returned to the user/bot.
        try:
            prediction_tracker.record_prediction(symbol, tf_hours, round(pred_change, 2),
                                                 int(confidence), grade, price, feature_vec)
        except Exception:
            pass

        return prediction_result

    except Exception as e:
        return {"symbol": symbol, "timeframe_hours": tf_hours, "error": str(e)}
