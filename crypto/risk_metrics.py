import numpy as np
import logging

logger = logging.getLogger(__name__)


def sharpe_ratio(returns, risk_free_rate=0.02):
    returns = np.asarray(returns)
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    excess = np.mean(returns) - risk_free_rate / 252
    return float(excess / np.std(returns) * np.sqrt(252))


def sortino_ratio(returns, risk_free_rate=0.02):
    returns = np.asarray(returns)
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return 1.0
    excess = np.mean(returns) - risk_free_rate / 252
    return float(excess / np.std(downside) * np.sqrt(252))


def calmar_ratio(returns):
    returns = np.asarray(returns)
    if len(returns) < 2:
        return 0.0
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = abs(float(np.min(dd)))
    annual_return = float((1 + np.mean(returns)) ** 252 - 1)
    return annual_return / max_dd if max_dd > 0 else 0.0


def max_drawdown(prices):
    prices = np.asarray(prices)
    if len(prices) < 2:
        return 0.0
    peak = np.maximum.accumulate(prices)
    dd = (prices - peak) / peak
    return float(abs(np.min(dd)) * 100)


def win_rate(predictions, actuals):
    preds = np.asarray(predictions)
    actuals = np.asarray(actuals)
    if len(preds) == 0:
        return 0.0
    wins = np.sum((preds > 0) & (actuals > 0)) + np.sum((preds < 0) & (actuals < 0))
    return float(wins / len(preds) * 100)


def profit_factor(predictions, actuals):
    preds = np.asarray(predictions)
    actuals = np.asarray(actuals)
    if len(preds) == 0:
        return 0.0
    gains = np.sum(actuals[(preds > 0) & (actuals > 0)])
    losses = abs(np.sum(actuals[(preds > 0) & (actuals < 0)]))
    return gains / losses if losses > 0 else float('inf')


def kelly_criterion(win_prob, avg_win, avg_loss):
    if avg_loss <= 0:
        return 0.0
    b = avg_win / avg_loss if avg_loss > 0 else 0
    p = win_prob
    q = 1 - p
    kelly = (b * p - q) / b if b > 0 else 0
    return float(max(0, min(kelly, 0.25)))


def suggested_position_size(confidence, volatility_pct, kelly_fraction=0.25):
    base = confidence / 100.0
    vol_penalty = min(volatility_pct / 10, 1.0)
    size = base * (1 - vol_penalty * 0.5) * kelly_fraction
    return round(max(0.01, min(size, 0.5)) * 100, 1)


def expected_value(predicted_change_pct, confidence, volatility_pct):
    ev = predicted_change_pct * (confidence / 100) * 0.01
    risk = volatility_pct * (1 - confidence / 100) * 0.01
    return round(ev - risk, 4)


def compute_risk_metrics(prediction_result, historical_prices=None):
    pred_change = prediction_result.get("predicted_change_pct", 0)
    confidence = prediction_result.get("confidence", 50)
    atr_pct = prediction_result.get("atr_pct", 2)

    ev = expected_value(pred_change, confidence, atr_pct)
    pos_size = suggested_position_size(confidence, atr_pct)

    if pred_change > 0:
        stop_loss = max(0.5, min(atr_pct * 1.5, 5))
        take_profit = max(stop_loss * 1.5, abs(pred_change) * 1.2)
    else:
        stop_loss = max(0.5, min(atr_pct * 1.5, 5))
        take_profit = max(stop_loss * 1.5, abs(pred_change) * 1.2)

    max_dd = 0.0
    if historical_prices is not None and len(historical_prices) > 20:
        max_dd = max_drawdown(historical_prices)

    return {
        "expected_value": ev,
        "suggested_position_size_pct": pos_size,
        "suggested_stop_loss_pct": round(stop_loss, 2),
        "suggested_take_profit_pct": round(take_profit, 2),
        "risk_reward_ratio": round(take_profit / stop_loss, 2) if stop_loss > 0 else 0,
        "max_drawdown_pct": round(max_dd, 2),
        "kelly_fraction": round(kelly_criterion(confidence / 100, abs(pred_change), atr_pct), 3),
    }