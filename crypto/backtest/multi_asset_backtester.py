"""
Multi-Asset Backtester v2.0
Tests prediction system across multiple symbols and timeframes.
Calculates portfolio-level metrics.
"""
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class MultiAssetBacktester:
    def __init__(self, initial_capital=100000, commission=0.001, slippage=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.results = {}

    def run(self, predictions_df, prices_df):
        merged = pd.merge(predictions_df, prices_df, on=["symbol", "timestamp"], how="inner")
        if merged.empty:
            return {}

        merged["position"] = 0
        merged.loc[merged["predicted_change_pct"] > 1.5, "position"] = 1
        merged.loc[merged["predicted_change_pct"] < -1.5, "position"] = -1
        merged.loc[merged["confidence"] < 50, "position"] = 0

        merged["actual_return"] = merged.groupby("symbol")["close"].pct_change()
        merged["strategy_return"] = merged["position"].shift(1) * merged["actual_return"]
        merged["strategy_return"] -= self.commission * abs(merged["position"].diff()).fillna(0)
        merged["strategy_return"] -= self.slippage * abs(merged["position"] - merged["position"].shift(1)).fillna(0)

        merged["capital"] = self.initial_capital * (1 + merged["strategy_return"]).cumprod()
        merged["buy_hold_return"] = merged.groupby("symbol")["close"].pct_change().groupby(merged["symbol"]).cumsum()

        stats = self._compute_stats(merged)
        self.results = stats
        return stats

    def _compute_stats(self, df):
        total_return = float(df["capital"].iloc[-1] / self.initial_capital - 1) * 100 if len(df) > 0 else 0
        daily_returns = df["strategy_return"].dropna()

        if len(daily_returns) < 2:
            return {"total_return_pct": total_return, "error": "insufficient_data"}

        sharpe = float(np.mean(daily_returns) / (np.std(daily_returns) + 1e-10) * np.sqrt(365))
        cum = (1 + daily_returns).cumprod()
        peak = cum.expanding().max()
        dd = (cum - peak) / peak
        max_dd = float(abs(dd.min()) * 100)
        calmar = total_return / max_dd if max_dd > 0 else 0

        win_trades = daily_returns[daily_returns > 0]
        loss_trades = daily_returns[daily_returns < 0]
        win_rate = float(len(win_trades) / max(len(daily_returns), 1) * 100)
        profit_factor = float(abs(win_trades.sum() / max(abs(loss_trades.sum()), 1e-10)))

        return {
            "total_return_pct": round(total_return, 2),
            "sharpe_ratio": round(sharpe, 3),
            "max_drawdown_pct": round(max_dd, 2),
            "calmar_ratio": round(calmar, 3),
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(profit_factor, 2),
            "total_trades": int(len(daily_returns)),
            "avg_return_per_trade_pct": round(float(daily_returns.mean() * 100), 3),
        }

    def backtest_symbol(self, symbol, predictor_func, start_date, end_date, timeframes=None):
        if timeframes is None:
            timeframes = [24]

        results = []
        current = start_date
        while current < end_date:
            for tf in timeframes:
                try:
                    pred = predictor_func(symbol, tf)
                    if pred and "error" not in pred:
                        results.append({
                            "symbol": symbol,
                            "timestamp": current,
                            "timeframe": tf,
                            "predicted_change_pct": pred["predicted_change_pct"],
                            "confidence": pred.get("confidence", 50),
                            "grade": pred.get("grade", ""),
                            "current_price": pred.get("current_price", 0),
                        })
                except Exception:
                    pass
            current += timedelta(hours=min(timeframes))

        if not results:
            return {"symbol": symbol, "error": "no_predictions"}

        df_pred = pd.DataFrame(results)
        closes = np.array([r["current_price"] for r in results])
        df_pred["close"] = closes
        df_pred["actual_return"] = df_pred["close"].pct_change().fillna(0)
        df_pred["position"] = np.where(
            (df_pred["predicted_change_pct"] > 1.5) & (df_pred["confidence"] >= 50), 1,
            np.where((df_pred["predicted_change_pct"] < -1.5) & (df_pred["confidence"] >= 50), -1, 0)
        )
        df_pred["strategy_return"] = df_pred["position"].shift(1) * df_pred["actual_return"]
        df_pred["strategy_return"] -= self.commission * abs(df_pred["position"].diff()).fillna(0)

        capital = self.initial_capital * (1 + df_pred["strategy_return"]).cumprod()
        total_ret = float(capital.iloc[-1] / self.initial_capital - 1) * 100 if len(capital) > 0 else 0

        return {
            "symbol": symbol,
            "total_return_pct": round(total_ret, 2),
            "sharpe_ratio": round(float(np.mean(df_pred["strategy_return"].dropna()) /
                                        (np.std(df_pred["strategy_return"].dropna()) + 1e-10) * np.sqrt(365)), 3),
            "win_rate_pct": round(float(len(df_pred[df_pred["strategy_return"] > 0]) /
                                        max(len(df_pred[df_pred["strategy_return"] != 0]), 1) * 100), 1),
            "total_signals": int(len(df_pred[df_pred["position"] != 0])),
        }


def quick_backtest(symbol, predictor_func, days=30):
    bt = MultiAssetBacktester()
    end = datetime.now()
    start = end - timedelta(days=days)
    return bt.backtest_symbol(symbol, predictor_func, start, end)


def quick_backtest_fast(symbol, predictor_func, days=30, initial_capital=100000, commission=0.001, max_samples=None):
    """Vectorized backtest using vectorbt when available (fallback to loop)."""
    try:
        from backtest.vectorbt_backtester import quick_backtest_vectorbt, HAS_VBT
        if HAS_VBT:
            result = quick_backtest_vectorbt(
                symbol, predictor_func, days=days,
                initial_capital=initial_capital, commission=commission,
                max_samples=max_samples
            )
            if "error" not in result:
                return result
    except Exception as e:
        logger.warning(f"vectorbt backtest failed ({e}), falling back to loop")
    return quick_backtest(symbol, predictor_func, days=days)