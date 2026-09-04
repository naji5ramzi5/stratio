"""
Vectorbt Backtester v1.0
Vectorized (Numba) backtesting built on vectorbt 1.x.
Replaces/accelerates the pandas-loop MultiAssetBacktester.

Key API
-------
VectorbtBacktester(initial_capital, commission, slippage)
  .backtest_ohlcv(ohlcv_df, entries, exits)      -> dict of stats
  .backtest_predictions(predictions_df)           -> dict of stats
  .backtest_portfolio(ohlcv, signals, symbols)    -> portfolio-level stats (multi-symbol)

Because vectorbt vectorizes the entire simulation, a full multi-year
multi-symbol backtest runs orders of magnitude faster than the old
per-row Python loop.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import vectorbt as vbt
    HAS_VBT = True
except Exception as e:  # pragma: no cover - env dependent
    HAS_VBT = False
    _VBT_ERR = e


class VectorbtBacktester:
    def __init__(self, initial_capital=100000, commission=0.001, slippage=0.0):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    @staticmethod
    def available():
        return HAS_VBT

    # ─── core engine ───────────────────────────────────────────────

    def _from_signals(self, close, entries, exits, freq=None):
        if not HAS_VBT:
            raise RuntimeError("vectorbt not installed")
        return vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=self.initial_capital,
            fees=self.commission,
            slippage=self.slippage,
            freq=freq,
        )

    def _stats(self, pf, label=""):
        try:
            stats = pf.stats()
            if isinstance(stats, pd.DataFrame):
                stats = stats.iloc[0].to_dict() if len(stats.columns) > 1 else stats.squeeze().to_dict()
            else:
                stats = {str(k): float(v) for k, v in stats.items()}
        except Exception:
            stats = {}
        try:
            trades = pf.trades
            returns = trades.returns.values if len(trades.returns) else np.array([])
        except Exception:
            returns = np.array([])
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        win_rate = len(wins) / len(returns) * 100 if len(returns) else 0.0
        profit_factor = (wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf") if len(wins) else 0.0

        out = {
            "total_return_pct": stats.get("Total Return [%]", 0.0),
            "sharpe_ratio": stats.get("Sharpe Ratio", 0.0),
            "max_drawdown_pct": stats.get("Max Drawdown [%]", 0.0),
            "calmar_ratio": stats.get("Calmar Ratio", 0.0),
            "win_rate_pct": round(win_rate, 1),
            "profit_factor": round(float(profit_factor), 2),
            "total_trades": int(len(returns)),
            "avg_return_per_trade_pct": round(float(returns.mean() * 100), 3) if len(returns) else 0.0,
            "final_value": round(float(stats.get("Final Value [%]", 0.0) / 100 * self.initial_capital + self.initial_capital), 2),
            "sortino_ratio": stats.get("Sortino Ratio", 0.0),
            "volatility_pct": stats.get("Volatility [%]", 0.0),
        }
        if label:
            out = {f"{label}{k}": v for k, v in out.items()}
        return out

    # ─── single-symbol from OHLCV + signals ────────────────────────

    def backtest_ohlcv(self, ohlcv_df, entries, exits=None):
        """Backtest a signal on OHLCV data.

        ohlcv_df: DataFrame with a 'Close' column (or lowercase 'close').
        entries/exits: boolean Series aligned with ohlcv_df.index.
        Returns stats dict (same shape as MultiAssetBacktester).
        """
        close = ohlcv_df["Close"] if "Close" in ohlcv_df.columns else ohlcv_df["close"]
        close = close.astype(float)
        if exits is None:
            exits = pd.Series(False, index=close.index)
        entries = pd.Series(entries, index=close.index).fillna(False).astype(bool)
        exits = pd.Series(exits, index=close.index).fillna(False).astype(bool)

        pf = self._from_signals(close, entries, exits)
        stats = self._stats(pf)
        stats["buy_hold_return_pct"] = round(float((close.iloc[-1] / close.iloc[0] - 1) * 100), 2)
        return stats

    # ─── single-symbol from predictions ────────────────────────────

    def backtest_predictions(self, predictions_df, prices_df=None):
        """Backtest a predictions frame exactly like MultiAssetBacktester.run().

        predictions_df: DataFrame with columns [symbol, timestamp,
                        predicted_change_pct, confidence, current_price].
        prices_df: optional OHLCV frame for a more realistic close series;
                   defaults to current_price from predictions.
        """
        if not HAS_VBT:
            return {}

        if prices_df is not None:
            merged = pd.merge(predictions_df, prices_df, on=["symbol", "timestamp"], how="inner")
            if merged.empty:
                return {}
            close = merged["close"].astype(float)
        else:
            merged = predictions_df.copy()
            if "close" not in merged.columns:
                merged["close"] = merged["current_price"]
            close = merged["close"].astype(float)

        position = pd.Series(0, index=merged.index, dtype=int)
        position[(merged["predicted_change_pct"] > 1.5) & (merged["confidence"] >= 50)] = 1
        position[(merged["predicted_change_pct"] < -1.5) & (merged["confidence"] >= 50)] = -1

        entries = (position == 1).values
        exits = (position == 0).values
        pf = self._from_signals(close, entries, exits)
        return self._stats(pf)

    # ─── multi-symbol portfolio (vectorized) ───────────────────────

    def backtest_portfolio(self, ohlcv, signals):
        """Run a portfolio backtest across many symbols in one vectorized pass.

        ohlcv:  DataFrame with MultiIndex columns (symbol, OHLCV column) OR
                dict {symbol: DataFrame with 'Close' column}.
        signals: dict {symbol: (entries_Series, exits_Series)} OR
                 dict {symbol: position_Series (1/0/-1)}.
        """
        if not HAS_VBT:
            return {}

        symbols = sorted(signals.keys())
        closes = {}
        entries_all = {}
        exits_all = {}
        for sym in symbols:
            if isinstance(ohlcv, dict):
                df = ohlcv[sym]
            else:
                df = ohlcv[sym]
            close = df["Close"] if "Close" in df.columns else df["close"]
            closes[sym] = close.astype(float)

            sig = signals[sym]
            if isinstance(sig, tuple):
                entries_all[sym], exits_all[sym] = sig[0], sig[1]
            else:
                pos = pd.Series(sig, index=close.index)
                entries_all[sym] = (pos == 1)
                exits_all[sym] = (pos == 0)

        close_df = pd.DataFrame(closes).ffill()
        entries_df = pd.DataFrame(entries_all).fillna(False).astype(bool)
        exits_df = pd.DataFrame(exits_all).fillna(False).astype(bool)

        pf = self._from_signals(close_df, entries_df, exits_df)
        stats = self._stats(pf, label="portfolio_")
        per_symbol = {}
        for sym in symbols:
            try:
                spf = vbt.Portfolio.from_signals(
                    close=close_df[sym],
                    entries=entries_df[sym],
                    exits=exits_df[sym],
                    init_cash=self.initial_capital,
                    fees=self.commission,
                    slippage=self.slippage,
                )
                per_symbol[sym] = self._stats(spf)
            except Exception:
                per_symbol[sym] = {"error": "vbt_failed"}
        stats["symbols"] = per_symbol
        return stats


def quick_backtest_vectorbt(symbol, predictor_func, days=30, initial_capital=100000, commission=0.001, max_samples=None):
    """Fast vectorized backtest of the prediction engine over the last N days.

    Prediction calls are expensive (network + ML). We sample `max_samples`
    points (default ~days, capped at 30) and vectorize the trade simulation
    with vectorbt. Set FOUNDATION_MODELS=0 to skip Chronos/TimesFM during
    backtests for much faster runs.
    """
    from datetime import datetime, timedelta

    bt = VectorbtBacktester(initial_capital=initial_capital, commission=commission)
    end = datetime.now()
    start = end - timedelta(days=days)

    # Pull live OHLCV history for a realistic close series
    try:
        from advanced_bot import get_klines
        klines = get_klines(symbol, "1h", min(int(days * 24) + 10, 1500))
        if not klines or len(klines) < 30:
            return {"symbol": symbol, "error": "no data"}
        ts = pd.to_datetime([int(k[0]) for k in klines], unit="ms")
        close = pd.Series([float(k[4]) for k in klines], index=ts)
    except Exception as e:
        return {"symbol": symbol, "error": f"ohlcv: {e}"}

    # Run the predictor on a capped sample of points
    sample_hours = max(3, days) if max_samples is None else max_samples
    step = max(1, len(close) // sample_hours)
    pred_series = {}
    for i in range(0, len(close), step):
        try:
            pred = predictor_func(symbol, 24)
            if pred and "error" not in pred:
                pred_series[close.index[i]] = pred["predicted_change_pct"]
        except Exception:
            continue
        if i % (step * 5) == 0:
            logger.info(f"  [{symbol}] backtest point {i}/{len(close)}")

    if not pred_series:
        return {"symbol": symbol, "error": "no_predictions"}

    s = pd.Series(pred_series).sort_index()
    position = pd.Series(0, index=close.index, dtype=int)
    for ts_, pct in s.items():
        if ts_ >= close.index[-1]:
            continue
        if pct > 1.5:
            position.loc[ts_:] = 1
        elif pct < -1.5:
            position.loc[ts_:] = -1
        else:
            position.loc[ts_:] = 0

    entries = (position == 1).values
    exits = (position == 0).values
    pf = bt._from_signals(close, entries, exits)
    stats = bt._stats(pf)
    stats["symbol"] = symbol
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("vectorbt available:", HAS_VBT)
    print(quick_backtest_vectorbt("BTCUSDT", lambda sym, tf: None, days=5))
