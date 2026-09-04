"""
Tests for backtest/vectorbt_backtester.py
Run: python backtest/test_vectorbt_backtester.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd


def _ohlcv(n=3000, seed=0, start=30000.0):
    rng = np.random.default_rng(seed)
    close = start + np.cumsum(rng.normal(0, 40, n))
    close = np.maximum(close, 1)
    open_ = np.roll(close, 1); open_[0] = close[0]
    high = np.maximum(open_, close) + abs(rng.normal(0, 20, n))
    low = np.minimum(open_, close) - abs(rng.normal(0, 20, n))
    idx = pd.date_range("2025-01-01", periods=n, freq="1h")
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=idx)


def _predictions(symbols=("BTCUSDT", "ETHUSDT"), n=200):
    rng = np.random.default_rng(1)
    idx = pd.date_range("2025-06-01", periods=n, freq="1h")
    rows = []
    for sym in symbols:
        price = 30000.0 + np.cumsum(rng.normal(0, 30, n))
        for i, ts in enumerate(idx):
            rows.append({
                "symbol": sym,
                "timestamp": ts,
                "predicted_change_pct": float(rng.normal(0, 2)),
                "confidence": float(rng.integers(20, 95)),
                "current_price": float(price[i]),
            })
    return pd.DataFrame(rows)


def test_available():
    from vectorbt_backtester import VectorbtBacktester, HAS_VBT
    assert HAS_VBT, "vectorbt must be installed"
    assert VectorbtBacktester.available()
    print("OK test_available")


def test_backtest_ohlcv():
    from vectorbt_backtester import VectorbtBacktester
    df = _ohlcv()
    entries = pd.Series(False, index=df.index)
    entries.iloc[10::100] = True
    exits = pd.Series(False, index=df.index)
    exits.iloc[60::100] = True
    bt = VectorbtBacktester(initial_capital=100000, commission=0.001)
    stats = bt.backtest_ohlcv(df, entries, exits)
    for k in ("total_return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct", "total_trades", "buy_hold_return_pct"):
        assert k in stats, f"missing {k}"
    assert stats["total_trades"] > 0
    print(f"OK test_backtest_ohlcv return={stats['total_return_pct']} trades={stats['total_trades']}")


def test_backtest_predictions():
    from vectorbt_backtester import VectorbtBacktester
    preds = _predictions()
    bt = VectorbtBacktester(initial_capital=100000, commission=0.001, slippage=0.001)
    stats = bt.backtest_predictions(preds)
    assert "total_return_pct" in stats
    assert "profit_factor" in stats
    print(f"OK test_backtest_predictions return={stats['total_return_pct']}")


def test_backtest_portfolio():
    from vectorbt_backtester import VectorbtBacktester
    df1 = _ohlcv(seed=5)
    df2 = _ohlcv(seed=6)
    ohlcv = {"BTCUSDT": df1, "ETHUSDT": df2}
    signals = {}
    for sym, df in ohlcv.items():
        pos = pd.Series(0, index=df.index)
        pos.iloc[20::100] = 1
        pos.iloc[70::100] = 0
        signals[sym] = pos
    bt = VectorbtBacktester(initial_capital=100000, commission=0.001)
    res = bt.backtest_portfolio(ohlcv, signals)
    assert "portfolio_total_return_pct" in res
    assert "symbols" in res and "BTCUSDT" in res["symbols"]
    print(f"OK test_backtest_portfolio {res['portfolio_total_return_pct']}")


def test_quick_backtest_live():
    from vectorbt_backtester import quick_backtest_vectorbt
    res = quick_backtest_vectorbt("BTCUSDT", lambda sym, tf: None, days=3)
    assert "error" in res, "predictor_func returns None, so expected error path"
    print("OK test_quick_backtest_live (graceful error path)")


if __name__ == "__main__":
    test_available()
    test_backtest_ohlcv()
    test_backtest_predictions()
    test_backtest_portfolio()
    test_quick_backtest_live()
    print("\nAll vectorbt backtester tests passed.")
