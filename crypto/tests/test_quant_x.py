"""Offline unit tests for the QUANT-X indicator heuristics (no network).

Synthetic OHLCV frames with known shapes are used so the tests assert on
behaviour, not on live market data.
"""
import numpy as np
import pandas as pd
import unittest

from quant_x.indicators import (
    detect_swings, market_structure, liquidity_levels, fair_value_gaps,
    order_blocks, premium_discount, wyckoff_analysis, volume_analysis,
)


def _frame(closes=None, vols=None, n=None, seed=1):
    if n is None:
        n = len(closes)
    rng = np.random.default_rng(seed)
    close = np.array(closes, dtype=float) if closes is not None else \
        100 + np.cumsum(rng.normal(0, 0.4, n))
    open_ = close + rng.normal(0, 0.05, n)
    high = np.maximum(close, open_) + rng.uniform(0, 0.2, n)
    low = np.minimum(close, open_) - rng.uniform(0, 0.2, n)
    volume = np.full(n, 1000.0) if vols is None else np.array(vols, dtype=float)
    taker = 0.5 * volume + rng.normal(0, 20, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "taker_buy_vol": taker,
    })


class TestMarketStructure(unittest.TestCase):
    def test_uptrend_detected(self):
        i = np.arange(120)
        closes = 100 + 0.8 * i + 8 * np.sin(i / 5)
        s = market_structure(_frame(closes=closes.tolist()))
        self.assertEqual(s["trend"], "UPTREND")

    def test_downtrend_detected(self):
        i = np.arange(120)
        closes = 300 - 0.8 * i + 8 * np.sin(i / 5)
        s = market_structure(_frame(closes=closes.tolist()))
        self.assertEqual(s["trend"], "DOWNTREND")

    def test_insufficient_swings(self):
        df = _frame(n=12)
        s = market_structure(df)
        self.assertEqual(s["status"], "insufficient_swings")


class TestFVG(unittest.TestCase):
    def test_detects_bullish_gap(self):
        # big up candle two bars after a low close -> bullish FVG
        closes = [100, 100, 100, 100, 100, 106, 106, 106]
        lows = [99, 99, 99, 99, 99, 99, 105, 105]
        highs = [101, 101, 101, 101, 101, 101, 107, 107]
        opens = [100, 100, 100, 100, 100, 100, 105, 105]
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=8, freq="h"),
            "open": opens, "high": highs, "low": lows, "close": closes,
            "volume": np.ones(8) * 100, "taker_buy_vol": np.ones(8) * 50,
        })
        fvgs = fair_value_gaps(df)
        self.assertTrue(any(g["type"] == "bullish" for g in fvgs))


class TestPremiumDiscount(unittest.TestCase):
    def test_premium_and_discount_zones(self):
        lows = np.linspace(90, 110, 80)  # range 90..110
        df = _frame(closes=lows.tolist())
        pd_ = premium_discount(df)
        self.assertIn(pd_["zone"], ("PREMIUM", "DISCOUNT"))
        self.assertGreaterEqual(pd_["position_pct"], 0.0)
        self.assertLessEqual(pd_["position_pct"], 100.0)


class TestWyckoff(unittest.TestCase):
    def test_accumulation_label(self):
        # downtrend then a slowly-drifting-down flat range (sellers exhausted)
        pre = [200 - i for i in range(120)]
        rng = [170 + 3 * np.sin(i / 2) - 0.03 * i for i in range(100)]
        closes = pre + rng
        df = _frame(closes=closes)
        w = wyckoff_analysis(df)
        self.assertEqual(w["status"], "ok")
        self.assertIn("ACCUMULATION", w["phase"])

    def test_volume_profile_shapes(self):
        df = _frame(n=120)
        v = volume_analysis(df)
        self.assertEqual(v["status"], "ok")
        self.assertIn("poc", v)
        self.assertIsInstance(v["hvn"], list)
        self.assertIsInstance(v["lvn"], list)


class TestLiquidity(unittest.TestCase):
    def test_pools_are_levels(self):
        df = _frame(n=150)
        l = liquidity_levels(df)
        self.assertEqual(l["status"], "ok")
        for p in l["pools"]:
            self.assertIn("price", p)
            self.assertIn("type", p)


if __name__ == "__main__":
    unittest.main()
