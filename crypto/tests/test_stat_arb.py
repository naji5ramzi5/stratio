"""Offline tests for the relative-value (stat-arb) engine. No network.

Uses synthetic cointegrated / independent random walks so the tests assert on
the statistics, not on live market data.
"""
import unittest

import numpy as np
import pandas as pd

from stat_arb import (
    ols_hedge_ratio, spread_series, rolling_zscore, test_cointegration,
    backtest_spread, walk_forward,
)


def _cointegrated_pair(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 0.01, n))
    noise_a = rng.normal(0, 0.01, n)
    noise_b = rng.normal(0, 0.01, n)
    a = np.exp(common + noise_a)
    b = np.exp(common + noise_b)
    return a, b


def _independent_pair(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    a = np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    b = np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return a, b


class TestCointegration(unittest.TestCase):
    def test_cointegrated_pair_passes(self):
        a, b = _cointegrated_pair()
        passed, pvalue, hedge = test_cointegration(a, b)
        self.assertTrue(passed, f"expected cointegrated, p={pvalue}")
        self.assertLess(pvalue, 0.05)
        self.assertAlmostEqual(hedge, 1.0, delta=0.15)

    def test_independent_pair_fails(self):
        a, b = _independent_pair()
        passed, pvalue, hedge = test_cointegration(a, b)
        self.assertFalse(passed)
        self.assertGreaterEqual(pvalue, 0.05)

    def test_short_series_rejected(self):
        a, b = _cointegrated_pair(n=50)
        passed, _, _ = test_cointegration(a, b)
        self.assertFalse(passed)


class TestSpread(unittest.TestCase):
    def test_hedge_ratio_recovers_one(self):
        a, b = _cointegrated_pair()
        self.assertAlmostEqual(ols_hedge_ratio(a, b), 1.0, delta=0.15)

    def test_rolling_zscore_bounds(self):
        a, b = _cointegrated_pair()
        z = rolling_zscore(spread_series(a, b, 1.0), window=200)
        valid = z.dropna()
        self.assertGreater(len(valid), 100)
        self.assertLess(valid.abs().max(), 8.0)


class TestBacktest(unittest.TestCase):
    def test_mean_reversion_earns(self):
        spread = np.array([0.0, 0.5, 1.0, -2.5, -2.5, -2.5, -1.0, 0.0,
                           2.5, 2.5, 0.0], dtype=float)
        ret, trades, m = backtest_spread(spread, np.zeros(len(spread)), 1.0,
                                         0.0, 1.0, z_entry=2.0, z_exit=0.5,
                                         cost_bps=10.0)
        self.assertGreaterEqual(len(trades), 1)
        self.assertGreater(m["return_pct"], 0.0)
        self.assertEqual(m["n_trades"], len(trades))

    def test_no_signal_no_trades(self):
        spread = np.zeros(300)
        _, trades, m = backtest_spread(spread, np.zeros(len(spread)), 1.0,
                                       0.0, 1.0, z_entry=2.0, z_exit=0.5)
        self.assertEqual(len(trades), 0)
        self.assertEqual(m["n_trades"], 0)


class TestWalkForward(unittest.TestCase):
    def _df(self, a, b):
        return pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=len(a), freq="h"),
            "a": a, "b": b,
        })

    def test_cointegrated_pair_has_metrics(self):
        a, b = _cointegrated_pair()
        m = walk_forward(self._df(a, b), train_bars=1000, step=250)
        for k in ("cointegrated", "n_folds", "n_trades", "return_pct",
                  "annualized_pct", "sharpe", "max_dd_pct", "win_rate_pct"):
            self.assertIn(k, m)
        self.assertGreaterEqual(m["n_trades"], 0)
        self.assertGreaterEqual(m["n_folds"], 2)

    def test_independent_pair_not_traded(self):
        a, b = _independent_pair()
        m = walk_forward(self._df(a, b), train_bars=1000, step=250)
        self.assertEqual(m["n_trades"], 0)


if __name__ == "__main__":
    unittest.main()
