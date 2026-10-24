import unittest

import numpy as np

from market_events.research import experiments as ex
from market_events.research import regimes
from market_events.research.costs import CostConfig, measure_spread_bps
from market_events.research.stats import compute_stats


class TestCosts(unittest.TestCase):
    def test_round_trip(self):
        c = CostConfig(taker_bps=5.0, slippage_bps=1.0, spread_bps=0.05)
        self.assertAlmostEqual(c.round_trip_bps(), 12.05)

    def test_spot_costs_higher(self):
        c = CostConfig(taker_bps=10.0, slippage_bps=1.0, spread_bps=0.1)
        self.assertAlmostEqual(c.round_trip_bps(), 22.1)

    def test_measure_spread_proxy(self):
        ts = np.arange(10, dtype=np.int64) * 100_000
        price = np.full(10, 100.0)
        bm = np.zeros(10, dtype=bool)
        # all buys -> asks only, no bids -> no spread measurable
        self.assertEqual(measure_spread_bps(ts, price, bm), 0.0)
        bm2 = np.array([True, False] * 5, dtype=bool)
        price2 = np.where(~bm2, 100.05, 99.95).astype(float)  # buys at ask
        # asks ~100.05, bids ~99.95 -> spread ~10bps
        s = measure_spread_bps(ts, price2, bm2)
        self.assertAlmostEqual(s, 10.0, delta=1.0)


class TestStats(unittest.TestCase):
    def test_known_correlation(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=2000)
        y = x * 0.5 + rng.normal(scale=0.1, size=2000)
        s = compute_stats(x, y, feature="f", symbol="s", horizon_us=300_000_000,
                          split="DEV")
        self.assertGreater(s.spearman, 0.9)
        self.assertGreater(s.pearson, 0.9)

    def test_directional_accuracy_perfect(self):
        x = np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
        y = np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3])
        s = compute_stats(x, y, feature="f", symbol="s", horizon_us=1, split="DEV")
        self.assertEqual(s.directional_accuracy, 1.0)

    def test_quintiles_fixed(self):
        x = np.arange(100, dtype=float)
        y = x / 10000.0
        s = compute_stats(x, y, feature="f", symbol="s", horizon_us=1, split="DEV")
        # Q5-Q1 spread must be strictly positive (monotone increasing)
        self.assertGreater(s.q5_q1_gross_bps, 0)
        self.assertGreater(s.monotonicity, 0.9)

    def test_small_sample_returns_empty_stats(self):
        s = compute_stats(np.array([1.0, 2.0]), np.array([0.1, 0.2]),
                          feature="f", symbol="s", horizon_us=1, split="DEV")
        self.assertEqual(s.n, 2)
        self.assertTrue(np.isnan(s.spearman))

    def test_missing_counts(self):
        x = np.array([1.0, np.nan, 3.0, np.nan])
        y = np.array([0.1, 0.1, 0.3, 0.3])
        s = compute_stats(x, y, feature="f", symbol="s", horizon_us=1, split="DEV")
        self.assertEqual(s.n_missing, 2)
        self.assertEqual(s.n, 2)


class TestOosSplit(unittest.TestCase):
    def test_split_days(self):
        self.assertEqual(len(regimes.DEV_DAYS), 10)
        self.assertEqual(len(regimes.OOS_DAYS), 6)
        self.assertEqual(regimes.DEV_DAYS[0], "2026-07-25")
        self.assertEqual(regimes.DEV_DAYS[-1], "2026-08-03")
        self.assertEqual(regimes.OOS_DAYS[0], "2026-08-04")
        self.assertEqual(regimes.OOS_DAYS[-1], "2026-08-09")
        self.assertEqual(set(regimes.DEV_DAYS) & set(regimes.OOS_DAYS), set())

    def test_all_days_contiguous(self):
        from datetime import date, timedelta

        expected = []
        d = date(2026, 7, 25)
        while d <= date(2026, 8, 9):
            expected.append(d.isoformat())
            d += timedelta(days=1)
        self.assertEqual(regimes.ALL_DAYS, expected)

    def test_regime_classification(self):
        rng = np.random.default_rng(1)
        returns = rng.normal(size=100)
        terc = {"low": 0.001, "high": 0.01}
        r = regimes.classify_day("2026-07-25", returns, terc)
        self.assertIn(r["vol"], ("low", "mid", "high"))
        self.assertIn(r["direction"], ("ranging", "trend_up", "trend_down"))


class TestLedger(unittest.TestCase):
    def test_classifications(self):
        r = ex.classify_dev({"BTCUSDT": 0.03, "ETHUSDT": 0.025, "spot": -0.01})
        self.assertTrue(r["promising"])
        r2 = ex.classify_dev({"BTCUSDT": 0.01, "ETHUSDT": -0.01, "spot": 0.02})
        self.assertFalse(r2["promising"])

    def test_ledger_counts(self):
        led = ex.Ledger(path=":memory:")
        led.record(ex.Experiment("a", "s", 1, "DEV", 10, 0.1, 1.0, 0.5,
                                 ex.CLASS_REJECTED))
        led.record(ex.Experiment("b", "s", 1, "DEV", 10, 0.2, 2.0, 1.0,
                                 ex.CLASS_PROMISING))
        c = led.counts()
        self.assertEqual(c["total"], 2)
        self.assertEqual(c["by_classification"][ex.CLASS_REJECTED], 1)


if __name__ == "__main__":
    unittest.main()
