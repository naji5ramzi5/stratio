import unittest

import numpy as np

from market_events.research.features_trade import TradeFeatureEngine
from market_events.research.pit import PitView

TS = np.array([0, 60, 120, 180, 240, 300, 360, 420], dtype=np.int64) * 1_000_000
PRICE = np.array([100.0, 101.0, 99.0, 102.0, 100.5, 101.5, 103.0, 102.5])
QTY = np.array([1.0, 2.0, 0.5, 1.5, 1.0, 0.8, 2.2, 1.1])
BM = np.array([False, True, False, True, False, True, False, True])  # True = sell


def make_engine(grid_s):
    v = PitView(TS, price=PRICE, qty=QTY, buyer_maker=BM)
    grid = (np.arange(grid_s[0], grid_s[1] + 1, grid_s[2]) * 1_000_000).astype(np.int64)
    return TradeFeatureEngine(v, grid)


class TestTradeFeatures(unittest.TestCase):
    def setUp(self):
        self.e = make_engine([0, 5, 1])  # grid at T = 0..5 minutes (60s steps)

    def test_trade_count_60s(self):
        # window [T-60s, T] for T=1min..5min; T=0 has no history -> 0 events
        c = self.e.f_trade_count(60_000_000)
        self.assertEqual(c[0], 1.0)  # [0,60]: event at 60? no — event at 60 is IN
        # events at 0,60,120,... — window at T=60s: events in [0,60] = 2
        self.assertEqual(c[1], 2.0)
        self.assertEqual(c[5], 2.0)  # T=300s: events in [240,300] = 2

    def test_buy_sell_split(self):
        buy = self.e.f_buy_volume(60_000_000)
        sell = self.e.f_sell_volume(60_000_000)
        vol = self.e.f_trade_volume(60_000_000)
        # T=120s: events in [60,120]: 60(sell 2.0),120(buy 0.5)
        self.assertEqual(buy[2], 0.5)
        self.assertEqual(sell[2], 2.0)
        self.assertEqual(vol[2], 2.5)

    def test_imbalance(self):
        imb = self.e.f_buy_sell_imbalance(60_000_000)
        # T=120: (0.5-2.0)/(2.5) = -0.6
        self.assertAlmostEqual(imb[2], -0.6)
        # T=0: only the buy at ts=0 -> +1.0
        self.assertAlmostEqual(imb[0], 1.0)

    def test_aggr_buy_intensity(self):
        a = self.e.f_aggr_buy_intensity(60_000_000)
        self.assertAlmostEqual(a[2], 0.2)  # 0.5/2.5

    def test_size_mean_std(self):
        m = self.e.f_trade_size_mean(60_000_000)
        s = self.e.f_trade_size_std(60_000_000)
        # T=120: sizes {2.0, 0.5}: mean 1.25
        self.assertAlmostEqual(m[2], 1.25)
        self.assertAlmostEqual(s[2], np.std([2.0, 0.5], ddof=1))

    def test_ret_window(self):
        r = self.e.f_ret(120_000_000)
        # T=180s: last price at 180 (102.0) vs last at 60 (101.0)
        self.assertAlmostEqual(r[3], np.log(102.0) - np.log(101.0))

    def test_momentum(self):
        mom = self.e.f_momentum(60_000_000, 180_000_000)
        r60 = self.e.f_ret(60_000_000)
        r180 = self.e.f_ret(180_000_000)
        np.testing.assert_allclose(mom, r180 - r60, equal_nan=True)

    def test_realized_vol_finite(self):
        v = self.e.f_realized_vol(300_000_000)
        self.assertTrue(np.isfinite(v[4]))
        self.assertTrue(v[4] > 0)
        self.assertTrue(np.isnan(v[0]))

    def test_nan_policy(self):
        c = self.e.f_trade_count(60_000_000)
        self.assertTrue(np.isfinite(c).all() or np.isnan(c).all())


class TestDeterminism(unittest.TestCase):
    def test_identical_inputs_identical_outputs(self):
        e1 = make_engine([0, 5, 1])
        e2 = make_engine([0, 5, 1])
        for name in ("ret", "trade_volume", "buy_sell_imbalance",
                     "realized_vol", "trade_size_z"):
            a = getattr(e1, f"f_{name}")(60_000_000)
            b = getattr(e2, f"f_{name}")(60_000_000)
            np.testing.assert_array_equal(a, b)

    def test_no_future_leak_by_construction(self):
        """Feature at T must equal the same computation truncated at T."""
        e = make_engine([0, 5, 1])
        full = e.f_ret(120_000_000)
        # recompute manually at T=180s using only rows with ts <= 180s
        end = int(np.searchsorted(TS, 180 * 1_000_000, side="right"))
        local_ts = TS[:end]
        local_p = PRICE[:end]
        self.assertEqual(local_ts[-1], 180 * 1_000_000)
        ret = np.log(local_p[-1]) - np.log(local_p[
            np.searchsorted(local_ts, 180 * 1_000_000 - 120 * 1_000_000, side="left") - 1])
        self.assertAlmostEqual(full[3], ret)


class TestRegistry(unittest.TestCase):
    def test_all_registered_features_computable(self):
        e = make_engine([0, 5, 1])
        for name in ("trade_count", "trade_volume", "buy_volume",
                     "sell_volume", "buy_sell_imbalance", "volume_delta",
                     "aggr_buy_intensity", "trade_size_mean", "trade_size_std",
                     "ret", "realized_vol", "volume_intensity", "trade_size_z"):
            series = e._REGISTRY[name](e, 60_000_000)
            self.assertEqual(len(series), len(e.grid))

    def test_unknown_feature_rejected(self):
        e = make_engine([0, 5, 1])
        with self.assertRaises(KeyError):
            e.compute("not_a_feature", 60_000_000, "BTCUSDT")


if __name__ == "__main__":
    unittest.main()
