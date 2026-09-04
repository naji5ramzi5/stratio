import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import pair_paper as pp


def _make_pair(n=1200, hedge=0.8, seed=7, final_z=-3.0):
    """Cointegrated pair whose spread is EXACTLY a stationary OU process.
    Longer n (default 1200 for 1h, use 2400 for 4h) gives the Engle-Granger
    test more power so cointegration reliably passes."""
    rng = np.random.default_rng(seed)
    log_a = np.cumsum(rng.normal(0, 0.01, n))
    s = np.zeros(n)
    for i in range(1, n):
        s[i] = (1 - 0.05) * s[i - 1] + rng.normal(0, 0.008)
    base_mean, base_std = float(np.mean(s[:-1])), float(np.std(s[:-1]))
    if base_std == 0:
        base_std = 1.0
    s[-1] = base_mean + final_z * base_std
    log_b = log_a / hedge - s / hedge
    return pd.DataFrame({"a": np.exp(log_a), "b": np.exp(log_b)})


# Seeds that reliably pass Engle-Granger at n=2400/train=400 (verified offline).
GOOD_4H_SEEDS = [25, 26]
BAD_4H_SEEDS = [22, 24]


class TestPairPaperBook(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._book_file = os.path.join(self._tmp.name, "paper_pairs.json")

    def tearDown(self):
        self._tmp.cleanup()

    def test_opens_on_stretched_spread(self):
        df = _make_pair(final_z=-3.0)
        df4 = _make_pair(n=2400, final_z=-3.0, seed=GOOD_4H_SEEDS[0])
        book = pp.PairPaperBook(self._book_file)
        events = book.step("BTCUSDT", "LTCUSDT", df=df, df_4h=df4, train_bars=1000)
        self.assertTrue(any(e["type"] == "open" for e in events), f"expected open, got {events}")
        self.assertEqual(book.open_positions[0].direction, 1.0)

    def test_multi_tf_rejects_when_4h_disagrees(self):
        """1h says long spread, but 4h not stretched → skip."""
        df = _make_pair(final_z=-3.0)
        df4 = _make_pair(final_z=0.0, seed=BAD_4H_SEEDS[0])
        book = pp.PairPaperBook(self._book_file)
        events = book.step("BTCUSDT", "LTCUSDT", df=df, df_4h=df4, train_bars=1000)
        self.assertTrue(any(e.get("type") == "skip_mtf" for e in events),
                        f"expected skip_mtf, got {events}")

    def test_closes_when_spread_reverts(self):
        df = _make_pair(final_z=-3.0)
        df4 = _make_pair(n=2400, final_z=-3.0, seed=GOOD_4H_SEEDS[0])
        book = pp.PairPaperBook(self._book_file)
        book.step("BTCUSDT", "LTCUSDT", df=df, df_4h=df4, train_bars=1000)
        self.assertEqual(len(book.open_positions), 1)
        df2 = _make_pair(final_z=0.0, seed=8)
        events2 = book.step("BTCUSDT", "LTCUSDT", df=df2, train_bars=1000)
        self.assertTrue(any(e["type"] == "close" for e in events2))

    def test_no_open_without_cointegration(self):
        rng = np.random.default_rng(1)
        n = 1200
        df = pd.DataFrame({"a": np.exp(np.cumsum(rng.normal(0, 0.01, n))),
                            "b": np.exp(np.cumsum(rng.normal(0, 0.01, n)))})
        book = pp.PairPaperBook(self._book_file)
        events = book.step("BTCUSDT", "LTCUSDT", df=df, train_bars=1000, z_entry=0.5)
        self.assertFalse(any(e["type"] == "open" for e in events))

    def test_expiry_closes_position(self):
        df = _make_pair(final_z=-3.0)
        df4 = _make_pair(n=2400, final_z=-3.0, seed=GOOD_4H_SEEDS[0])
        book = pp.PairPaperBook(self._book_file)
        book.step("BTCUSDT", "LTCUSDT", df=df, df_4h=df4, train_bars=1000)
        df2 = _make_pair(final_z=-2.5, seed=9)
        events = book.step("BTCUSDT", "LTCUSDT", df=df2, train_bars=1000, max_hold_bars=1)
        self.assertTrue(any(e["type"] == "close" and e["reason"] == "closed_expired"
                            for e in events))

    def test_stoploss_closes_when_z_explodes(self):
        df = _make_pair(final_z=-3.0)
        df4 = _make_pair(n=2400, final_z=-3.0, seed=GOOD_4H_SEEDS[1])
        book = pp.PairPaperBook(self._book_file)
        book.step("BTCUSDT", "LTCUSDT", df=df, df_4h=df4, train_bars=1000)
        df2 = _make_pair(final_z=-4.0, seed=11)
        events = book.step("BTCUSDT", "LTCUSDT", df=df2, train_bars=1000, z_stop=3.5)
        self.assertTrue(any(e["type"] == "close" and e["reason"] == "closed_stoploss"
                            for e in events), f"got {events}")

    def test_no_stoploss_if_z_within_band(self):
        df = _make_pair(final_z=-3.0)
        df4 = _make_pair(n=2400, final_z=-3.0, seed=GOOD_4H_SEEDS[0])
        book = pp.PairPaperBook(self._book_file)
        book.step("BTCUSDT", "LTCUSDT", df=df, df_4h=df4, train_bars=1000)
        df2 = _make_pair(final_z=-3.0, seed=12)
        events = book.step("BTCUSDT", "LTCUSDT", df=df2, train_bars=1000, z_stop=3.5)
        self.assertFalse(any(e.get("reason") == "closed_stoploss" for e in events))

    def test_position_sizing_scales_with_z(self):
        size_2 = pp._size_for_z(2.0)
        size_3 = pp._size_for_z(3.0)
        size_5 = pp._size_for_z(5.0)
        self.assertEqual(size_2, pp.MIN_POSITION_SIZE)
        self.assertGreater(size_3, size_2)
        self.assertEqual(size_5, pp.MAX_POSITION_SIZE)

    def test_gross_exposure_cap_blocks_new_entry(self):
        df = _make_pair(final_z=-3.0)
        df4 = _make_pair(n=2400, final_z=-3.0, seed=GOOD_4H_SEEDS[0])
        book = pp.PairPaperBook(self._book_file)
        pos = pp.PairPosition("ETHUSDT", "BTCUSDT", 1.0, 0.5, 0.0, -2.5,
                              "2024-01-01T00:00:00+00:00", 25.0, 0.30)
        book.positions.append(pos)
        events = book.step("BTCUSDT", "LTCUSDT", df=df, df_4h=df4, train_bars=1000, max_gross=0.3)
        self.assertTrue(any(e.get("type") == "skip_exposure" for e in events))

    def test_stats_after_roundtrip(self):
        df = _make_pair(final_z=-3.0)
        df4 = _make_pair(n=2400, final_z=-3.0, seed=GOOD_4H_SEEDS[1])
        book = pp.PairPaperBook(self._book_file)
        book.step("BTCUSDT", "LTCUSDT", df=df, df_4h=df4, train_bars=1000)
        df2 = _make_pair(final_z=0.0, seed=10)
        book.step("BTCUSDT", "LTCUSDT", df=df2, train_bars=1000)
        st = book.stats()
        self.assertEqual(st["total_closed"], 1)
        self.assertIn("win_rate_pct", st)
        self.assertIn("gross_exposure", st)

    def test_direction_df_basic(self):
        book = pp.PairPaperBook(self._book_file)
        df_down = _make_pair(final_z=3.0, seed=30)  # spread stretched UP -> short signal
        d, z, h, p = book._direction_df(df_down, 1000)
        self.assertEqual(d, -1)
        self.assertGreater(z, 2.0)


if __name__ == "__main__":
    unittest.main()
