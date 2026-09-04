import unittest

import numpy as np

from market_events.research.pit import PitView, PitViolationError
from market_events.research.labels import LabelEngine


def make_view(ts, price=None, qty=None, bm=None):
    n = len(ts)
    return PitView(
        np.asarray(ts, dtype=np.int64),
        price=np.asarray(price if price is not None else np.linspace(100, 200, n)),
        qty=np.asarray(qty if qty is not None else np.ones(n)),
        buyer_maker=np.asarray(bm if bm is not None else np.zeros(n, dtype=bool)),
    )


class TestPitView(unittest.TestCase):
    def test_sorted_required(self):
        with self.assertRaises(PitViolationError):
            make_view([2, 1, 3])

    def test_empty_rejected(self):
        with self.assertRaises(PitViolationError):
            make_view([])

    def test_length_mismatch(self):
        with self.assertRaises(PitViolationError):
            PitView(np.array([1, 2]), price=np.array([1.0]))

    def test_end_index_event_exactly_at_T(self):
        v = make_view([10, 20, 30])
        self.assertEqual(v.end_index(20), 2)  # event at 20 IS included
        self.assertEqual(v.end_index(19), 1)
        self.assertEqual(v.end_index(30), 3)

    def test_event_before_and_after_T(self):
        v = make_view([10, 20, 30])
        self.assertEqual(v.end_index(20), 2)  # <= 20: {10, 20}
        self.assertEqual(v.end_index(20 - 1), 1)  # only {10}
        self.assertEqual(v.end_index(100), 3)  # everything

    def test_duplicate_timestamps_handled(self):
        v = make_view([10, 10, 10, 20])
        self.assertEqual(v.end_index(10), 3)
        self.assertEqual(v.last_price_at(10), 100.0)

    def test_window_semantics(self):
        v = make_view([10, 20, 30, 40, 50])
        i0, i1 = v.window(40, 20)
        self.assertEqual((i0, i1), (2, 4))  # [20, 40]
        i0, i1 = v.window(30, 20)
        self.assertEqual((i0, i1), (1, 3))  # [10, 30]

    def test_last_price_at(self):
        v = make_view([10, 20], price=[100.0, 200.0])
        self.assertEqual(v.last_price_at(15), 100.0)
        self.assertEqual(v.last_price_at(20), 200.0)
        self.assertIsNone(v.last_price_at(5))

    def test_invalid_timestamp_rejected(self):
        v = make_view([10, 20])
        with self.assertRaises(PitViolationError):
            v.end_index(0)
        with self.assertRaises(PitViolationError):
            v.end_index(-5)


class TestLabelPitBoundaries(unittest.TestCase):
    def test_label_uses_only_events_up_to_T_plus_H(self):
        ts = np.array([10, 20, 30, 40, 50, 60])
        price = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        v = PitView(ts, price=price, qty=np.ones(6), buyer_maker=np.zeros(6, dtype=bool))
        eng = LabelEngine(v, np.array([10, 20, 30]))
        g = eng.gross_labels(20)
        # label at T=10: uses price at 10 (=100) and at 30 (=102): +2.0 log
        self.assertAlmostEqual(g["label"][0], np.log(102.0) - np.log(100.0))
        self.assertTrue(g["valid"][0])
        # label at T=20 -> T=40: log(103)-log(101)
        self.assertAlmostEqual(g["label"][1], np.log(103.0) - np.log(101.0))
        self.assertTrue(g["valid"][1])

    def test_label_invalid_without_trade_in_period(self):
        ts = np.array([10, 60])
        v = PitView(ts, price=np.array([100.0, 110.0]), qty=np.ones(2),
                    buyer_maker=np.zeros(2, dtype=bool))
        eng = LabelEngine(v, np.array([10]))
        g = eng.gross_labels(20)
        # T=10, H=20: no trade strictly inside (10, 30] -> invalid
        self.assertFalse(g["valid"][0])

    def test_label_at_exact_horizon_inclusive(self):
        ts = np.array([10, 30])
        v = PitView(ts, price=np.array([100.0, 110.0]), qty=np.ones(2),
                    buyer_maker=np.zeros(2, dtype=bool))
        eng = LabelEngine(v, np.array([10]))
        g = eng.gross_labels(20)
        # trade at exactly T+H=30 counts as inside (T, T+H]
        self.assertTrue(g["valid"][0])
        self.assertAlmostEqual(g["label"][0], np.log(110.0) - np.log(100.0))

    def test_feature_and_label_grid_aligned(self):
        """STEP 7: feature timestamp == label timestamp (same grid)."""
        ts = np.arange(10, 1000, 10)
        v = PitView(ts, price=np.full(len(ts), 100.0), qty=np.ones(len(ts)),
                    buyer_maker=np.zeros(len(ts), dtype=bool))
        grid = np.array([100, 200, 300])
        eng = LabelEngine(v, grid)
        g = eng.gross_labels(100)
        self.assertEqual(list(g["label"]), [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
