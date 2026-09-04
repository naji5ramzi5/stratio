"""Tests for the regime & volatility predictor."""
import unittest
import numpy as np
import pandas as pd

import regime_predictor as rp


def _klines_from_closes(closes, volumes=None, seed=0):
    """Build a minimal klines list from a close-price array."""
    rng = np.random.default_rng(seed)
    n = len(closes)
    if volumes is None:
        volumes = [1000.0 + rng.normal(0, 100) for _ in range(n)]
    klines = []
    for i in range(n):
        c = float(closes[i])
        h = c * 1.002
        l = c * 0.998
        o = c * (1 + rng.normal(0, 0.001))
        klines.append([
            1700000000000 + i * 3600000,  # timestamp
            o, h, l, c,
            max(volumes[i], 1.0),          # volume
            1700000000000 + (i + 1) * 3600000,  # close_time
            1000.0, 100.0, 500.0, 500.0, 0.0,
        ])
    return klines


class TestRegimePredictor(unittest.TestCase):
    def test_mean_reverting_series(self):
        """A mean-reverting (OU) series should be labeled MEAN_REVERTING."""
        rng = np.random.default_rng(42)
        s = 100.0
        closes = [s]
        for _ in range(200):
            s = s + 0.05 * (100 - s) + rng.normal(0, 0.3)
            closes.append(s)
        k = _klines_from_closes(closes)
        r = rp.predict_regime(k)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["regime"], "MEAN_REVERTING")
        self.assertGreaterEqual(r["score"], 60)
        self.assertEqual(r["action"], "TRADE")

    def test_trending_series(self):
        """A strong trending series should be labeled TRENDING."""
        rng = np.random.default_rng(7)
        s = 100.0
        closes = [s]
        for _ in range(200):
            s = s * (1 + 0.005 + rng.normal(0, 0.002))  # persistent up
            closes.append(s)
        k = _klines_from_closes(closes)
        r = rp.predict_regime(k)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["regime"], "TRENDING")
        self.assertLess(r["score"], 70)

    def test_volatile_series(self):
        """A high-vol series should be labeled VOLATILE."""
        rng = np.random.default_rng(99)
        s = 100.0
        closes = [s]
        for _ in range(200):
            s = s * (1 + rng.normal(0, 0.05))  # wild swings
            closes.append(max(s, 1.0))
        k = _klines_from_closes(closes)
        r = rp.predict_regime(k)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["regime"], "VOLATILE")
        self.assertLess(r["score"], 60)

    def test_insufficient_data(self):
        k = _klines_from_closes([100.0] * 10)
        r = rp.predict_regime(k)
        self.assertEqual(r["status"], "error")

    def test_features_shape(self):
        rng = np.random.default_rng(1)
        closes = list(100 + np.cumsum(rng.normal(0, 0.5, 100)))
        k = _klines_from_closes(closes)
        result = rp.compute_regime_features(k)
        self.assertIsNotNone(result)
        feats, names, diag = result
        self.assertEqual(len(feats), len(names))
        self.assertEqual(len(feats), 13)

    def test_score_bounds(self):
        rng = np.random.default_rng(5)
        for _ in range(10):
            closes = list(100 + np.cumsum(rng.normal(0, 1, 200)))
            k = _klines_from_closes(closes)
            r = rp.predict_regime(k)
            if r["status"] == "ok":
                self.assertGreaterEqual(r["score"], 0)
                self.assertLessEqual(r["score"], 100)


if __name__ == "__main__":
    unittest.main()
