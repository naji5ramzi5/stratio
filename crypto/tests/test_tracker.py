import os
import tempfile
import unittest
from datetime import datetime, timedelta

import prediction_tracker as pt


def _backdate(tracker, idx, hours):
    old = tracker.data["predictions"][idx]["timestamp"]
    t = datetime.fromisoformat(old) - timedelta(hours=hours)
    tracker.data["predictions"][idx]["timestamp"] = t.isoformat()
    # persist so the multi-process-safe read-modify-write path sees it
    pt.PredictionTracker._save(tracker.data)


class TestNormalizeSymbol(unittest.TestCase):
    def test_preserves_full_symbol(self):
        self.assertEqual(pt.normalize_symbol("NEARUSDT"), "NEARUSDT")

    def test_uppercases(self):
        self.assertEqual(pt.normalize_symbol("btcusdt"), "BTCUSDT")

    def test_bare_base_kept(self):
        self.assertEqual(pt.normalize_symbol("NEAR"), "NEAR")


class TestTrackerFlow(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig_file = pt.TRACKER_FILE
        pt.TRACKER_FILE = os.path.join(self._tmp.name, "tracker.json")

    def tearDown(self):
        pt.TRACKER_FILE = self._orig_file
        self._tmp.cleanup()

    def test_record_and_verify_all(self):
        t = pt.PredictionTracker()
        t.record_prediction("BTCUSDT", 1, 2.0, 80, "B", 100.0)
        _backdate(t, 0, hours=25)  # mature (>= 1h*0.9) and < 48h
        n = t.verify_all({"BTCUSDT": 103.0})
        self.assertEqual(n, 1)
        p = t.data["predictions"][0]
        self.assertTrue(p["verified"])
        self.assertAlmostEqual(p["actual_change_pct"], 3.0, places=6)
        stats = t.get_stats()
        self.assertEqual(stats["total_verified"], 1)
        self.assertEqual(stats["direction_accuracy_pct"], 100.0)

    def test_immature_record_not_verified(self):
        t = pt.PredictionTracker()
        t.record_prediction("BTCUSDT", 24, 2.0, 80, "B", 100.0)
        # only 2h old, horizon is 24h -> not yet verifiable
        _backdate(t, 0, hours=2)
        self.assertEqual(t.verify_all({"BTCUSDT": 103.0}), 0)
        self.assertFalse(t.data["predictions"][0]["verified"])

    def test_symbol_key_mismatch_does_not_verify(self):
        # The historical bug: record keyed "NEARUSDT", verifier asked "NEAR".
        t = pt.PredictionTracker()
        t.record_prediction("NEARUSDT", 1, 2.0, 80, "B", 5.0)
        _backdate(t, 0, hours=25)
        self.assertEqual(t.verify_all({"NEAR": 5.1}), 0)
        self.assertFalse(t.data["predictions"][0]["verified"])

    def test_verify_prediction_timing_rule(self):
        t = pt.PredictionTracker()
        t.record_prediction("ETHUSDT", 1, -1.0, 60, "S", 100.0)
        _backdate(t, 0, hours=25)
        n = t.verify_prediction("ETHUSDT", 1, 3.0)  # price went UP, pred DOWN
        self.assertEqual(n, 1)
        stats = t.get_stats()
        self.assertEqual(stats["direction_accuracy_pct"], 0.0)
        self.assertEqual(stats["total_verified"], 1)

    def test_stats_calibration_table(self):
        t = pt.PredictionTracker()
        t.record_prediction("BTCUSDT", 1, 2.0, 80, "B", 100.0)   # correct if up
        t.record_prediction("BTCUSDT", 1, 2.0, 80, "B", 100.0)   # correct if up
        t.record_prediction("BTCUSDT", 1, -2.0, 80, "S", 100.0)  # wrong if up
        for i in range(3):
            _backdate(t, i, hours=25)
        t.verify_all({"BTCUSDT": 103.0})
        cal = t.get_stats()["calibration"]
        self.assertIn("80-90", cal)
        self.assertEqual(cal["80-90"]["realized_acc"], round(2 / 3, 3))


if __name__ == "__main__":
    unittest.main()
