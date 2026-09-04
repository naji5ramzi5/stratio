import unittest

import ml_trainer as mt


class TestCalibrationBin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cal = mt._load_calibration()

    def test_calibration_file_loaded(self):
        self.assertIsInstance(self.cal, dict)
        self.assertIn("overall", self.cal)
        self.assertGreaterEqual(float(self.cal["overall"]), 0.0)
        self.assertLessEqual(float(self.cal["overall"]), 1.0)

    def test_bin_mapping_matches_file(self):
        for lo, hi in [(50, 60), (60, 70), (70, 80), (80, 90), (90, 101)]:
            key = f"{lo}-{hi}"
            if key == "90-101":
                key = "90-100"
            if key not in self.cal:
                continue
            mid = (lo + hi) / 2 if hi <= 100 else 95.0
            mapped = mt._calibration_bin(mid)
            self.assertIsNotNone(mapped)
            self.assertAlmostEqual(mapped, float(self.cal[key]), places=6)

    def test_missing_bin_returns_none(self):
        """Bins without enough OOS samples are absent from calibration.json;
        the caller must fall back to 'overall', not to the raw agreement."""
        absent = ["50-60", "60-70", "70-80", "80-90", "90-100"]
        missing = [k for k in absent if k not in self.cal]
        if not missing:
            self.skipTest("all bins present in calibration.json")
        a = 55.0 if "50-60" in missing else 75.0
        self.assertIsNone(mt._calibration_bin(a))

    def test_fallback_overall_is_honest_estimate(self):
        # any agreement outside known bins should resolve to overall, which is
        # a real OOS number (~0.5), far below a raw 90% agreement.
        overall = float(self.cal["overall"])
        self.assertLess(overall, 0.9)
        self.assertGreater(overall, 0.4)


if __name__ == "__main__":
    unittest.main()
