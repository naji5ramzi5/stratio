import os
import tempfile
import unittest

from utils import state_store as ss


class TestStateStore(unittest.TestCase):
    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "state.json")
            data = {"a": 1, "b": [1.5, "x"], "c": {"nested": True}}
            self.assertTrue(ss.save_json(path, data))
            self.assertEqual(ss.load_json(path), data)

    def test_load_missing_returns_default(self):
        self.assertEqual(ss.load_json(r"C:\definitely\missing.json", {"d": 2}), {"d": 2})

    def test_load_corrupt_returns_default(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("{not json")
            self.assertEqual(ss.load_json(path, {"fallback": 1}), {"fallback": 1})

    def test_safe_float(self):
        self.assertEqual(ss.safe_float("12.5"), 12.5)
        self.assertEqual(ss.safe_float(None), 0.0)
        self.assertEqual(ss.safe_float("nope"), 0.0)
        self.assertNotIn(float("nan"), (ss.safe_float(float("nan")),))
        self.assertEqual(ss.safe_float(float("inf"), default=7.0), 7.0)

    def test_deep_merge(self):
        base = {"a": 1, "b": {"x": 1, "y": 2}}
        override = {"b": {"y": 9, "z": 3}, "c": 4}
        out = ss.deep_merge(base, override)
        self.assertEqual(out["b"], {"x": 1, "y": 9, "z": 3})
        self.assertEqual(out["c"], 4)
        self.assertEqual(base["b"]["y"], 2)  # base untouched


if __name__ == "__main__":
    unittest.main()
