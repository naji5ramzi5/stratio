"""Offline tests for the regime gate. No network.

Synthetic close series with known shapes are used so the tests assert on the
gate logic (EMA24 vs EMA96, ATR cap, confidence fallback), not on live data.

Two modes are covered:
  * gate_blocks=True  — legacy hard gate (blocks counter-trend / low-conf).
  * gate_blocks=False — DEFAULT reporting mode. Our A/B proved blocking hurts
    a weak directional edge, so the gate now REPORTS the regime but keeps
    tradable=True. These tests document that honest behaviour.
"""
import unittest

import numpy as np
import pandas as pd

from regime_gate import detect_regime, gate_signal, GATE_BLOCKS


def _df(closes, n=300):
    c = np.asarray(closes, dtype=float)
    if len(c) < n:
        base = c[-1] if len(c) else 100.0
        c = np.concatenate([c, np.full(n - len(c), base)])
    return pd.DataFrame({"close": c})


def _rising(n=300):
    return 100 + np.linspace(0, 30, n)


def _falling(n=300):
    return 300 - np.linspace(0, 30, n)


def _flat(n=300, base=100.0):
    return base + 0.05 * np.sin(np.arange(n))


class TestDetectRegime(unittest.TestCase):
    def test_uptrend(self):
        r = detect_regime(_df(_rising()))
        self.assertEqual(r["trend"], "UP")

    def test_downtrend(self):
        r = detect_regime(_df(_falling()))
        self.assertEqual(r["trend"], "DOWN")

    def test_flat(self):
        r = detect_regime(_df(_flat()))
        self.assertEqual(r["trend"], "FLAT")

    def test_warmup(self):
        r = detect_regime(_df(_rising(60), n=60))
        self.assertEqual(r["status"], "warmup")

    def test_high_atr_reported(self):
        c = _rising(300)
        h = c + 3.0
        l = c - 3.0
        df = pd.DataFrame({"close": c, "high": h, "low": l})
        r = detect_regime(df)
        self.assertIn("atr_pct", r)


class TestGateSignal(unittest.TestCase):
    def test_long_with_uptrend_passes(self):
        ml = {"predicted_change_pct": 2.0, "confidence": 70.0}
        g = gate_signal("BTCUSDT", 24, ml, df=_df(_rising()))
        self.assertEqual(g["status"], "PASS")
        self.assertTrue(g["tradable"])

    def test_long_against_downtrend_reported_not_blocked_by_default(self):
        """Default mode (gate_blocks=False): counter-trend is REPORTED, not
        blocked — our A/B proved blocking destroys value on a weak edge."""
        ml = {"predicted_change_pct": 2.0, "confidence": 70.0}
        g = gate_signal("BTCUSDT", 24, ml, df=_df(_falling()))
        self.assertEqual(g["status"], "COUNTER_TREND")
        self.assertTrue(g["tradable"])

    def test_long_against_downtrend_blocks_when_enabled(self):
        """Legacy hard-gate mode still blocks when explicitly enabled."""
        ml = {"predicted_change_pct": 2.0, "confidence": 70.0}
        g = gate_signal("BTCUSDT", 24, ml, df=_df(_falling()), gate_blocks=True)
        self.assertEqual(g["status"], "BLOCKED")
        self.assertFalse(g["tradable"])

    def test_flat_high_confidence_conditional(self):
        ml = {"predicted_change_pct": 2.0, "confidence": 80.0}
        g = gate_signal("BTCUSDT", 24, ml, df=_df(_flat()))
        self.assertEqual(g["status"], "CONDITIONAL")
        self.assertTrue(g["tradable"])

    def test_flat_low_confidence_reported_not_blocked_by_default(self):
        """Default mode: flat + low confidence is reported, not blocked."""
        ml = {"predicted_change_pct": 2.0, "confidence": 50.0}
        g = gate_signal("BTCUSDT", 24, ml, df=_df(_flat()))
        self.assertEqual(g["status"], "FLAT_LOW_CONF")
        self.assertTrue(g["tradable"])

    def test_flat_low_confidence_blocks_when_enabled(self):
        ml = {"predicted_change_pct": 2.0, "confidence": 50.0}
        g = gate_signal("BTCUSDT", 24, ml, df=_df(_flat()), gate_blocks=True)
        self.assertEqual(g["status"], "BLOCKED")
        self.assertFalse(g["tradable"])

    def test_near_zero_prediction_neutral(self):
        ml = {"predicted_change_pct": 0.01, "confidence": 90.0}
        g = gate_signal("BTCUSDT", 24, ml, df=_df(_rising()))
        self.assertEqual(g["status"], "NEUTRAL")

    def test_missing_df_unavailable(self):
        ml = {"predicted_change_pct": 2.0, "confidence": 70.0}
        g = gate_signal("BTCUSDT", 24, ml, df=None, fetch_features=False)
        self.assertEqual(g["status"], "UNAVAILABLE")
        self.assertFalse(g["tradable"])
        self.assertIn("reason", g)

    def test_short_with_downtrend_passes(self):
        ml = {"predicted_change_pct": -2.0, "confidence": 70.0}
        g = gate_signal("BTCUSDT", 24, ml, df=_df(_falling()))
        self.assertEqual(g["status"], "PASS")
        self.assertTrue(g["tradable"])

    def test_default_gate_blocks_is_false(self):
        """The default must stay False unless a future A/B proves blocking
        helps — see the module docstring for the evidence."""
        self.assertFalse(GATE_BLOCKS)


if __name__ == "__main__":
    unittest.main()
