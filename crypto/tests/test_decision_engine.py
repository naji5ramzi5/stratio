import unittest

from decision_engine import evaluate_signal


class TestDecisionEngine(unittest.TestCase):
    @staticmethod
    def _prediction(**updates):
        base = {
            "prob_up": 0.72, "prob_down": 0.18, "prob_neutral": 0.10,
            "expected_vol_pct": 4.0, "confidence_available": True,
        }
        base.update(updates)
        return base

    def test_calibrated_convergent_signal_is_trade_ready(self):
        decision = evaluate_signal(
            self._prediction(), {"imbalance_pct": 20}, {"buy_pct": 60}, 70)
        self.assertEqual(decision.decision, "TRADE_READY")
        self.assertEqual(decision.direction, "LONG")

    def test_uncalibrated_signal_is_alert_only(self):
        decision = evaluate_signal(
            self._prediction(confidence_available=False), {"imbalance_pct": 20}, {"buy_pct": 60}, 70)
        self.assertEqual(decision.decision, "ALERT_ONLY")

    def test_conflicting_liquidity_blocks_trade(self):
        decision = evaluate_signal(
            self._prediction(), {"imbalance_pct": -25}, {"buy_pct": 40}, 70)
        self.assertEqual(decision.decision, "NO_TRADE")
        self.assertTrue(decision.conflicts)

    def test_neutral_probability_blocks_trade(self):
        decision = evaluate_signal(
            self._prediction(prob_up=0.38, prob_down=0.34, prob_neutral=0.28),
            {"imbalance_pct": 20}, {"buy_pct": 60}, 70)
        self.assertEqual(decision.decision, "NO_TRADE")


if __name__ == "__main__":
    unittest.main()
