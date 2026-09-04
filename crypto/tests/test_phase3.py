"""Tests for Phase 3 validation components."""
import unittest
import numpy as np
import os
import json

from phase3_validation import BaselineCreator, SharpeAuditor, VerdictEngine, Verdict, Baseline
from risk_engine import kelly_size, correlation_risk_check


class TestBaseline(unittest.TestCase):
    def test_create_baseline(self):
        baseline = BaselineCreator.create()
        self.assertEqual(baseline.strategy_version, "pairs_v2")
        self.assertEqual(baseline.pair_universe, 82)
        self.assertTrue(os.path.exists("phase2_baseline.json"))

    def test_load_baseline(self):
        baseline = Baseline.load()
        self.assertIsNotNone(baseline)
        self.assertEqual(baseline.pair_universe, 82)


class TestSharpeAudit(unittest.TestCase):
    def test_audit_basic(self):
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.01, 50)
        trades = [{"pnl": r} for r in returns]
        audit = SharpeAuditor.audit(returns, trades)
        self.assertIn("trade_based_sharpe", audit)
        self.assertIn("confidence_interval_95", audit)

    def test_extreme_sharpe(self):
        extreme_trades = [{"pnl": 0.05}, {"pnl": 0.03}, {"pnl": 0.04}]
        result = SharpeAuditor.investigate_extreme(50.0, extreme_trades)
        self.assertTrue(result["is_extreme"])
        self.assertEqual(result["assessment"], "statistically_unreliable")

    def test_normal_sharpe(self):
        result = SharpeAuditor.investigate_extreme(3.0, [{"pnl": 0.01}])
        self.assertFalse(result["is_extreme"])


class TestVerdict(unittest.TestCase):
    def test_red_verdict(self):
        verdict, reason = VerdictEngine.evaluate(
            live_results={"win_rate": 0.3, "n_trades": 2, "max_drawdown_pct": 30, "avg_slippage_bps": 30},
            oos_results={"mean_win_rate": 0.55, "total_trades": 1000},
            n_days=14,
        )
        self.assertEqual(verdict, Verdict.RED)

    def test_yellow_verdict(self):
        verdict, reason = VerdictEngine.evaluate(
            live_results={"win_rate": 0.50, "n_trades": 5, "max_drawdown_pct": 10, "avg_slippage_bps": 5},
            oos_results={"mean_win_rate": 0.55, "total_trades": 1000},
            n_days=3,
        )
        self.assertEqual(verdict, Verdict.YELLOW)

    def test_green_verdict(self):
        verdict, reason = VerdictEngine.evaluate(
            live_results={"win_rate": 0.55, "n_trades": 50, "max_drawdown_pct": 8, "avg_slippage_bps": 5},
            oos_results={"mean_win_rate": 0.55, "total_trades": 1000},
            n_days=30,
        )
        self.assertEqual(verdict, Verdict.GREEN)


class TestKellyCriterion(unittest.TestCase):
    def test_kelly_calculation(self):
        k = kelly_size(win_rate=0.55, avg_win=0.02, avg_loss=0.015)
        self.assertGreater(k, 0)
        self.assertLessEqual(k, 0.05)

    def test_kelly_capped(self):
        # Very favorable odds should still be capped
        k = kelly_size(win_rate=0.9, avg_win=0.1, avg_loss=0.01)
        self.assertLessEqual(k, 0.05)


class TestCorrelationRisk(unittest.TestCase):
    def test_high_correlation_rejected(self):
        np.random.seed(42)
        base = np.random.normal(0, 0.01, 100)
        existing = [base + np.random.normal(0, 0.0005, 100) for _ in range(3)]
        new_pair = base + np.random.normal(0, 0.0003, 100)
        ok, msg = correlation_risk_check(new_pair, existing)
        self.assertFalse(ok)

    def test_independent_approved(self):
        np.random.seed(42)
        existing = [np.random.normal(0, 0.01, 100) for _ in range(3)]
        new_pair = np.random.normal(0, 0.01, 100)
        ok, msg = correlation_risk_check(new_pair, existing)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
