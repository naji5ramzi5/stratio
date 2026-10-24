"""Additional tests — Phase 10.
Failure tests, integration tests, and edge cases."""
import unittest
import numpy as np
import os
import json
import tempfile

from risk_engine import RiskEngine, RiskLimits, Position, RiskLevel, correlation_risk_check
from execution_engine import ExecutionEngine, ExecutionConfig, OrderStatus
from alerting import AlertManager


class TestFailureScenarios(unittest.TestCase):
    """Test system behavior under failure conditions."""

    def setUp(self):
        self.engine = RiskEngine()

    def test_api_outage_simulation(self):
        """Simulate API failure — risk engine should halt."""
        self.engine.trip_circuit_breaker("API timeout")
        pos = Position(pair="BTCUSDT/LTCUSDT", direction=1, size_pct=0.03, entry_z=-2.5)
        result = self.engine.check_position(pos, {"current_exposure_pct": 0, "n_open_positions": 0,
                                                    "daily_pnl_pct": 0, "weekly_pnl_pct": 0,
                                                    "current_drawdown_pct": 0, "open_pairs": [],
                                                    "current_volatility_pct": 10})
        self.assertFalse(result.approved)
        self.assertIn("circuit breaker", result.reason.lower())

    def test_corrupted_data_recovery(self):
        """Test that corrupted data doesn't crash the system."""
        # Simulate corrupted position data
        pos = Position(pair="", direction=0, size_pct=-1.0, entry_z=float('nan'))
        # The risk engine should handle this gracefully
        result = self.engine.check_position(pos, {"current_exposure_pct": 0, "n_open_positions": 0,
                                                    "daily_pnl_pct": 0, "weekly_pnl_pct": 0,
                                                    "current_drawdown_pct": 0, "open_pairs": [],
                                                    "current_volatility_pct": 10})
        # Should either reject or handle gracefully
        self.assertIsInstance(result.approved, bool)

    def test_extreme_volatility(self):
        """Test behavior under extreme volatility."""
        portfolio = {"current_exposure_pct": 0, "n_open_positions": 0,
                     "daily_pnl_pct": 0, "weekly_pnl_pct": 0,
                     "current_drawdown_pct": 0, "open_pairs": [],
                     "current_volatility_pct": 500.0}  # extreme vol
        pos = Position(pair="BTCUSDT/LTCUSDT", direction=1, size_pct=0.05, entry_z=-2.5)
        result = self.engine.check_position(pos, portfolio)
        # Should approve but with very small size due to vol scaling
        if result.approved:
            self.assertLessEqual(result.adjusted_size_pct, 0.02)  # heavily reduced


class TestIntegration(unittest.TestCase):
    """Integration tests across subsystems."""

    def test_full_trade_flow(self):
        """Test a complete trade flow: signal -> risk -> execution -> alert."""
        # Use temporary alert file to avoid loading previous alerts
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            # 1. Signal: want to open a position
            pos = Position(pair="BTCUSDT/LTCUSDT", direction=1, size_pct=0.05, entry_z=-2.5)

            # 2. Risk check
            engine = RiskEngine()
            portfolio = {"current_exposure_pct": 0.1, "n_open_positions": 2,
                         "daily_pnl_pct": -0.5, "weekly_pnl_pct": -1.0,
                         "current_drawdown_pct": 3.0, "open_pairs": [],
                         "current_volatility_pct": 12.0}
            risk_result = engine.check_position(pos, portfolio)
            self.assertTrue(risk_result.approved)

            # 3. Execution
            exec_engine = ExecutionEngine()
            exec_result = exec_engine.execute_pair_trade(
                pair=pos.pair, direction=pos.direction,
                size_usd=risk_result.adjusted_size_pct * 10000,
                prices={"BTCUSDT": 65000, "LTCUSDT": 85},
                volatilities={"BTCUSDT": 0.5, "LTCUSDT": 0.6},
            )
            self.assertTrue(exec_result.success)

            # 4. Alert
            alert_mgr = AlertManager(alerts_path=tmp_path)
            if exec_result.success:
                alert_mgr.emit("info", "execution", "Trade executed",
                              f"Executed {pos.pair} size={risk_result.adjusted_size_pct:.1%}")

            self.assertEqual(len(alert_mgr.get_unacknowledged()), 1)
        finally:
            os.unlink(tmp_path)

    def test_portfolio_correlation_flow(self):
        """Test that correlation risk is checked before opening."""
        np.random.seed(42)
        base = np.random.normal(0, 0.01, 100)
        existing = [base + np.random.normal(0, 0.001, 100) for _ in range(3)]
        new_pair = base + np.random.normal(0, 0.0005, 100)  # nearly identical to base

        ok, msg = correlation_risk_check(new_pair, existing)
        self.assertFalse(ok)
        self.assertIn("correlated", msg.lower())


class TestEdgeCases(unittest.TestCase):
    """Edge cases and boundary conditions."""

    def test_zero_size_position(self):
        """Zero size should be rejected."""
        engine = RiskEngine()
        pos = Position(pair="BTCUSDT/LTCUSDT", direction=1, size_pct=0.0, entry_z=-2.5)
        result = engine.check_position(pos, {"current_exposure_pct": 0, "n_open_positions": 0,
                                              "daily_pnl_pct": 0, "weekly_pnl_pct": 0,
                                              "current_drawdown_pct": 0, "open_pairs": [],
                                              "current_volatility_pct": 10})
        self.assertFalse(result.approved)

    def test_max_drawdown_exactly_at_limit(self):
        """Test behavior exactly at drawdown limit."""
        engine = RiskEngine()
        portfolio = {"current_exposure_pct": 0, "n_open_positions": 0,
                     "daily_pnl_pct": 0, "weekly_pnl_pct": 0,
                     "current_drawdown_pct": engine.limits.max_drawdown_pct,
                     "open_pairs": [], "current_volatility_pct": 10}
        pos = Position(pair="BTCUSDT/LTCUSDT", direction=1, size_pct=0.03, entry_z=-2.5)
        result = engine.check_position(pos, portfolio)
        self.assertFalse(result.approved)  # should halt at limit

    def test_execution_with_missing_prices(self):
        """Execution should fail gracefully with missing prices."""
        engine = ExecutionEngine()
        result = engine.execute_pair_trade(
            pair="BTCUSDT/LTCUSDT", direction=1, size_usd=1000,
            prices={},  # no prices
        )
        self.assertFalse(result.success)

    def test_consecutive_loss_recovery(self):
        """Test that consecutive losses are tracked correctly."""
        engine = RiskEngine()
        for _ in range(3):
            engine.record_trade_result(-0.01)  # losses
        self.assertEqual(engine.consecutive_losses, 3)

        engine.record_trade_result(0.02)  # win resets
        self.assertEqual(engine.consecutive_losses, 0)


if __name__ == "__main__":
    unittest.main()
