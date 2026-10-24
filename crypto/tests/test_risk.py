"""Tests for the Risk Engine."""
import unittest
import numpy as np
from risk_engine import RiskEngine, RiskLimits, Position, RiskLevel, kelly_size, correlation_risk_check


class TestRiskEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RiskEngine()
        self.base_portfolio = {
            "current_exposure_pct": 0.3,
            "n_open_positions": 5,
            "daily_pnl_pct": -1.0,
            "weekly_pnl_pct": -2.0,
            "current_drawdown_pct": 5.0,
            "open_pairs": ["ETHUSDT/XRPUSDT"],
            "current_volatility_pct": 15.0,
        }

    def test_normal_position_approved(self):
        pos = Position(pair="BTCUSDT/LTCUSDT", direction=1, size_pct=0.03, entry_z=-2.5)
        result = self.engine.check_position(pos, self.base_portfolio)
        self.assertTrue(result.approved)

    def test_exposure_limit_adjusts_size(self):
        portfolio = dict(self.base_portfolio, current_exposure_pct=0.55)
        pos = Position(pair="BTCUSDT/LTCUSDT", direction=1, size_pct=0.10, entry_z=-2.5)
        result = self.engine.check_position(pos, portfolio)
        self.assertTrue(result.approved)
        self.assertLessEqual(result.adjusted_size_pct, 0.05)  # max_position_pct

    def test_max_drawdown_halt(self):
        portfolio = dict(self.base_portfolio, current_drawdown_pct=20.0)
        pos = Position(pair="BTCUSDT/LTCUSDT", direction=1, size_pct=0.03, entry_z=-2.5)
        result = self.engine.check_position(pos, portfolio)
        self.assertFalse(result.approved)
        self.assertEqual(result.risk_level, RiskLevel.HALTED)

    def test_daily_loss_limit(self):
        portfolio = dict(self.base_portfolio, daily_pnl_pct=-5.0)
        pos = Position(pair="BTCUSDT/LTCUSDT", direction=1, size_pct=0.03, entry_z=-2.5)
        result = self.engine.check_position(pos, portfolio)
        self.assertFalse(result.approved)

    def test_max_positions(self):
        portfolio = dict(self.base_portfolio, n_open_positions=15)
        pos = Position(pair="BTCUSDT/LTCUSDT", direction=1, size_pct=0.03, entry_z=-2.5)
        result = self.engine.check_position(pos, portfolio)
        self.assertFalse(result.approved)

    def test_duplicate_pair_rejected(self):
        portfolio = dict(self.base_portfolio, open_pairs=["ETHUSDT/XRPUSDT", "BTCUSDT/LTCUSDT"])
        pos = Position(pair="BTCUSDT/LTCUSDT", direction=1, size_pct=0.03, entry_z=-2.5)
        result = self.engine.check_position(pos, portfolio)
        self.assertFalse(result.approved)

    def test_volatility_scaling(self):
        # High vol should reduce size relative to moderate vol
        portfolio_high_vol = dict(self.base_portfolio, current_exposure_pct=0.1,
                                   current_volatility_pct=30.0, open_pairs=[])
        pos = Position(pair="BTCUSDT/LTCUSDT", direction=1, size_pct=0.05, entry_z=-2.5)
        result_high = self.engine.check_position(pos, portfolio_high_vol)

        portfolio_mod_vol = dict(self.base_portfolio, current_exposure_pct=0.1,
                                 current_volatility_pct=10.0, open_pairs=[])
        pos2 = Position(pair="ETHUSDT/XRPUSDT", direction=1, size_pct=0.05, entry_z=-2.5)
        result_mod = self.engine.check_position(pos2, portfolio_mod_vol)

        self.assertTrue(result_high.approved)
        self.assertTrue(result_mod.approved)
        # High vol should give smaller size than moderate vol
        self.assertLess(result_high.adjusted_size_pct, result_mod.adjusted_size_pct)

    def test_circuit_breaker(self):
        self.engine.trip_circuit_breaker("test")
        pos = Position(pair="BTCUSDT/LTCUSDT", direction=1, size_pct=0.03, entry_z=-2.5)
        result = self.engine.check_position(pos, self.base_portfolio)
        self.assertFalse(result.approved)

    def test_record_trade_result(self):
        self.engine.record_trade_result(-0.01)  # loss
        self.assertEqual(self.engine.consecutive_losses, 1)
        self.engine.record_trade_result(0.02)  # win
        self.assertEqual(self.engine.consecutive_losses, 0)


class TestKelly(unittest.TestCase):
    def test_kelly_basic(self):
        k = kelly_size(win_rate=0.55, avg_win=0.02, avg_loss=0.015)
        self.assertGreater(k, 0)
        self.assertLessEqual(k, 0.05)

    def test_kelly_zero_loss(self):
        k = kelly_size(win_rate=0.55, avg_win=0.02, avg_loss=0.0)
        self.assertEqual(k, 0.0)

    def test_kelly_extreme_win_rate(self):
        k = kelly_size(win_rate=1.0, avg_win=0.02, avg_loss=0.015)
        self.assertEqual(k, 0.0)


class TestCorrelation(unittest.TestCase):
    def test_high_correlation_rejected(self):
        np.random.seed(42)
        r1 = np.random.normal(0, 0.01, 100)
        r2 = r1 + np.random.normal(0, 0.001, 100)  # nearly identical
        ok, msg = correlation_risk_check(r1, [r2])
        self.assertFalse(ok)

    def test_low_correlation_approved(self):
        np.random.seed(42)
        r1 = np.random.normal(0, 0.01, 100)
        r2 = np.random.normal(0, 0.01, 100)  # independent
        ok, msg = correlation_risk_check(r1, [r2])
        self.assertTrue(ok)

    def test_no_existing_positions(self):
        r1 = np.random.normal(0, 0.01, 100)
        ok, msg = correlation_risk_check(r1, [])
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
