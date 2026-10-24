"""
Unit & Integration Test Suite for Bot 5 «الغني 💰».
Covers all safety invariants, risk calculations, order states, protection, reconciliation, and recovery.
"""
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from decimal import Decimal

from crypto.rich.config import RichConfig, TradingMode
from crypto.rich.db import RichDB
from crypto.rich.execution_engine import ExecutionEngine
from crypto.rich.kill_switch import KillSwitch, KillSwitchState
from crypto.rich.models import (
    Fill, OrderState, PositionState, ProtectionState,
    RichOrder, RichPosition, SignalContract, TradeSide
)
from crypto.rich.order_manager import OrderManager
from crypto.rich.position_manager import PositionManager
from crypto.rich.protection_manager import ProtectionManager
from crypto.rich.reconciliation_engine import ReconciliationEngine
from crypto.rich.recovery_engine import RecoveryEngine
from crypto.rich.risk_engine_rich import RiskEngine
from crypto.rich.secret_redactor import SecretRedactor, register_secret
from crypto.rich.signal_validator import SignalValidator


def make_signal(
    signal_id: str = "sig_test_01",
    symbol: str = "BTCUSDT",
    side: TradeSide = TradeSide.BUY,
    confidence: float = 85.0,
    pred_change_pct: float = 0.03,
    entry_ref: float = 60000.0,
) -> SignalContract:
    return SignalContract(
        signal_id=signal_id,
        signal_version="1.0",
        strategy_version="v1.0",
        source="BOT4_MASTER",
        symbol=symbol,
        side=side,
        signal_timestamp=datetime.now(timezone.utc).isoformat(),
        confidence=confidence,
        predicted_change_pct=pred_change_pct,
        expected_vol_pct=0.02,
        entry_reference=entry_ref,
    )


class TestSecretRedactor(unittest.TestCase):
    def test_redact_sensitive_strings(self):
        secret = "AIzaSyD-1234567890abcdef"
        register_secret(secret)
        msg = f"Connecting with key {secret} to binance"
        redacted = SecretRedactor.redact_text(msg)
        self.assertNotIn(secret, redacted)
        self.assertIn("REDACTED", redacted)


class TestSignalValidator(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test.db")
        self.config = RichConfig(trading_mode=TradingMode.PAPER, db_path=self.db_path)
        self.db = RichDB(self.db_path)
        self.validator = SignalValidator(self.config, self.db)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_valid_signal(self):
        sig = make_signal(signal_id="sig_btc_01", symbol="BTCUSDT", confidence=85.0)
        res = self.validator.validate(sig)
        self.assertTrue(res.is_valid)

    def test_invalid_symbol_rejected(self):
        sig = make_signal(signal_id="sig_btc_02", symbol="BTCETH", confidence=85.0)
        res = self.validator.validate(sig)
        self.assertFalse(res.is_valid)
        self.assertIn("USDT", res.reason)



class TestRiskEngine(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_risk.db")
        self.config = RichConfig(trading_mode=TradingMode.PAPER, db_path=self.db_path)
        self.db = RichDB(self.db_path)
        self.risk_engine = RiskEngine(self.config, self.db)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_risk_evaluation_and_sizing(self):
        sig = make_signal(signal_id="sig_eth_01", symbol="ETHUSDT", confidence=90.0, entry_ref=3000.0, pred_change_pct=0.04)
        portfolio_state = {
            "total_equity": Decimal("10000.0"),
            "open_positions": [],
            "active_trades_count": 0,
        }
        decision = self.risk_engine.evaluate(sig, portfolio_state)
        self.assertTrue(decision.approved)
        self.assertGreater(decision.allocated_quantity, Decimal("0"))
        self.assertIsNotNone(decision.stop_loss_price)
        self.assertIsNotNone(decision.take_profit_price)
        self.assertLess(decision.stop_loss_price, Decimal("3000.0"))
        self.assertGreater(decision.take_profit_price, Decimal("3000.0"))


class TestPositionManager(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_pos.db")
        self.config = RichConfig(trading_mode=TradingMode.PAPER, db_path=self.db_path)
        self.db = RichDB(self.db_path)
        self.pos_mgr = PositionManager(self.config, self.db)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_open_and_close_position_pnl(self):
        sig = make_signal(signal_id="sig_sol_01", symbol="SOLUSDT", confidence=88.0, entry_ref=150.0, pred_change_pct=0.05)
        order = RichOrder(
            internal_order_id="ord_sol_01",
            client_order_id="rch_sol_01",
            signal_id=sig.signal_id,
            position_id="pos_sol_01",
            symbol="SOLUSDT",
            side=TradeSide.BUY,
            order_type="MARKET",
            requested_qty=Decimal("10.0"),
        )
        entry_fill = Fill(
            fill_id="f1",
            order_id="ord_sol_01",
            symbol="SOLUSDT",
            side=TradeSide.BUY,
            price=Decimal("150.0"),
            quantity=Decimal("10.0"),
            commission=Decimal("1.5"),
            commission_asset="USDT",
            filled_at=datetime.now(timezone.utc).isoformat(),
        )
        pos = self.pos_mgr.open_position(sig, order, [entry_fill])
        self.assertEqual(pos.quantity, Decimal("10.0"))
        self.assertEqual(pos.average_entry, Decimal("150.0"))
        self.assertEqual(pos.fees_paid, Decimal("1.5"))

        # Update price to 160 (profit)
        updated = self.pos_mgr.update_price(pos, Decimal("160.0"))
        self.assertEqual(updated.unrealized_pnl, Decimal("100.0"))
        self.assertEqual(updated.mfe, Decimal("100.0"))

        # Exit at 160
        exit_fill = Fill(
            fill_id="f2",
            order_id="ord_sol_exit",
            symbol="SOLUSDT",
            side=TradeSide.SELL,
            price=Decimal("160.0"),
            quantity=Decimal("10.0"),
            commission=Decimal("1.6"),
            commission_asset="USDT",
            filled_at=datetime.now(timezone.utc).isoformat(),
        )
        closed = self.pos_mgr.close_position(pos, [exit_fill], exit_reason="TAKE_PROFIT")
        self.assertEqual(closed.position_state, PositionState.CLOSED)
        self.assertEqual(closed.gross_pnl, Decimal("100.0"))
        self.assertEqual(closed.fees_paid, Decimal("3.1"))
        self.assertEqual(closed.net_pnl, Decimal("100.0") - Decimal("3.1"))


class TestProtectionAndKillSwitch(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_kill.db")
        self.config = RichConfig(trading_mode=TradingMode.PAPER, db_path=self.db_path)
        self.db = RichDB(self.db_path)
        self.kill_switch = KillSwitch(self.config, self.db)
        self.prot_mgr = ProtectionManager(self.config, self.db, kill_switch=self.kill_switch)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_protection_failure_triggers_kill_switch(self):
        pos = RichPosition(
            position_id="pos_fail_01",
            signal_id="sig_01",
            symbol="BTCUSDT",
            side=TradeSide.BUY,
            quantity=Decimal("1.0"),
            average_entry=None,  # Will cause failure
            position_state=PositionState.OPEN,
        )
        res = self.prot_mgr.setup_protection(pos)
        self.assertEqual(res.protection_state, ProtectionState.FAILED)
        self.assertFalse(self.kill_switch.can_trade())


class TestFullEndToEndSimulation(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp_dir, "test_e2e.db")
        self.config = RichConfig(trading_mode=TradingMode.PAPER, db_path=self.db_path)
        self.db = RichDB(self.db_path)
        self.kill_switch = KillSwitch(self.config, self.db)
        self.exec_engine = ExecutionEngine(self.config, self.db, kill_switch=self.kill_switch)
        self.reconciler = ReconciliationEngine(self.config, self.db, kill_switch=self.kill_switch)
        self.recovery = RecoveryEngine(self.config, self.db, kill_switch=self.kill_switch)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_end_to_end_trade_lifecycle(self):
        # 1. Recovery on startup
        rec_summary = self.recovery.run_startup_recovery()
        self.assertTrue(rec_summary.is_safe_to_trade)

        # 2. Ingest signal
        sig = make_signal(signal_id="sig_e2e_01", symbol="BTCUSDT", confidence=88.0, entry_ref=65000.0, pred_change_pct=0.03)
        pos = self.exec_engine.process_trade_signal(sig)
        self.assertIsNotNone(pos)
        self.assertEqual(pos.symbol, "BTCUSDT")
        self.assertEqual(pos.protection_state, ProtectionState.ACTIVE)

        # 3. Reconcile
        rec_report = self.reconciler.run_reconciliation()
        self.assertTrue(rec_report.is_clean)

        # 4. Market price movement triggers Take Profit
        tp_price = pos.take_profit_1 or Decimal("67000.0")
        closed = self.exec_engine.update_market_prices({"BTCUSDT": tp_price + Decimal("500.0")})
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0].position_state, PositionState.CLOSED)
        self.assertGreater(closed[0].net_pnl, Decimal("0"))


if __name__ == "__main__":
    unittest.main()
