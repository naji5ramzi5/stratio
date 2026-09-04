"""Phase 3: Live Paper Trading Engine.
Runs the strategy in real-time with actual market data but simulates execution.
Logs all signals, trades, and produces validation reports."""
from __future__ import annotations

import json
import os
import time
import numpy as np
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional

from data_loader.binance_ohlcv import fetch_klines
from stat_arb import ols_hedge_ratio, spread_series, test_cointegration, half_life
from pair_selector import PairLifecycleManager, PairState
from risk_engine import RiskEngine, Position, RiskLevel
from execution_engine import ExecutionEngine, ExecutionConfig
from phase3_validation import Baseline, SharpeAuditor, VerdictEngine, LiveSignal, LiveTrade, Verdict

logger = logging.getLogger("live_paper")


class LivePaperEngine:
    """Live paper trading engine for Phase 3 validation."""

    def __init__(self, capital: float = 10000.0):
        self.capital = capital
        self.pair_mgr = PairLifecycleManager()
        self.risk_engine = RiskEngine()
        self.exec_engine = ExecutionEngine()
        self.trades: List[LiveTrade] = []
        self.signals: List[LiveSignal] = []
        self.open_trades: Dict[str, LiveTrade] = {}
        self.day_start_capital = capital
        self.week_start_capital = capital
        self.peak_capital = capital
        self.start_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def get_live_signal(self, pair: str, timeframe: str = "1h") -> Optional[LiveSignal]:
        """Generate a live signal for a pair using real market data."""
        a_sym, b_sym = pair.split("/")

        # Fetch real-time data
        lookback = {"5m": 2000, "15m": 1500, "1h": 3000, "4h": 1500, "1d": 800}.get(timeframe, 3000)
        train_bars = {"5m": 800, "15m": 600, "1h": 1000, "4h": 500, "1d": 300}.get(timeframe, 1000)

        ka = fetch_klines(a_sym, timeframe, lookback)
        kb = fetch_klines(b_sym, timeframe, lookback)

        if not ka or not kb or len(ka) < train_bars + 50:
            return None

        # Align
        ta = {int(r[0]): float(r[4]) for r in ka}
        tb = {int(r[0]): float(r[4]) for r in kb}
        common = sorted(set(ta) & set(tb))
        if len(common) < train_bars + 10:
            return None

        a = np.array([ta[t] for t in common][-lookback:])
        b = np.array([tb[t] for t in common][-lookback:])

        # Compute hedge ratio and cointegration on train window
        h = ols_hedge_ratio(a[:train_bars], b[:train_bars])
        passed, pval, h = test_cointegration(a[:train_bars], b[:train_bars], hedge_ratio=h)

        if not passed:
            return None

        # Current spread and z-score
        s_tr = spread_series(a[:train_bars], b[:train_bars], h)
        mu, sd = float(np.mean(s_tr)), float(np.std(s_tr))
        if sd <= 0:
            return None

        current_spread = float(spread_series(a[-1:], b[-1:], h)[0])
        z = (current_spread - mu) / sd

        # Half-life
        hl = half_life(s_tr)

        # Create signal
        signal = LiveSignal(
            signal_id=f"{pair}_{datetime.now().timestamp():.0f}",
            timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            pair=pair,
            timeframe=timeframe,
            price_a=float(a[-1]),
            price_b=float(b[-1]),
            spread=current_spread,
            hedge_ratio=h,
            z_score=round(z, 3),
            half_life=round(hl, 1) if hl else 0,
            cointegration_pvalue=round(pval, 4),
            stability_score=0,
            market_regime="unknown",
            expected_slippage_bps=5.0,
            expected_fees_bps=20.0,
            risk_score=0,
            decision="pending",
        )

        return signal

    def validate_signal(self, signal: LiveSignal) -> LiveSignal:
        """Validate a signal through risk engine."""
        # Check if entry condition met
        if abs(signal.z_score) < 2.0:
            signal.decision = "rejected"
            signal.rejection_reason = "z-score below entry threshold"
            return signal

        direction = 1 if signal.z_score < 0 else -1

        # Risk check
        pos = Position(
            pair=signal.pair, direction=direction,
            size_pct=0.03, entry_z=signal.z_score,
        )
        portfolio = {
            "current_exposure_pct": len(self.open_trades) * 0.03,
            "n_open_positions": len(self.open_trades),
            "daily_pnl_pct": (self.capital - self.day_start_capital) / self.day_start_capital * 100,
            "weekly_pnl_pct": (self.capital - self.week_start_capital) / self.week_start_capital * 100,
            "current_drawdown_pct": (1 - self.capital / self.peak_capital) * 100,
            "open_pairs": list(self.open_trades.keys()),
            "current_volatility_pct": 10.0,
        }

        risk_result = self.risk_engine.check_position(pos, portfolio)

        if not risk_result.approved:
            signal.decision = "rejected"
            signal.rejection_reason = risk_result.reason
        else:
            signal.decision = "accepted"
            signal.direction = direction

        return signal

    def execute_signal(self, signal: LiveSignal) -> Optional[LiveTrade]:
        """Execute an accepted signal as a paper trade."""
        if signal.decision != "accepted":
            return None

        result = self.exec_engine.execute_pair_trade(
            pair=signal.pair,
            direction=signal.direction,
            size_usd=0.03 * self.capital,
            prices={signal.pair.split("/")[0]: signal.price_a,
                    signal.pair.split("/")[1]: signal.price_b},
            volatilities={signal.pair.split("/")[0]: 0.5, signal.pair.split("/")[1]: 0.6},
        )

        if not result.success:
            return None

        trade = LiveTrade(
            trade_id=f"trade_{datetime.now().timestamp():.0f}",
            signal_id=signal.signal_id,
            pair=signal.pair,
            timeframe=signal.timeframe,
            direction=signal.direction,
            entry_timestamp=signal.timestamp,
            entry_price_a=result.leg_a.fill_price,
            entry_price_b=result.leg_b.fill_price,
            entry_z=signal.z_score,
            signal_price_a=signal.price_a,
            signal_price_b=signal.price_b,
            execution_price_a=result.leg_a.fill_price,
            execution_price_b=result.leg_b.fill_price,
            slippage_a_bps=result.leg_a.slippage_pct,
            slippage_b_bps=result.leg_b.slippage_pct,
            fees_a=result.leg_a.fee,
            fees_b=result.leg_b.fee,
            size_usd=0.03 * self.capital,
        )

        self.trades.append(trade)
        self.open_trades[signal.pair] = trade
        return trade

    def run_once(self, timeframe: str = "1h") -> Dict:
        """Run one iteration of live paper trading."""
        results = {"signals": 0, "accepted": 0, "rejected": 0, "trades": 0}

        # Get active pairs
        active_pairs = self.pair_mgr.get_active_pairs()
        if not active_pairs:
            # Initialize from DB if empty
            self.pair_mgr.initialize_from_db()
            active_pairs = self.pair_mgr.get_active_pairs()

        for pair_rec in active_pairs[:10]:  # limit per iteration
            pair = pair_rec.pair
            tf = pair_rec.timeframe

            # Skip if already open
            if pair in self.open_trades:
                continue

            # Generate signal
            signal = self.get_live_signal(pair, tf)
            if signal is None:
                continue

            results["signals"] += 1

            # Validate
            signal = self.validate_signal(signal)
            self.signals.append(signal)

            if signal.decision == "accepted":
                results["accepted"] += 1
                trade = self.execute_signal(signal)
                if trade:
                    results["trades"] += 1
            else:
                results["rejected"] += 1

        return results

    def get_validation_report(self) -> Dict:
        """Generate live validation report."""
        # Compute live metrics
        closed_trades = [t for t in self.trades if t.status == "closed"]
        open_trades = [t for t in self.trades if t.status == "open"]

        if closed_trades:
            pnls = [t.pnl_net for t in closed_trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            win_rate = len(wins) / len(pnls) if pnls else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf')
            expectancy = np.mean(pnls) if pnls else 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = expectancy = 0

        n_days = (datetime.now() - datetime.strptime(self.start_date, "%Y-%m-%d")).days or 1

        live_results = {
            "n_trades": len(closed_trades),
            "n_open": len(open_trades),
            "win_rate": round(win_rate, 3),
            "avg_win_pct": round(avg_win * 100, 3),
            "avg_loss_pct": round(avg_loss * 100, 3),
            "profit_factor": round(profit_factor, 2),
            "expectancy_pct": round(expectancy * 100, 3),
            "max_drawdown_pct": round((1 - self.capital / self.peak_capital) * 100, 2),
            "avg_slippage_bps": round(np.mean([t.slippage_total_bps for t in closed_trades]), 2) if closed_trades else 0,
        }

        # Load baseline
        baseline = Baseline.load()
        oos = baseline.oos_results if baseline else {}

        # Verdict
        verdict, reason = VerdictEngine.evaluate(live_results, oos, n_days)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_days": n_days,
            "capital": round(self.capital, 2),
            "live_results": live_results,
            "oos_baseline": oos,
            "verdict": verdict.value,
            "verdict_reason": reason,
        }


if __name__ == "__main__":
    engine = LivePaperEngine()
    print("Running live paper trading iteration...")
    results = engine.run_once("1h")
    print(f"Results: {results}")

    report = engine.get_validation_report()
    print(f"\nValidation Report:")
    print(f"  Verdict: {report['verdict']}")
    print(f"  Reason: {report['verdict_reason']}")
    print(f"  Live trades: {report['live_results']['n_trades']}")
    print(f"  Win rate: {report['live_results']['win_rate']:.0%}")
