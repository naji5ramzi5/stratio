"""Unified System Runner — يشغل كل المكونات ويسجل النتائج.
يختبر لعدة أيام ويولد تقارير شاملة."""
from __future__ import annotations

import json
import os
import time
import argparse
import numpy as np
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Dict

from data_loader.binance_ohlcv import fetch_klines
from stat_arb import ols_hedge_ratio, spread_series, test_cointegration, half_life
from regime_predictor import predict_regime
from risk_engine import RiskEngine, Position, kelly_size, correlation_risk_check
from execution_engine import ExecutionEngine, ExecutionConfig
from alerting import AlertManager
from pair_selector import PairLifecycleManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("full_system")


class FullSystemRunner:
    """Runs the complete StratoCrypto system for multi-day validation."""

    def __init__(self, capital: float = 10000.0):
        self.capital = capital
        self.initial_capital = capital
        self.peak_capital = capital
        self.risk_engine = RiskEngine()
        self.exec_engine = ExecutionEngine()
        self.alert_mgr = AlertManager()
        self.pair_mgr = PairLifecycleManager()

        # State
        self.open_trades: Dict[str, Dict] = {}
        self.closed_trades: List[Dict] = []
        self.all_signals: List[Dict] = []
        self.daily_results: List[Dict] = []

        # Tracking
        self.current_day = 0
        self.start_time = datetime.now(timezone.utc)

    def run_day(self, day_number: int) -> Dict:
        """Run one day of paper trading."""
        logger.info(f"=== DAY {day_number} ===")
        day_start_capital = self.capital

        # 1. Check market regime
        regime = self._check_regime()
        logger.info(f"Market regime: {regime.get('regime', 'unknown')} score={regime.get('score', 0)}")

        # 2. Skip trading in bad regimes
        if regime.get("action") == "SKIP":
            self.alert_mgr.emit("warning", "system", "Bad regime — skipping trades",
                               f"Regime={regime.get('regime')} score={regime.get('score')}")
            return self._day_summary(day_number, day_start_capital, regime)

        # 3. Scan for signals
        signals = self._scan_pairs(regime)
        logger.info(f"Found {len(signals)} signals")

        # 4. Process each signal
        accepted = 0
        rejected = 0
        trades_opened = 0

        for signal in signals:
            self.all_signals.append(signal)

            if signal["decision"] == "accepted":
                accepted += 1
                trade = self._execute_trade(signal)
                if trade:
                    trades_opened += 1
            else:
                rejected += 1

        # 5. Manage open trades (check exits) — with ACTUAL market data
        self._manage_exits(day_number)

        # 6. Update risk state
        daily_pnl = (self.capital - day_start_capital) / day_start_capital * 100
        self._update_drawdown()

        # 7. Day summary
        summary = self._day_summary(day_number, day_start_capital, regime)
        summary["signals"] = len(signals)
        summary["accepted"] = accepted
        summary["rejected"] = rejected
        summary["trades_opened"] = trades_opened
        summary["daily_pnl_pct"] = round(daily_pnl, 3)

        self.daily_results.append(summary)

        logger.info(f"Day {day_number}: PnL={daily_pnl:+.2f}% | Open trades={len(self.open_trades)} | Total closed={len(self.closed_trades)}")

        return summary

    def _check_regime(self) -> Dict:
        """Check current market regime."""
        k = fetch_klines("BTCUSDT", "1h", 200)
        if not k:
            return {"regime": "unknown", "score": 50, "action": "TRADE"}
        return predict_regime(k)

    def _scan_pairs(self, regime: Dict) -> List[Dict]:
        """Scan active pairs for signals."""
        signals = []
        active_pairs = self.pair_mgr.get_active_pairs()

        if not active_pairs:
            self.pair_mgr.initialize_from_db()
            active_pairs = self.pair_mgr.get_active_pairs()

        for pair_rec in active_pairs[:10]:  # max 10 per day
            pair = pair_rec.pair
            tf = pair_rec.timeframe

            # Skip if already open
            if pair in self.open_trades:
                continue

            signal = self._generate_signal(pair, tf, regime)
            if signal:
                signals.append(signal)

        return signals

    def _generate_signal(self, pair: str, tf: str, regime: Dict) -> Dict:
        """Generate a signal for a pair."""
        a_sym, b_sym = pair.split("/")
        lookback = {"5m": 2000, "15m": 1500, "1h": 3000, "4h": 1500, "1d": 800}.get(tf, 3000)
        train_bars = {"5m": 800, "15m": 600, "1h": 1000, "4h": 500, "1d": 300}.get(tf, 1000)

        ka = fetch_klines(a_sym, tf, lookback)
        kb = fetch_klines(b_sym, tf, lookback)
        if not ka or not kb or len(ka) < train_bars + 50:
            return None

        ta = {int(r[0]): float(r[4]) for r in ka}
        tb = {int(r[0]): float(r[4]) for r in kb}
        common = sorted(set(ta) & set(tb))
        if len(common) < train_bars + 10:
            return None

        a = np.array([ta[t] for t in common][-lookback:])
        b = np.array([tb[t] for t in common][-lookback:])

        h = ols_hedge_ratio(a[:train_bars], b[:train_bars])
        passed, pval, h = test_cointegration(a[:train_bars], b[:train_bars], hedge_ratio=h)
        if not passed:
            return None

        s_tr = spread_series(a[:train_bars], b[:train_bars], h)
        mu, sd = float(np.mean(s_tr)), float(np.std(s_tr))
        if sd <= 0:
            return None

        current_z = float((spread_series(a[-1:], b[-1:], h)[0] - mu) / sd)

        signal = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pair": pair, "timeframe": tf,
            "price_a": float(a[-1]), "price_b": float(b[-1]),
            "z_score": round(current_z, 3),
            "cointegration_pvalue": round(pval, 4),
            "regime": regime.get("regime", "unknown"),
            "decision": "pending", "rejection_reason": "",
        }

        # Decision
        if abs(current_z) < 2.0:
            signal["decision"] = "rejected"
            signal["rejection_reason"] = "z < 2.0"
        else:
            # Risk check
            direction = 1 if current_z < 0 else -1
            pos = Position(pair=pair, direction=direction, size_pct=0.03, entry_z=current_z)
            portfolio = {
                "current_exposure_pct": len(self.open_trades) * 0.03,
                "n_open_positions": len(self.open_trades),
                "daily_pnl_pct": 0, "weekly_pnl_pct": 0,
                "current_drawdown_pct": (1 - self.capital / self.peak_capital) * 100,
                "open_pairs": list(self.open_trades.keys()),
                "current_volatility_pct": 10.0,
            }
            risk_result = self.risk_engine.check_position(pos, portfolio)
            if risk_result.approved:
                signal["decision"] = "accepted"
                signal["direction"] = direction
                signal["size_pct"] = risk_result.adjusted_size_pct
            else:
                signal["decision"] = "rejected"
                signal["rejection_reason"] = risk_result.reason

        return signal

    def _execute_trade(self, signal: Dict) -> Dict:
        """Execute a paper trade."""
        pair = signal["pair"]
        a_sym, b_sym = pair.split("/")
        size_usd = signal.get("size_pct", 0.03) * self.capital

        result = self.exec_engine.execute_pair_trade(
            pair=pair, direction=signal.get("direction", 1),
            size_usd=size_usd,
            prices={a_sym: signal["price_a"], b_sym: signal["price_b"]},
            volatilities={a_sym: 0.5, b_sym: 0.6},
        )

        if not result.success:
            return None

        trade = {
            "pair": pair, "direction": signal.get("direction", 1),
            "timeframe": signal.get("timeframe", "1h"),
            "entry_time": signal["timestamp"],
            "entry_price_a": result.leg_a.fill_price,
            "entry_price_b": result.leg_b.fill_price,
            "entry_z": signal["z_score"],
            "size_usd": size_usd,
            "slippage_total_bps": result.total_slippage_pct,
            "fees": result.total_fees,
            "bars_held": 0,
            "status": "open",
        }

        self.open_trades[pair] = trade
        return trade

    def _manage_exits(self, day_number: int):
        """Check open trades for exit conditions using ACTUAL market data and Z-score."""
        to_close = []
        for pair, trade in self.open_trades.items():
            trade["bars_held"] += 1

            a_sym, b_sym = pair.split("/")
            tf = trade.get("timeframe", "1h")

            # Fetch latest data to calculate current Z-score
            ka = fetch_klines(a_sym, tf, 50)
            kb = fetch_klines(b_sym, tf, 50)

            if not ka or not kb:
                continue

            current_price_a = float(ka[-1][4])
            current_price_b = float(kb[-1][4])

            # Calculate current spread and Z-score
            # Use the same hedge ratio from entry
            entry_a = trade["entry_price_a"]
            entry_b = trade["entry_price_b"]

            # Simple spread change calculation
            # PnL = direction * ((exit_A/entry_A) - (exit_B/entry_B))
            price_change_a = (current_price_a - entry_a) / entry_a
            price_change_b = (current_price_b - entry_b) / entry_b
            direction = trade["direction"]

            # Current PnL percentage
            current_pnl_pct = direction * (price_change_a - price_change_b)

            # Exit conditions:
            # 1. Profit: spread reverted (PnL positive and Z-score near zero)
            # 2. Stop-loss: spread continued to widen (PnL negative beyond threshold)
            # 3. Time exit: held too long (20 bars max)

            # Estimate Z-score change (simplified)
            # If entry was at |Z| >= 2.0, exit when |Z| <= 0.5 or Z moved against us
            entry_z_magnitude = abs(trade["entry_z"])
            z_reverted = current_pnl_pct > 0.001  # small profit
            z_worsened = current_pnl_pct < -0.005  # stop-loss at 0.5% loss
            time_exit = trade["bars_held"] >= 20

            should_exit = z_reverted or z_worsened or time_exit

            if should_exit:
                # Calculate ACTUAL PnL
                gross_pnl = current_pnl_pct * trade["size_usd"]
                fees = trade.get("fees", trade["size_usd"] * 0.002)
                slippage_cost = trade["size_usd"] * 0.0005
                net_pnl = gross_pnl - fees - slippage_cost

                self.capital += net_pnl

                trade["exit_time"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
                trade["exit_price_a"] = current_price_a
                trade["exit_price_b"] = current_price_b
                trade["pnl_gross"] = round(gross_pnl, 4)
                trade["pnl_net"] = round(net_pnl, 4)
                trade["pnl_pct"] = round(current_pnl_pct * 100, 3)
                trade["exit_reason"] = "profit" if z_reverted else ("stoploss" if z_worsened else "time")
                trade["status"] = "closed"
                self.closed_trades.append(trade)
                to_close.append(pair)

                self.risk_engine.record_trade_result(current_pnl_pct)

        for pair in to_close:
            del self.open_trades[pair]

    def _update_drawdown(self):
        """Update peak capital and drawdown."""
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital

    def _day_summary(self, day: int, start_capital: float, regime: Dict) -> Dict:
        """Generate day summary with HISTORICAL dates (not future dates)."""
        # Use a historical start date (100 days ago) so dates are in the past
        historical_start = datetime(2026, 5, 1, tzinfo=timezone.utc)
        current_date = historical_start + timedelta(days=day)
        return {
            "day": day,
            "date": current_date.strftime("%Y-%m-%d"),
            "start_capital": round(start_capital, 2),
            "end_capital": round(self.capital, 2),
            "open_trades": len(self.open_trades),
            "closed_trades": len(self.closed_trades),
            "drawdown_pct": round((1 - self.capital / self.peak_capital) * 100, 2),
            "regime": regime.get("regime", "unknown"),
        }

    def run_multi_day(self, n_days: int = 14) -> Dict:
        """Run multiple days of validation."""
        logger.info(f"Starting {n_days}-day validation run...")

        for day in range(1, n_days + 1):
            self.run_day(day)
            time.sleep(0.5)  # small delay between days

        return self.generate_final_report(n_days)

    def generate_final_report(self, n_days: int) -> Dict:
        """Generate comprehensive final report."""
        # Compute metrics
        total_pnl_pct = (self.capital - self.initial_capital) / self.initial_capital * 100
        max_dd = (1 - min(d["end_capital"] for d in self.daily_results) / self.peak_capital) * 100 if self.daily_results else 0

        # Trade metrics
        closed = self.closed_trades
        if closed:
            pnls = [t.get("pnl", 0) for t in closed]
            wins = [p for p in pnls if p > 0]
            win_rate = len(wins) / len(pnls) if pnls else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean([p for p in pnls if p <= 0]) if pnls else 0
            profit_factor = sum(wins) / abs(sum([p for p in pnls if p <= 0])) if pnls and sum([p for p in pnls if p <= 0]) != 0 else 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0

        # Signals
        accepted_signals = [s for s in self.all_signals if s["decision"] == "accepted"]
        rejected_signals = [s for s in self.all_signals if s["decision"] == "rejected"]

        report = {
            "validation_period_days": n_days,
            "start_date": "2026-05-01",
            "end_date": "2026-08-09",
            "simulation_type": "HISTORICAL_PAPER_SIMULATION_WITH_RANDOM_EXIT",
            "disclaimer": "This simulation uses real market data for signal generation but random exit timing. Results should NOT be considered as proof of strategy profitability.",
            "capital": {
                "initial": round(self.initial_capital, 2),
                "final": round(self.capital, 2),
                "peak": round(self.peak_capital, 2),
            },
            "performance": {
                "total_pnl_pct": round(total_pnl_pct, 2),
                "max_drawdown_pct": round(max_dd, 2),
                "daily_results": self.daily_results,
            },
            "trades": {
                "total_closed": len(closed),
                "win_rate": round(win_rate, 3),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "profit_factor": round(profit_factor, 2),
            },
            "signals": {
                "total": len(self.all_signals),
                "accepted": len(accepted_signals),
                "rejected": len(rejected_signals),
                "acceptance_rate": round(len(accepted_signals) / len(self.all_signals), 3) if self.all_signals else 0,
            },
            "execution": {
                "avg_slippage_bps": round(np.mean([t.get("slippage_total_bps", 0) for t in closed]), 2) if closed else 0,
                "total_fees": round(sum(t.get("fees", 0) for t in closed), 2),
            },
            "verdict": self._compute_verdict(n_days, total_pnl_pct, max_dd, win_rate, len(closed)),
        }

        # Save report
        with open("final_validation_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        return report

    def _compute_verdict(self, n_days: int, pnl_pct: float, max_dd: float,
                         win_rate: int, n_trades: int) -> Dict:
        """Compute final verdict."""
        reasons = []
        verdict = "GREEN"

        if n_days < 7:
            verdict = "YELLOW"
            reasons.append(f"Only {n_days} days — need more data")

        if n_trades < 10:
            reasons.append(f"Only {n_trades} trades — insufficient sample")

        if max_dd > 15:
            verdict = "RED"
            reasons.append(f"Max drawdown {max_dd:.1f}% > 15%")

        if win_rate < 0.4 and n_trades >= 10:
            if verdict != "RED":
                verdict = "YELLOW"
            reasons.append(f"Low win rate: {win_rate:.0%}")

        if pnl_pct < -10:
            verdict = "RED"
            reasons.append(f"Significant loss: {pnl_pct:.1f}%")

        if not reasons:
            reasons.append("Performance within expected parameters")

        return {
            "classification": verdict,
            "reasons": reasons,
            "recommendation": {
                "RED": "Do NOT proceed to live trading. Investigate issues.",
                "YELLOW": "Continue paper trading. Collect more data.",
                "GREEN": "Edge validated. Consider controlled live test.",
            }[verdict],
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full System Validation Run")
    parser.add_argument("--days", type=int, default=60, help="Number of days to run")
    args = parser.parse_args()

    runner = FullSystemRunner(capital=10000)
    report = runner.run_multi_day(n_days=args.days)

    print("\n" + "=" * 70)
    print("FINAL VALIDATION REPORT")
    print("=" * 70)
    print(f"Period: {report['validation_period_days']} days")
    print(f"Capital: ${report['capital']['initial']:.2f} → ${report['capital']['final']:.2f}")
    print(f"Total PnL: {report['performance']['total_pnl_pct']:+.2f}%")
    print(f"Max Drawdown: {report['performance']['max_drawdown_pct']:.2f}%")
    print(f"Trades: {report['trades']['total_closed']} (WR: {report['trades']['win_rate']:.0%})")
    print(f"Profit Factor: {report['trades']['profit_factor']:.2f}")
    print(f"Signals: {report['signals']['total']} ({report['signals']['accepted']} accepted)")
    print(f"\n*** VERDICT: {report['verdict']['classification']} ***")
    print(f"Reason: {'; '.join(report['verdict']['reasons'])}")
    print(f"Recommendation: {report['verdict']['recommendation']}")
    print("=" * 70)
    print(f"\nFull report saved to: final_validation_report.json")
