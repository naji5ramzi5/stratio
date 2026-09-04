"""Multi-Period Robustness Test.
Runs the EXACT SAME strategy over multiple historical periods.
No parameter changes between periods."""
from __future__ import annotations

import json
import os
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict

DATA_DIR = "historical_data"


class MultiPeriodTester:
    """Tests the same strategy across multiple historical periods."""

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.data: Dict[str, np.ndarray] = {}
        self.timestamps: Dict[str, np.ndarray] = {}
        self._load_data()

    def _load_data(self):
        for f in os.listdir(DATA_DIR):
            if f.endswith(".json") and f != "metadata.json":
                key = f.replace(".json", "")
                with open(os.path.join(DATA_DIR, f)) as fh:
                    raw = json.load(fh)
                arr = np.array([[c[0], c[1], c[2], c[3], c[4], c[5]] for c in raw], dtype=float)
                self.data[key] = arr
                self.timestamps[key] = arr[:, 0]

    def get_candles(self, symbol: str, end_ts: int, limit: int = 500) -> np.ndarray:
        key = f"{symbol}_1h"
        if key not in self.data:
            return np.array([])
        mask = self.timestamps[key] <= end_ts
        candles = self.data[key][mask]
        return candles[-limit:]

    def run_period(self, start_date: str, end_date: str, period_name: str) -> Dict:
        """Run the exact same strategy for one period."""
        print(f"\n{'='*60}")
        print(f"Period: {period_name} ({start_date} to {end_date})")
        print(f"{'='*60}")

        capital = self.initial_capital
        peak_capital = capital
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000) + 86400000

        btc_key = "BTCUSDT_1h"
        if btc_key not in self.timestamps:
            return {"error": "BTC data not available"}

        btc_ts = self.timestamps[btc_key]
        period_mask = (btc_ts >= start_ts) & (btc_ts <= end_ts)
        sim_timestamps = btc_ts[period_mask]

        if len(sim_timestamps) == 0:
            return {"error": "No data for this period"}

        # SAME pairs as before - NO CHANGES
        pairs = [("BTCUSDT", "ETHUSDT"), ("SOLUSDT", "XRPUSDT"), ("ADAUSDT", "LTCUSDT")]

        # SAME parameters - NO CHANGES
        z_entry = 2.0
        z_exit_profit = 0.001  # 0.1% pnl
        z_exit_loss = -0.005   # 0.5% stop loss
        max_hold_bars = 24
        position_size_pct = 0.03
        fee_rate = 0.002       # 0.2% round trip
        slippage_rate = 0.0005 # 5 bps

        open_positions = {}
        closed_trades = []
        equity_curve = []
        regime_counts = {"MEAN_REVERTING": 0, "TRENDING": 0, "VOLATILE": 0, "OTHER": 0}

        for i, current_ts in enumerate(sim_timestamps):
            # Manage exits
            to_close = []
            for pair, pos in open_positions.items():
                pos["bars_held"] += 1
                sym_a, sym_b = pair.split("/")
                ca = self.get_candles(sym_a, current_ts, 10)
                cb = self.get_candles(sym_b, current_ts, 10)

                if len(ca) < 2 or len(cb) < 2:
                    continue

                curr_a = float(ca[-1, 4])
                curr_b = float(cb[-1, 4])
                entry_a = pos["entry_price_a"]
                entry_b = pos["entry_price_b"]
                direction = pos["direction"]

                pnl_pct = direction * ((curr_a - entry_a) / entry_a - (curr_b - entry_b) / entry_b)

                if pnl_pct > z_exit_profit or pnl_pct < z_exit_loss or pos["bars_held"] >= max_hold_bars:
                    gross_pnl = pnl_pct * pos["size_usd"]
                    fees = pos["size_usd"] * fee_rate
                    slippage = pos["size_usd"] * slippage_rate
                    net_pnl = gross_pnl - fees - slippage
                    capital += net_pnl

                    closed_trades.append({
                        "pair": pair,
                        "entry_ts": pos["entry_ts"],
                        "exit_ts": current_ts,
                        "entry_a": entry_a, "entry_b": entry_b,
                        "exit_a": curr_a, "exit_b": curr_b,
                        "pnl_pct": round(pnl_pct * 100, 3),
                        "pnl_gross": round(gross_pnl, 4),
                        "pnl_net": round(net_pnl, 4),
                        "fees": round(fees + slippage, 2),
                        "exit_reason": "profit" if pnl_pct > z_exit_profit else ("stoploss" if pnl_pct < z_exit_loss else "time"),
                        "bars_held": pos["bars_held"],
                    })
                    to_close.append(pair)

            for p in to_close:
                del open_positions[p]

            # Generate signals
            for sym_a, sym_b in pairs:
                pair = f"{sym_a}/{sym_b}"
                if pair in open_positions:
                    continue

                ca = self.get_candles(sym_a, current_ts, 500)
                cb = self.get_candles(sym_b, current_ts, 500)
                if len(ca) < 200 or len(cb) < 200:
                    continue

                from stat_arb import ols_hedge_ratio, spread_series
                h = ols_hedge_ratio(ca[:200, 4], cb[:200, 4])
                spread_train = spread_series(ca[:200, 4], cb[:200, 4], h)
                mu, sd = float(np.mean(spread_train)), float(np.std(spread_train))
                if sd <= 0:
                    continue

                current_spread = float(np.log(ca[-1, 4]) - h * np.log(cb[-1, 4]))
                z = (current_spread - mu) / sd

                if abs(z) >= z_entry:
                    direction = 1 if z < 0 else -1
                    open_positions[pair] = {
                        "pair": pair, "direction": direction,
                        "entry_ts": current_ts,
                        "entry_price_a": float(ca[-1, 4]),
                        "entry_price_b": float(cb[-1, 4]),
                        "entry_z": round(z, 3),
                        "size_usd": position_size_pct * capital,
                        "bars_held": 0,
                    }

            # Record equity
            unrealized = 0.0
            for pair, pos in open_positions.items():
                sym_a, sym_b = pair.split("/")
                ca = self.get_candles(sym_a, current_ts, 5)
                cb = self.get_candles(sym_b, current_ts, 5)
                if len(ca) > 1 and len(cb) > 1:
                    curr_a = float(ca[-1, 4])
                    curr_b = float(cb[-1, 4])
                    change_a = (curr_a - pos["entry_price_a"]) / pos["entry_price_a"]
                    change_b = (curr_b - pos["entry_price_b"]) / pos["entry_price_b"]
                    unrealized += pos["direction"] * (change_a - change_b) * pos["size_usd"]

            equity = capital + unrealized
            if equity > peak_capital:
                peak_capital = equity
            dd = (equity - peak_capital) / peak_capital * 100 if peak_capital > 0 else 0

            equity_curve.append({"ts": current_ts, "equity": equity, "dd": dd})

        # Calculate metrics
        if closed_trades:
            wins = [t for t in closed_trades if t["pnl_pct"] > 0]
            losses = [t for t in closed_trades if t["pnl_pct"] <= 0]
            win_rate = len(wins) / len(closed_trades) if closed_trades else 0
            gross_profit = sum(t["pnl_gross"] for t in closed_trades if t["pnl_gross"] > 0)
            gross_loss = abs(sum(t["pnl_gross"] for t in closed_trades if t["pnl_gross"] <= 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
            avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0
            total_fees = sum(t["fees"] for t in closed_trades)
        else:
            win_rate = 0
            gross_profit = gross_loss = 0
            profit_factor = 0
            avg_win = avg_loss = 0
            total_fees = 0
            wins = []
            losses = []

        total_return = (capital - self.initial_capital) / self.initial_capital * 100
        max_dd = min(e["dd"] for e in equity_curve) if equity_curve else 0
        days = len(sim_timestamps) / 24  # approximate calendar days
        trades_per_day = len(closed_trades) / days if days > 0 else 0

        return {
            "period_name": period_name,
            "start": start_date,
            "end": end_date,
            "days": round(days, 1),
            "total_trades": len(closed_trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 3),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "total_fees": round(total_fees, 2),
            "net_pnl": round(capital - self.initial_capital, 2),
            "return_pct": round(total_return, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "avg_win_pct": round(avg_win, 3),
            "avg_loss_pct": round(avg_loss, 3),
            "trades_per_day": round(trades_per_day, 2),
            "final_capital": round(capital, 2),
        }


def main():
    tester = MultiPeriodTester(initial_capital=10000.0)

    # Test periods - predefined, no cherry-picking
    periods = [
        ("2026-06-29", "2026-07-05", "Period_1_Early_July"),
        ("2026-07-06", "2026-07-12", "Period_2_Mid_July"),
        ("2026-07-13", "2026-07-19", "Period_3_Late_July"),
        ("2026-07-20", "2026-07-26", "Period_4_End_July"),
        ("2026-07-27", "2026-08-02", "Period_5_Early_Aug"),
        ("2026-08-03", "2026-08-07", "Period_6_Late_Aug"),
    ]

    results = []
    for start, end, name in periods:
        result = tester.run_period(start, end, name)
        results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("MULTI-PERIOD ROBUSTNESS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Period':<25} {'Trades':>6} {'WR':>6} {'Return':>8} {'PF':>6} {'MaxDD':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['period_name']:<25} {r['total_trades']:>6} {r['win_rate']:>6.1%} {r['return_pct']:>+7.2f}% {r['profit_factor']:>6.2f} {r['max_drawdown_pct']:>7.2f}%")

    # Classification
    profitable = sum(1 for r in results if r["return_pct"] > 0)
    losing = sum(1 for r in results if r["return_pct"] <= 0)
    avg_return = np.mean([r["return_pct"] for r in results])
    avg_pf = np.mean([r["profit_factor"] for r in results])
    avg_dd = np.mean([r["max_drawdown_pct"] for r in results])

    print(f"\n{'='*80}")
    print(f"Profitable periods: {profitable}/{len(results)}")
    print(f"Losing periods: {losing}/{len(results)}")
    print(f"Average return: {avg_return:+.2f}%")
    print(f"Average Profit Factor: {avg_pf:.2f}")
    print(f"Average Max Drawdown: {avg_dd:.2f}%")

    if profitable == len(results):
        classification = "CONSISTENTLY PROFITABLE"
    elif profitable > losing:
        classification = "MOSTLY PROFITABLE"
    elif profitable > 0:
        classification = "REGIME DEPENDENT or INCONSISTENT"
    else:
        classification = "CONSISTENTLY UNPROFITABLE"

    print(f"\nCLASSIFICATION: {classification}")
    print(f"{'='*80}")

    # Save report
    with open("multi_period_report.json", "w") as f:
        json.dump({"results": results, "classification": classification}, f, indent=2)


if __name__ == "__main__":
    main()
