"""Fast Historical Backtest Engine - Optimized for speed.
Loads all data into memory once, then simulates quickly."""
from __future__ import annotations

import json
import os
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict

DATA_DIR = "historical_data"


class FastHistoricalBacktester:
    """Optimized historical backtest - loads data once, simulates fast."""

    def __init__(self, start_date: str, end_date: str, initial_capital: float = 10000.0):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.peak_capital = initial_capital

        # Load all data into memory once
        self.data: Dict[str, np.ndarray] = {}
        self.timestamps: Dict[str, np.ndarray] = {}
        self._load_data()

        # Simulation period
        self.start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
        self.end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000) + 86400000

        # State
        self.open_positions: Dict[str, Dict] = {}
        self.closed_trades: List[Dict] = []
        self.equity_curve: List[Dict] = []

    def _load_data(self):
        """Load all historical data into numpy arrays."""
        for f in os.listdir(DATA_DIR):
            if f.endswith(".json") and f != "metadata.json":
                key = f.replace(".json", "")
                with open(os.path.join(DATA_DIR, f)) as fh:
                    raw = json.load(fh)
                # Convert to numpy: [timestamp, open, high, low, close, volume]
                arr = np.array([[c[0], c[1], c[2], c[3], c[4], c[5]] for c in raw], dtype=float)
                self.data[key] = arr
                self.timestamps[key] = arr[:, 0]

    def get_candles_up_to(self, symbol: str, end_ts: int, limit: int = 500) -> np.ndarray:
        """Get candles up to end_ts (point-in-time safe)."""
        key = f"{symbol}_1h"
        if key not in self.data:
            return np.array([])
        mask = self.timestamps[key] <= end_ts
        candles = self.data[key][mask]
        return candles[-limit:]

    def run(self) -> Dict:
        """Run the backtest."""
        print(f"Running backtest: {self.start_date} to {self.end_date}")

        # Get common timestamps from BTC
        btc_key = "BTCUSDT_1h"
        if btc_key not in self.timestamps:
            return {"error": "BTC data not available"}

        # Filter timestamps in our period
        btc_ts = self.timestamps[btc_key]
        period_mask = (btc_ts >= self.start_ts) & (btc_ts <= self.end_ts)
        sim_timestamps = btc_ts[period_mask]

        print(f"Simulating {len(sim_timestamps)} timestamps...")

        pairs_to_check = [
            ("BTCUSDT", "ETHUSDT"),
            ("SOLUSDT", "XRPUSDT"),
            ("ADAUSDT", "LTCUSDT"),
        ]

        for i, current_ts in enumerate(sim_timestamps):
            if i % 50 == 0:
                print(f"  Progress: {i}/{len(sim_timestamps)} ({datetime.fromtimestamp(current_ts/1000).strftime('%Y-%m-%d')})", flush=True)

            # 1. Manage exits
            self._manage_exits(current_ts)

            # 2. Generate signals
            for sym_a, sym_b in pairs_to_check:
                pair = f"{sym_a}/{sym_b}"
                if pair in self.open_positions:
                    continue

                signal = self._generate_signal(sym_a, sym_b, current_ts)
                if signal:
                    self._execute_trade(signal, current_ts)

            # 3. Record equity
            self._record_equity(current_ts)

        return self._generate_report()

    def _generate_signal(self, sym_a: str, sym_b: str, current_ts: int) -> Dict:
        """Generate signal using only data up to current_ts."""
        candles_a = self.get_candles_up_to(sym_a, current_ts, 500)
        candles_b = self.get_candles_up_to(sym_b, current_ts, 500)

        if len(candles_a) < 200 or len(candles_b) < 200:
            return None

        closes_a = candles_a[:, 4]  # close price
        closes_b = candles_b[:, 4]

        # Hedge ratio (train on first 200 bars)
        from stat_arb import ols_hedge_ratio, spread_series
        h = ols_hedge_ratio(closes_a[:200], closes_b[:200])
        spread_train = spread_series(closes_a[:200], closes_b[:200], h)
        mu, sd = float(np.mean(spread_train)), float(np.std(spread_train))

        if sd <= 0:
            return None

        # Current Z-score
        current_spread = float(np.log(closes_a[-1]) - h * np.log(closes_b[-1]))
        z = (current_spread - mu) / sd

        if abs(z) >= 2.0:
            return {
                "timestamp": current_ts,
                "pair": f"{sym_a}/{sym_b}",
                "direction": 1 if z < 0 else -1,
                "z_score": round(z, 3),
                "price_a": float(closes_a[-1]),
                "price_b": float(closes_b[-1]),
            }
        return None

    def _execute_trade(self, signal: Dict, current_ts: int):
        """Execute trade."""
        pair = signal["pair"]
        self.open_positions[pair] = {
            "pair": pair,
            "direction": signal["direction"],
            "entry_ts": current_ts,
            "entry_price_a": signal["price_a"],
            "entry_price_b": signal["price_b"],
            "entry_z": signal["z_score"],
            "size_usd": 0.03 * self.capital,
            "bars_held": 0,
            "status": "open",
        }

    def _manage_exits(self, current_ts: int):
        """Check exits using actual prices."""
        to_close = []

        for pair, pos in self.open_positions.items():
            pos["bars_held"] += 1

            sym_a, sym_b = pair.split("/")
            candles_a = self.get_candles_up_to(sym_a, current_ts, 10)
            candles_b = self.get_candles_up_to(sym_b, current_ts, 10)

            if len(candles_a) < 2 or len(candles_b) < 2:
                continue

            curr_a = float(candles_a[-1, 4])
            curr_b = float(candles_b[-1, 4])

            # PnL calculation
            entry_a = pos["entry_price_a"]
            entry_b = pos["entry_price_b"]
            direction = pos["direction"]

            price_change_a = (curr_a - entry_a) / entry_a
            price_change_b = (curr_b - entry_b) / entry_b
            pnl_pct = direction * (price_change_a - price_change_b)

            # Exit conditions
            exit_profit = pnl_pct > 0.001
            exit_loss = pnl_pct < -0.005
            exit_time = pos["bars_held"] >= 24

            if exit_profit or exit_loss or exit_time:
                gross_pnl = pnl_pct * pos["size_usd"]
                fees = pos["size_usd"] * 0.002  # 0.2% round trip
                slippage = pos["size_usd"] * 0.0005  # 5 bps
                net_pnl = gross_pnl - fees - slippage

                self.capital += net_pnl

                pos["exit_ts"] = current_ts
                pos["exit_price_a"] = curr_a
                pos["exit_price_b"] = curr_b
                pos["pnl_gross"] = round(gross_pnl, 4)
                pos["pnl_net"] = round(net_pnl, 4)
                pos["pnl_pct"] = round(pnl_pct * 100, 3)
                pos["exit_reason"] = "profit" if exit_profit else ("stoploss" if exit_loss else "time")
                pos["fees"] = round(fees + slippage, 2)
                pos["status"] = "closed"

                self.closed_trades.append(pos)
                to_close.append(pair)

        for pair in to_close:
            del self.open_positions[pair]

    def _record_equity(self, current_ts: int):
        """Record equity."""
        unrealized = 0.0
        for pair, pos in self.open_positions.items():
            sym_a, sym_b = pair.split("/")
            candles_a = self.get_candles_up_to(sym_a, current_ts, 5)
            candles_b = self.get_candles_up_to(sym_b, current_ts, 5)
            if len(candles_a) > 1 and len(candles_b) > 1:
                curr_a = float(candles_a[-1, 4])
                curr_b = float(candles_b[-1, 4])
                change_a = (curr_a - pos["entry_price_a"]) / pos["entry_price_a"]
                change_b = (curr_b - pos["entry_price_b"]) / pos["entry_price_b"]
                unrealized += pos["direction"] * (change_a - change_b) * pos["size_usd"]

        equity = self.capital + unrealized
        if equity > self.peak_capital:
            self.peak_capital = equity

        dd = (equity - self.peak_capital) / self.peak_capital * 100 if self.peak_capital > 0 else 0

        self.equity_curve.append({
            "timestamp": current_ts,
            "date": datetime.fromtimestamp(current_ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "cash": round(self.capital, 2),
            "unrealized": round(unrealized, 2),
            "equity": round(equity, 2),
            "peak": round(self.peak_capital, 2),
            "drawdown_pct": round(dd, 2),
        })

    def _generate_report(self) -> Dict:
        """Generate final report."""
        closed = self.closed_trades
        if closed:
            pnls = [t["pnl_pct"] for t in closed]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            win_rate = len(wins) / len(pnls) if pnls else 0
            gross_profit = sum(t["pnl_gross"] for t in closed if t["pnl_gross"] > 0)
            gross_loss = abs(sum(t["pnl_gross"] for t in closed if t["pnl_gross"] <= 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
        else:
            win_rate = 0
            gross_profit = gross_loss = 0
            profit_factor = 0
            avg_win = avg_loss = 0
            wins = []
            losses = []

        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        max_dd = min(e["drawdown_pct"] for e in self.equity_curve) if self.equity_curve else 0

        report = {
            "run_id": f"FAST_HISTORICAL_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}",
            "simulation_type": "REAL_HISTORICAL_BACKTEST",
            "data_source": "Binance Historical OHLCV",
            "data_period": f"{self.start_date} to {self.end_date}",
            "timeframe": "1h",
            "initial_capital": self.initial_capital,
            "final_equity": round(self.capital, 2),
            "total_return_pct": round(total_return, 2),
            "max_drawdown_pct": round(max_dd, 2),
            "total_trades": len(closed),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate, 3),
            "profit_factor": round(profit_factor, 2),
            "avg_win_pct": round(avg_win, 3),
            "avg_loss_pct": round(avg_loss, 3),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "total_fees": round(sum(t.get("fees", 0) for t in closed), 2),
            "closed_trades": closed,
            "equity_curve_points": len(self.equity_curve),
        }

        with open("fast_historical_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        return report


if __name__ == "__main__":
    backtester = FastHistoricalBacktester(
        start_date="2026-06-29",
        end_date="2026-08-07",
        initial_capital=10000.0,
    )
    report = backtester.run()

    print("\n" + "=" * 60)
    print("FAST HISTORICAL BACKTEST REPORT")
    print("=" * 60)
    print(f"Period: {report['data_period']}")
    print(f"Return: {report['total_return_pct']}%")
    print(f"Max Drawdown: {report['max_drawdown_pct']}%")
    print(f"Trades: {report['total_trades']} (W:{report['winning_trades']} L:{report['losing_trades']})")
    print(f"Win Rate: {report['win_rate']}")
    print(f"Profit Factor: {report['profit_factor']}")
    print(f"Fees: ${report['total_fees']}")
    print("=" * 60)
