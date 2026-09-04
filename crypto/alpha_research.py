"""Alpha Discovery Research Engine.
Tests multiple strategy families on real historical data.
Focus: Discovery of robust alpha, not backtest optimization."""
import json
import os
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict

DATA_DIR = "historical_data"

def load_all_data():
    """Load all historical data into memory."""
    data, timestamps = {}, {}
    for f in os.listdir(DATA_DIR):
        if f.endswith(".json") and f != "metadata.json":
            key = f.replace(".json", "")
            with open(os.path.join(DATA_DIR, f)) as fh:
                raw = json.load(fh)
            arr = np.array([[c[0], c[1], c[2], c[3], c[4], c[5]] for c in raw], dtype=float)
            data[key] = arr
            timestamps[key] = arr[:, 0]
    return data, timestamps

def get_candles(data, timestamps, symbol, end_ts, limit=500):
    """Get candles up to end_ts (point-in-time safe)."""
    key = f"{symbol}_1h"
    if key not in data:
        return np.array([])
    mask = timestamps[key] <= end_ts
    return data[key][mask][-limit:]

def run_strategy(data, timestamps, start_date, end_date, strategy_func, params, name):
    """Generic strategy runner."""
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000) + 86400000

    btc_ts = timestamps["BTCUSDT_1h"]
    sim_timestamps = btc_ts[(btc_ts >= start_ts) & (btc_ts <= end_ts)]

    capital = 10000.0
    peak_capital = capital
    open_positions = {}
    closed_trades = []
    equity_curve = []

    fee_rate = params.get("fee_rate", 0.0025)
    slippage_rate = params.get("slippage_rate", 0.0005)

    for i, current_ts in enumerate(sim_timestamps):
        if i % 200 == 0:
            print(f"  [{name}] Progress: {i}/{len(sim_timestamps)}", flush=True)

        # 1. Manage exits
        to_close = []
        for symbol, pos in open_positions.items():
            pos["bars_held"] += 1
            ca = get_candles(data, timestamps, symbol, current_ts, 10)
            if len(ca) < 2:
                continue

            curr_price = float(ca[-1, 4])
            pnl_pct = pos["direction"] * (curr_price - pos["entry_price"]) / pos["entry_price"]

            exit_signal = strategy_func("exit", ca, pos, params)
            exit_time = pos["bars_held"] >= params.get("max_hold", 48)

            if exit_signal or exit_time:
                gross_pnl = pnl_pct * pos["size_usd"]
                costs = pos["size_usd"] * (fee_rate + slippage_rate)
                net_pnl = gross_pnl - costs
                capital += net_pnl

                closed_trades.append({
                    "symbol": symbol,
                    "direction": pos["direction"],
                    "entry_price": pos["entry_price"],
                    "exit_price": curr_price,
                    "pnl_pct": round(pnl_pct * 100, 3),
                    "pnl_net": round(net_pnl, 2),
                    "exit_reason": "signal" if exit_signal else "time",
                    "bars_held": pos["bars_held"],
                })
                to_close.append(symbol)

        for s in to_close:
            del open_positions[s]

        # 2. Generate signals
        signal = strategy_func("signal", None, None, params, current_ts, data, timestamps)
        if signal and len(open_positions) < params.get("max_positions", 3):
            for sym, direction in signal.items():
                if sym in open_positions:
                    continue
                ca = get_candles(data, timestamps, sym, current_ts, 5)
                if len(ca) < 2:
                    continue
                open_positions[sym] = {
                    "symbol": sym,
                    "direction": direction,
                    "entry_price": float(ca[-1, 4]),
                    "size_usd": params["size_pct"] * capital,
                    "bars_held": 0,
                }

        # 3. Record equity
        unrealized = 0.0
        for sym, pos in open_positions.items():
            ca = get_candles(data, timestamps, sym, current_ts, 5)
            if len(ca) > 1:
                curr = float(ca[-1, 4])
                unrealized += pos["direction"] * (curr - pos["entry_price"]) / pos["entry_price"] * pos["size_usd"]

        equity = capital + unrealized
        if equity > peak_capital:
            peak_capital = equity
        dd = (equity - peak_capital) / peak_capital * 100 if peak_capital > 0 else 0
        equity_curve.append({"equity": equity, "dd": dd})

    # Calculate metrics
    wins = [t for t in closed_trades if t["pnl_pct"] > 0]
    losses = [t for t in closed_trades if t["pnl_pct"] <= 0]
    gross_profit = sum(t["pnl_net"] for t in closed_trades if t["pnl_net"] > 0)
    gross_loss = abs(sum(t["pnl_net"] for t in closed_trades if t["pnl_net"] <= 0))
    net_pnl = sum(t["pnl_net"] for t in closed_trades)
    total_costs = sum(t["pnl_net"] - t["pnl_pct"] * 100 for t in closed_trades) if closed_trades else 0

    max_dd = min(e["dd"] for e in equity_curve) if equity_curve else 0

    return {
        "name": name,
        "trades": len(closed_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed_trades), 3) if closed_trades else 0,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "net_pnl": round(net_pnl, 2),
        "return_pct": round(net_pnl / 100, 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
        "max_drawdown_pct": round(max_dd, 2),
        "avg_win": round(np.mean([t["pnl_pct"] for t in wins]), 3) if wins else 0,
        "avg_loss": round(np.mean([t["pnl_pct"] for t in losses]), 3) if losses else 0,
        "trades_per_day": round(len(closed_trades) / ((end_ts - start_ts) / 86400000), 2),
    }


def momentum_strategy(phase, candles, pos, params, current_ts=None, data=None, timestamps=None):
    """Momentum strategy: Buy when price breaks above N-period high, sell when below N-period low."""
    if phase == "exit":
        if candles is None or len(candles) < 2:
            return False
        curr = float(candles[-1, 4])
        # Exit if price drops below entry by stop loss or rises above take profit
        pnl_pct = pos["direction"] * (curr - pos["entry_price"]) / pos["entry_price"]
        return pnl_pct > params.get("tp", 0.01) or pnl_pct < params.get("sl", -0.005)

    elif phase == "signal":
        # Generate buy signals for strong momentum
        signals = {}
        for symbol in params.get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"]):
            ca = get_candles(data, timestamps, symbol, current_ts, 100)
            if len(ca) < 50:
                continue

            # Momentum: price vs 20-period moving average
            close = ca[:, 4]
            ma20 = np.mean(close[-20:])
            ma50 = np.mean(close[-50:]) if len(close) >= 50 else np.mean(close)

            # Buy when price > MA20 and MA20 > MA50 (uptrend)
            if close[-1] > ma20 and ma20 > ma50:
                signals[symbol] = 1  # Long
            # Sell when price < MA20 and MA20 < MA50 (downtrend)
            elif close[-1] < ma20 and ma20 < ma50:
                signals[symbol] = -1  # Short

        return signals if signals else None


def trend_following_strategy(phase, candles, pos, params, current_ts=None, data=None, timestamps=None):
    """Trend following: ADX-like trend strength + moving average cross."""
    if phase == "exit":
        if candles is None or len(candles) < 2:
            return False
        curr = float(candles[-1, 4])
        pnl_pct = pos["direction"] * (curr - pos["entry_price"]) / pos["entry_price"]
        return pnl_pct > params.get("tp", 0.015) or pnl_pct < params.get("sl", -0.008)

    elif phase == "signal":
        signals = {}
        for symbol in params.get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"]):
            ca = get_candles(data, timestamps, symbol, current_ts, 100)
            if len(ca) < 50:
                continue

            close = ca[:, 4]
            # Simple ADX approximation: directional movement
            diffs = np.diff(close[-20:])
            plus_dm = np.sum(diffs[diffs > 0])
            minus_dm = abs(np.sum(diffs[diffs < 0]))
            total_dm = plus_dm + minus_dm
            if total_dm == 0:
                continue

            dx = abs(plus_dm - minus_dm) / total_dm * 100

            # Only trade strong trends (DX > 25)
            if dx > 25:
                if plus_dm > minus_dm:
                    signals[symbol] = 1
                else:
                    signals[symbol] = -1

        return signals if signals else None


def cross_sectional_momentum(phase, candles, pos, params, current_ts=None, data=None, timestamps=None):
    """Cross-sectional momentum: Rank assets by recent return, go long top performers."""
    if phase == "exit":
        if candles is None or len(candles) < 2:
            return False
        curr = float(candles[-1, 4])
        pnl_pct = pos["direction"] * (curr - pos["entry_price"]) / pos["entry_price"]
        return pnl_pct > params.get("tp", 0.008) or pnl_pct < params.get("sl", -0.004)

    elif phase == "signal":
        symbols = params.get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LTCUSDT"])
        returns = []

        for symbol in symbols:
            ca = get_candles(data, timestamps, symbol, current_ts, 100)
            if len(ca) < 50:
                continue
            # 10-period return
            ret = (ca[-1, 4] - ca[-10, 4]) / ca[-10, 4]
            returns.append((symbol, ret))

        if not returns:
            return None

        # Rank by return
        returns.sort(key=lambda x: x[1], reverse=True)
        signals = {}

        # Go long top 2, short bottom 2
        for sym, ret in returns[:2]:
            if ret > 0.005:  # Only if positive momentum
                signals[sym] = 1
        for sym, ret in returns[-2:]:
            if ret < -0.005:
                signals[sym] = -1

        return signals if signals else None


def volatility_strategy(phase, candles, pos, params, current_ts=None, data=None, timestamps=None):
    """Volatility-based: Enter when vol contracts, exit when vol expands."""
    if phase == "exit":
        if candles is None or len(candles) < 2:
            return False
        curr = float(candles[-1, 4])
        pnl_pct = pos["direction"] * (curr - pos["entry_price"]) / pos["entry_price"]
        return pnl_pct > params.get("tp", 0.006) or pnl_pct < params.get("sl", -0.003)

    elif phase == "signal":
        signals = {}
        for symbol in params.get("symbols", ["BTCUSDT", "ETHUSDT", "SOLUSDT"]):
            ca = get_candles(data, timestamps, symbol, current_ts, 100)
            if len(ca) < 50:
                continue

            close = ca[:, 4]
            returns = np.diff(np.log(close[-50:]))

            # Current volatility vs historical
            current_vol = np.std(returns[-10:])
            historical_vol = np.std(returns)

            # Enter when vol is contracting (current < 0.7 * historical)
            if current_vol < 0.7 * historical_vol and historical_vol > 0:
                # Direction: slight momentum
                if returns[-1] > 0:
                    signals[symbol] = 1
                else:
                    signals[symbol] = -1

        return signals if signals else None


if __name__ == "__main__":
    data, timestamps = load_all_data()

    # Development period
    dev_start, dev_end = "2026-06-29", "2026-07-20"
    oos_start, oos_end = "2026-07-20", "2026-08-07"

    # Test parameters
    momentum_params = {
        "fee_rate": 0.002, "slippage_rate": 0.0005, "tp": 0.01, "sl": -0.005,
        "max_hold": 48, "size_pct": 0.05, "max_positions": 3,
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    }

    trend_params = {
        "fee_rate": 0.002, "slippage_rate": 0.0005, "tp": 0.015, "sl": -0.008,
        "max_hold": 72, "size_pct": 0.05, "max_positions": 3,
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    }

    cs_params = {
        "fee_rate": 0.002, "slippage_rate": 0.0005, "tp": 0.008, "sl": -0.004,
        "max_hold": 36, "size_pct": 0.03, "max_positions": 4,
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"],
    }

    vol_params = {
        "fee_rate": 0.002, "slippage_rate": 0.0005, "tp": 0.006, "sl": -0.003,
        "max_hold": 24, "size_pct": 0.04, "max_positions": 3,
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    }

    print("=" * 70)
    print("ALPHA DISCOVERY RESEARCH ENGINE")
    print("=" * 70)

    strategies = [
        ("Momentum", momentum_strategy, momentum_params),
        ("Trend Following", trend_following_strategy, trend_params),
        ("Cross-Sectional Momentum", cross_sectional_momentum, cs_params),
        ("Volatility", volatility_strategy, vol_params),
    ]

    results = []

    for name, func, params in strategies:
        print(f"\n--- Testing {name} ---")

        # Development period
        dev_result = run_strategy(data, timestamps, dev_start, dev_end, func, params, f"{name}_DEV")
        print(f"DEV: {dev_result['trades']} trades, WR={dev_result['win_rate']}, PF={dev_result['profit_factor']}, Return={dev_result['return_pct']}%")

        # OOS period
        oos_result = run_strategy(data, timestamps, oos_start, oos_end, func, params, f"{name}_OOS")
        print(f"OOS: {oos_result['trades']} trades, WR={oos_result['win_rate']}, PF={oos_result['profit_factor']}, Return={oos_result['return_pct']}%")

        results.append({
            "name": name,
            "development": dev_result,
            "oos": oos_result,
        })

    # Summary
    print(f"\n{'='*70}")
    print("RESEARCH SUMMARY")
    print(f"{'='*70}")
    print(f"{'Strategy':<25} {'Dev PF':>8} {'Dev Ret':>8} {'OOS PF':>8} {'OOS Ret':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<25} {r['development']['profit_factor']:>8.2f} {r['development']['return_pct']:>+7.2f}% {r['oos']['profit_factor']:>8.2f} {r['oos']['return_pct']:>+7.2f}%")

    with open("alpha_research_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nSaved to alpha_research_results.json")
