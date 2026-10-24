"""RESEARCH_V2: Redesigned strategy with regime awareness, volatility-adjusted exits, and fewer pairs."""
import json
import numpy as np
from datetime import datetime, timezone

DATA_DIR = "historical_data"


def load_data():
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
    key = f"{symbol}_1h"
    if key not in data:
        return np.array([])
    mask = timestamps[key] <= end_ts
    return data[key][mask][-limit:]


def detect_regime(closes_1h, current_ts_idx):
    """Simple regime detection based on recent price behavior."""
    if current_ts_idx < 50:
        return "UNKNOWN"

    recent_closes = closes_1h[current_ts_idx-50:current_ts_idx]

    # Trend: ADX-like simple measure
    diffs = np.diff(recent_closes)
    trend_strength = abs(np.sum(diffs)) / (np.sum(np.abs(diffs)) + 1e-9)

    # Volatility
    returns = np.diff(np.log(recent_closes + 1e-9))
    vol = np.std(returns)

    if trend_strength > 0.6:
        return "TRENDING"
    elif vol > 0.01:
        return "VOLATILE"
    else:
        return "MEAN_REVERTING"


def run_research_v2(data, timestamps, start_date, end_date):
    """Redesigned strategy."""
    from stat_arb import ols_hedge_ratio, spread_series

    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000) + 86400000

    btc_ts = timestamps["BTCUSDT_1h"]
    sim_timestamps = btc_ts[(btc_ts >= start_ts) & (btc_ts <= end_ts)]

    capital = 10000.0
    peak_capital = capital
    open_positions = {}
    closed_trades = []
    equity_curve = []

    # Only the best pair (from our research)
    pairs = [("BTCUSDT", "ETHUSDT")]

    # Parameters
    z_entry = 2.5
    fee_rate = 0.002
    slippage_rate = 0.0005
    position_size = 0.05  # larger per trade but fewer trades
    max_hold = 48

    regime_counts = {}

    for i, current_ts in enumerate(sim_timestamps):
        # Exits
        to_close = []
        for pair, pos in open_positions.items():
            pos["bars_held"] += 1
            sym_a, sym_b = pair.split("/")
            ca = get_candles(data, timestamps, sym_a, current_ts, 10)
            cb = get_candles(data, timestamps, sym_b, current_ts, 10)
            if len(ca) < 2 or len(cb) < 2:
                continue

            curr_a, curr_b = float(ca[-1, 4]), float(cb[-1, 4])
            pnl_pct = pos["direction"] * ((curr_a - pos["entry_price_a"]) / pos["entry_price_a"] - (curr_b - pos["entry_price_b"]) / pos["entry_price_b"])

            # Dynamic exit based on Z-score reversal
            # Exit when spread reverts to within 0.5 std
            z_revert = abs(pnl_pct) < 0.0005

            exit_profit = pnl_pct > 0.003 or z_revert
            exit_loss = pnl_pct < -0.003
            exit_time = pos["bars_held"] >= max_hold

            if exit_profit or exit_loss or exit_time:
                gross_pnl = pnl_pct * pos["size_usd"]
                fees = pos["size_usd"] * (fee_rate + slippage_rate)
                net_pnl = gross_pnl - fees
                capital += net_pnl

                closed_trades.append({
                    "pair": pair,
                    "pnl_pct": round(pnl_pct * 100, 3),
                    "pnl_net": round(net_pnl, 2),
                    "exit_reason": "profit" if exit_profit else ("stoploss" if exit_loss else "time"),
                    "bars_held": pos["bars_held"],
                    "regime": pos.get("regime", "unknown"),
                })
                to_close.append(pair)

        for p in to_close:
            del open_positions[p]

        # Signals - only enter in MEAN_REVERTING regime
        for sym_a, sym_b in pairs:
            pair = f"{sym_a}/{sym_b}"
            if pair in open_positions:
                continue

            ca = get_candles(data, timestamps, sym_a, current_ts, 500)
            cb = get_candles(data, timestamps, sym_b, current_ts, 500)
            if len(ca) < 200 or len(cb) < 200:
                continue

            # Check regime first
            regime = detect_regime(ca[:, 4], len(ca) - 1)
            if regime not in regime_counts:
                regime_counts[regime] = 0
            regime_counts[regime] += 1

            # Skip if not mean-reverting
            if regime != "MEAN_REVERTING":
                continue

            h = ols_hedge_ratio(ca[:200, 4], cb[:200, 4])
            spread_train = spread_series(ca[:200, 4], cb[:200, 4], h)
            mu, sd = float(np.mean(spread_train)), float(np.std(spread_train))
            if sd <= 0:
                continue

            current_spread = float(np.log(ca[-1, 4]) - h * np.log(cb[-1, 4]))
            z = (current_spread - mu) / sd

            # Only enter if expected edge > costs
            # Expected edge = (z_entry - 0.5) * std * direction
            expected_edge = (abs(z) - 0.5) * sd * 100  # rough estimate in pct
            if abs(z) >= z_entry and expected_edge > 0.3:  # must be > 0.3% to cover costs
                open_positions[pair] = {
                    "direction": 1 if z < 0 else -1,
                    "entry_price_a": float(ca[-1, 4]),
                    "entry_price_b": float(cb[-1, 4]),
                    "size_usd": position_size * capital,
                    "bars_held": 0,
                    "regime": regime,
                }

        # Record equity
        equity = capital  # simplified
        if equity > peak_capital:
            peak_capital = equity
        dd = (equity - peak_capital) / peak_capital * 100 if peak_capital > 0 else 0
        equity_curve.append({"equity": equity, "dd": dd})

    # Final metrics
    wins = [t for t in closed_trades if t["pnl_pct"] > 0]
    losses = [t for t in closed_trades if t["pnl_pct"] <= 0]
    gross_profit = sum(t["pnl_net"] for t in closed_trades if t["pnl_net"] > 0)
    gross_loss = abs(sum(t["pnl_net"] for t in closed_trades if t["pnl_net"] <= 0))
    net_pnl = sum(t["pnl_net"] for t in closed_trades)

    max_dd = min(e["dd"] for e in equity_curve) if equity_curve else 0

    return {
        "strategy": "RESEARCH_V2",
        "period": f"{start_date} to {end_date}",
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
        "regime_counts": regime_counts,
    }


if __name__ == "__main__":
    import os
    data, timestamps = load_data()

    # Test on development period
    dev_result = run_research_v2(data, timestamps, "2026-06-29", "2026-07-20")
    print("=== DEVELOPMENT PERIOD ===")
    for k, v in dev_result.items():
        print(f"{k}: {v}")

    # Test on OOS period (unseen)
    oos_result = run_research_v2(data, timestamps, "2026-07-20", "2026-08-07")
    print("\n=== OOS PERIOD (UNSEEN) ===")
    for k, v in oos_result.items():
        print(f"{k}: {v}")

    with open("research_v2_results.json", "w") as f:
        json.dump({"development": dev_result, "oos": oos_result}, f, indent=2)
