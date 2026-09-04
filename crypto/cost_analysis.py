"""Gross Edge vs Net Edge Analysis - Test under different cost scenarios."""
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


def run_with_costs(data, timestamps, start_date, end_date, fee_rate, slippage_rate):
    """Run with specific cost assumptions."""
    from stat_arb import ols_hedge_ratio, spread_series

    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000) + 86400000

    btc_ts = timestamps["BTCUSDT_1h"]
    sim_timestamps = btc_ts[(btc_ts >= start_ts) & (btc_ts <= end_ts)]

    capital = 10000.0
    open_positions = {}
    closed_trades = []
    pairs = [("BTCUSDT", "ETHUSDT"), ("SOLUSDT", "XRPUSDT"), ("ADAUSDT", "LTCUSDT")]

    for current_ts in sim_timestamps:
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

            if pnl_pct > 0.001 or pnl_pct < -0.005 or pos["bars_held"] >= 24:
                gross_pnl = pnl_pct * pos["size_usd"]
                costs = pos["size_usd"] * (fee_rate + slippage_rate)
                net_pnl = gross_pnl - costs
                capital += net_pnl
                closed_trades.append({"pnl_gross": gross_pnl, "costs": costs, "pnl_pct": pnl_pct * 100})
                to_close.append(pair)

        for p in to_close:
            del open_positions[p]

        for sym_a, sym_b in pairs:
            pair = f"{sym_a}/{sym_b}"
            if pair in open_positions:
                continue
            ca = get_candles(data, timestamps, sym_a, current_ts, 500)
            cb = get_candles(data, timestamps, sym_b, current_ts, 500)
            if len(ca) < 200 or len(cb) < 200:
                continue
            h = ols_hedge_ratio(ca[:200, 4], cb[:200, 4])
            spread_train = spread_series(ca[:200, 4], cb[:200, 4], h)
            mu, sd = float(np.mean(spread_train)), float(np.std(spread_train))
            if sd <= 0:
                continue
            current_spread = float(np.log(ca[-1, 4]) - h * np.log(cb[-1, 4]))
            z = (current_spread - mu) / sd
            if abs(z) >= 2.0:
                open_positions[pair] = {
                    "direction": 1 if z < 0 else -1,
                    "entry_price_a": float(ca[-1, 4]),
                    "entry_price_b": float(cb[-1, 4]),
                    "size_usd": 0.03 * capital,
                    "bars_held": 0,
                }

    gross_pnl = sum(t["pnl_gross"] for t in closed_trades)
    total_costs = sum(t["costs"] for t in closed_trades)
    net_pnl = gross_pnl - total_costs
    wins = [t for t in closed_trades if t["pnl_pct"] > 0]
    losses = [t for t in closed_trades if t["pnl_pct"] <= 0]
    gross_profit = sum(t["pnl_gross"] for t in closed_trades if t["pnl_gross"] > 0)
    gross_loss = abs(sum(t["pnl_gross"] for t in closed_trades if t["pnl_gross"] <= 0))

    return {
        "fee_rate": fee_rate,
        "slippage_rate": slippage_rate,
        "trades": len(closed_trades),
        "win_rate": len(wins) / len(closed_trades) if closed_trades else 0,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "gross_pnl": round(gross_pnl, 2),
        "total_costs": round(total_costs, 2),
        "net_pnl": round(net_pnl, 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
    }


if __name__ == "__main__":
    import os
    data, timestamps = load_data()

    scenarios = [
        (0.0, 0.0, "No costs"),
        (0.001, 0.0, "0.1% fees only"),
        (0.002, 0.0, "0.2% fees only"),
        (0.002, 0.0005, "0.2% fees + 5bps slippage (current)"),
        (0.002, 0.001, "0.2% fees + 10bps slippage"),
        (0.005, 0.002, "0.5% fees + 20bps slippage (stress)"),
    ]

    print("=== GROSS EDGE vs NET EDGE ===")
    print(f"{'Scenario':<40} {'Trades':>6} {'Gross':>8} {'Costs':>7} {'Net':>8} {'PF':>6}")
    print("-" * 80)

    results = []
    for fee, slip, name in scenarios:
        r = run_with_costs(data, timestamps, "2026-06-29", "2026-08-07", fee, slip)
        r["name"] = name
        results.append(r)
        print(f"{name:<40} {r['trades']:>6} {r['gross_pnl']:>+7.2f} {r['total_costs']:>6.2f} {r['net_pnl']:>+7.2f} {r['profit_factor']:>5.2f}")

    with open("cost_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nKey finding: Even with NO costs, the strategy is barely positive or negative.")
    print("This means the CORE SIGNAL is weak, not just the costs.")
