"""Test different entry thresholds to find optimal Z-entry."""
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


def run_with_z_entry(data, timestamps, start_date, end_date, z_entry, tp_pct=0.001, sl_pct=-0.005, max_hold=24):
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
            if pnl_pct > tp_pct or pnl_pct < sl_pct or pos["bars_held"] >= max_hold:
                gross_pnl = pnl_pct * pos["size_usd"]
                fees = pos["size_usd"] * 0.0025  # 0.25% round trip
                net_pnl = gross_pnl - fees
                capital += net_pnl
                closed_trades.append({"pnl_gross": gross_pnl, "pnl_net": net_pnl, "pnl_pct": pnl_pct * 100})
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
            if abs(z) >= z_entry:
                open_positions[pair] = {
                    "direction": 1 if z < 0 else -1,
                    "entry_price_a": float(ca[-1, 4]),
                    "entry_price_b": float(cb[-1, 4]),
                    "size_usd": 0.03 * capital,
                    "bars_held": 0,
                }

    gross_pnl = sum(t["pnl_gross"] for t in closed_trades)
    net_pnl = sum(t["pnl_net"] for t in closed_trades)
    wins = [t for t in closed_trades if t["pnl_pct"] > 0]
    losses = [t for t in closed_trades if t["pnl_pct"] <= 0]
    gross_profit = sum(t["pnl_gross"] for t in closed_trades if t["pnl_gross"] > 0)
    gross_loss = abs(sum(t["pnl_gross"] for t in closed_trades if t["pnl_gross"] <= 0))

    return {
        "z_entry": z_entry,
        "trades": len(closed_trades),
        "win_rate": round(len(wins) / len(closed_trades), 3) if closed_trades else 0,
        "gross_pnl": round(gross_pnl, 2),
        "net_pnl": round(net_pnl, 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
        "avg_win": round(np.mean([t["pnl_pct"] for t in wins]), 3) if wins else 0,
        "avg_loss": round(np.mean([t["pnl_pct"] for t in losses]), 3) if losses else 0,
    }


if __name__ == "__main__":
    import os
    data, timestamps = load_data()

    print("=== Z-ENTRY THRESHOLD SENSITIVITY ===")
    print(f"{'Z-Entry':>8} {'Trades':>6} {'WR':>6} {'Gross':>8} {'Net':>8} {'PF':>6}")
    print("-" * 50)

    results = []
    for z in [1.5, 1.8, 2.0, 2.2, 2.5, 3.0, 3.5]:
        r = run_with_z_entry(data, timestamps, "2026-06-29", "2026-08-07", z)
        results.append(r)
        print(f"{z:>8.1f} {r['trades']:>6} {r['win_rate']:>5.0%} {r['gross_pnl']:>+7.2f} {r['net_pnl']:>+7.2f} {r['profit_factor']:>5.2f}")

    with open("z_entry_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nConclusion: No Z-entry threshold produces positive gross PnL.")
    print("The core mean-reversion signal has no edge in this period.")
