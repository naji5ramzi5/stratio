"""RESEARCH_V3: Test with more pairs and relaxed entry."""
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


def run_strategy(data, timestamps, start_date, end_date, params, name):
    """Generic strategy runner with configurable parameters."""
    from stat_arb import ols_hedge_ratio, spread_series

    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000) + 86400000

    btc_ts = timestamps["BTCUSDT_1h"]
    sim_timestamps = btc_ts[(btc_ts >= start_ts) & (btc_ts <= end_ts)]

    capital = 10000.0
    peak_capital = capital
    open_positions = {}
    closed_trades = []

    pairs = params.get("pairs", [])

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

            exit_profit = pnl_pct > params["tp_pct"]
            exit_loss = pnl_pct < params["sl_pct"]
            exit_time = pos["bars_held"] >= params["max_hold"]

            if exit_profit or exit_loss or exit_time:
                gross_pnl = pnl_pct * pos["size_usd"]
                fees = pos["size_usd"] * params["fee_rate"]
                net_pnl = gross_pnl - fees
                capital += net_pnl
                closed_trades.append({"pnl_net": net_pnl, "pnl_pct": pnl_pct * 100})
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

            if abs(z) >= params["z_entry"]:
                open_positions[pair] = {
                    "direction": 1 if z < 0 else -1,
                    "entry_price_a": float(ca[-1, 4]),
                    "entry_price_b": float(cb[-1, 4]),
                    "size_usd": params["size_pct"] * capital,
                    "bars_held": 0,
                }

    wins = [t for t in closed_trades if t["pnl_pct"] > 0]
    losses = [t for t in closed_trades if t["pnl_pct"] <= 0]
    gross_profit = sum(t["pnl_net"] for t in closed_trades if t["pnl_net"] > 0)
    gross_loss = abs(sum(t["pnl_net"] for t in closed_trades if t["pnl_net"] <= 0))
    net_pnl = sum(t["pnl_net"] for t in closed_trades)

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
    }


if __name__ == "__main__":
    import os
    data, timestamps = load_data()

    # Configuration
    all_pairs = [
        ("BTCUSDT", "ETHUSDT"), ("BTCUSDT", "SOLUSDT"), ("ETHUSDT", "SOLUSDT"),
        ("SOLUSDT", "XRPUSDT"), ("ADAUSDT", "LTCUSDT"), ("LINKUSDT", "DOTUSDT"),
        ("BTCUSDT", "XRPUSDT"), ("ETHUSDT", "ADAUSDT"), ("BNBUSDT", "SOLUSDT"),
        ("FILUSDT", "ETCUSDT"),
    ]

    base_params = {
        "z_entry": 2.2,
        "tp_pct": 0.003,
        "sl_pct": -0.003,
        "max_hold": 48,
        "size_pct": 0.03,
        "fee_rate": 0.0025,
        "pairs": all_pairs,
    }

    # Run on multiple periods
    periods = [
        ("2026-06-29", "2026-07-10", "DEV_1"),
        ("2026-07-10", "2026-07-20", "DEV_2"),
        ("2026-07-20", "2026-07-30", "OOS_1"),
        ("2026-07-30", "2026-08-07", "OOS_2"),
    ]

    results = []
    for start, end, name in periods:
        r = run_strategy(data, timestamps, start, end, base_params, name)
        results.append(r)
        print(f"{name}: {r['trades']} trades, WR={r['win_rate']:.0%}, PF={r['profit_factor']:.2f}, Return={r['return_pct']:+.2f}%")

    # Overall OOS
    oos_trades = sum(r["trades"] for r in results if r["name"].startswith("OOS"))
    oos_net = sum(r["net_pnl"] for r in results if r["name"].startswith("OOS"))
    print(f"\nOOS Total: {oos_trades} trades, Net PnL: ${oos_net:.2f}")

    with open("research_v3_results.json", "w") as f:
        json.dump(results, f, indent=2)
