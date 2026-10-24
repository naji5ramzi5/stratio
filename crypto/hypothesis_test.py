"""Hypothesis Testing - Test specific improvements."""
import json
import numpy as np
from datetime import datetime, timezone

DATA_DIR = "historical_data"


def load_data():
    data = {}
    timestamps = {}
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
    candles = data[key][mask]
    return candles[-limit:]


def test_hypothesis(data, timestamps, start_date, end_date, params, name):
    """Test one hypothesis with specific parameters."""
    from stat_arb import ols_hedge_ratio, spread_series

    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000) + 86400000

    btc_key = "BTCUSDT_1h"
    btc_ts = timestamps[btc_key]
    period_mask = (btc_ts >= start_ts) & (btc_ts <= end_ts)
    sim_timestamps = btc_ts[period_mask]

    capital = 10000.0
    open_positions = {}
    closed_trades = []

    pairs = params.get("pairs", [("BTCUSDT", "ETHUSDT"), ("SOLUSDT", "XRPUSDT"), ("ADAUSDT", "LTCUSDT")])

    for current_ts in sim_timestamps:
        # Exits
        to_close = []
        for pair, pos in open_positions.items():
            pos["bars_held"] += 1
            sym_a, sym_b = pair.split("/")
            ca = get_candles(data, timestamps, sym_a, current_ts, 10)
            cb = get_candles(data, timestamps, sym_b, current_ts, 10)

            if len(ca) < 2 or len(cb) < 2:
                continue

            curr_a = float(ca[-1, 4])
            curr_b = float(cb[-1, 4])
            entry_a = pos["entry_price_a"]
            entry_b = pos["entry_price_b"]
            direction = pos["direction"]

            pnl_pct = direction * ((curr_a - entry_a) / entry_a - (curr_b - entry_b) / entry_b)

            exit_profit = pnl_pct > params["tp_pct"]
            exit_loss = pnl_pct < params["sl_pct"]
            exit_time = pos["bars_held"] >= params["max_hold"]

            if exit_profit or exit_loss or exit_time:
                gross_pnl = pnl_pct * pos["size_usd"]
                fees = pos["size_usd"] * params["fee_rate"]
                net_pnl = gross_pnl - fees
                capital += net_pnl

                closed_trades.append({
                    "pnl_pct": round(pnl_pct * 100, 3),
                    "pnl_gross": round(gross_pnl, 4),
                    "pnl_net": round(net_pnl, 4),
                    "fees": round(fees, 2),
                    "exit_reason": "profit" if exit_profit else ("stoploss" if exit_loss else "time"),
                })
                to_close.append(pair)

        for p in to_close:
            del open_positions[p]

        # Signals
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
                direction = 1 if z < 0 else -1
                open_positions[pair] = {
                    "direction": direction,
                    "entry_price_a": float(ca[-1, 4]),
                    "entry_price_b": float(cb[-1, 4]),
                    "size_usd": params["size_pct"] * capital,
                    "bars_held": 0,
                }

    # Calculate metrics
    wins = [t for t in closed_trades if t["pnl_pct"] > 0]
    losses = [t for t in closed_trades if t["pnl_pct"] <= 0]
    gross_profit = sum(t["pnl_gross"] for t in closed_trades if t["pnl_gross"] > 0)
    gross_loss = abs(sum(t["pnl_gross"] for t in closed_trades if t["pnl_gross"] <= 0))
    total_fees = sum(t["fees"] for t in closed_trades)
    gross_pnl = gross_profit - gross_loss
    net_pnl = gross_pnl - total_fees

    return {
        "name": name,
        "params": params,
        "trades": len(closed_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed_trades), 3) if closed_trades else 0,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "gross_pnl": round(gross_pnl, 2),
        "total_fees": round(total_fees, 2),
        "net_pnl": round(net_pnl, 2),
        "return_pct": round(net_pnl / 100, 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss > 0 else 0,
        "avg_win": round(np.mean([t["pnl_pct"] for t in wins]), 3) if wins else 0,
        "avg_loss": round(np.mean([t["pnl_pct"] for t in losses]), 3) if losses else 0,
    }


if __name__ == "__main__":
    import os
    data, timestamps = load_data()

    # BASELINE
    baseline_params = {
        "z_entry": 2.0, "tp_pct": 0.001, "sl_pct": -0.005,
        "max_hold": 24, "size_pct": 0.03, "fee_rate": 0.002,
    }

    # Hypothesis 1: Tighter stops
    h1_params = baseline_params.copy()
    h1_params["sl_pct"] = -0.002

    # Hypothesis 2: Higher entry threshold
    h2_params = baseline_params.copy()
    h2_params["z_entry"] = 2.5

    # Hypothesis 3: Fewer pairs (only best)
    h3_params = baseline_params.copy()
    h3_params["pairs"] = [("BTCUSDT", "ETHUSDT")]

    # Hypothesis 4: No time exit (only signal exit)
    h4_params = baseline_params.copy()
    h4_params["max_hold"] = 72

    # Hypothesis 5: Combined improvements
    h5_params = baseline_params.copy()
    h5_params["z_entry"] = 2.5
    h5_params["sl_pct"] = -0.002
    h5_params["tp_pct"] = 0.005
    h5_params["max_hold"] = 48

    # Run all tests
    start, end = "2026-06-29", "2026-08-07"
    results = []

    print("Testing hypotheses...")
    results.append(test_hypothesis(data, timestamps, start, end, baseline_params, "BASELINE"))
    results.append(test_hypothesis(data, timestamps, start, end, h1_params, "H1_TIGHTER_STOP"))
    results.append(test_hypothesis(data, timestamps, start, end, h2_params, "H2_HIGHER_ENTRY"))
    results.append(test_hypothesis(data, timestamps, start, end, h3_params, "H3_ONE_PAIR"))
    results.append(test_hypothesis(data, timestamps, start, end, h4_params, "H4_NO_TIME_EXIT"))
    results.append(test_hypothesis(data, timestamps, start, end, h5_params, "H5_COMBINED"))

    print("\n=== HYPOTHESIS TESTING RESULTS ===")
    print(f"{'Name':<20} {'Trades':>6} {'WR':>6} {'Gross':>8} {'Fees':>7} {'Net':>8} {'PF':>6}")
    print("-" * 70)
    for r in results:
        print(f"{r['name']:<20} {r['trades']:>6} {r['win_rate']:>5.0%} {r['gross_pnl']:>+7.2f} {r['total_fees']:>6.2f} {r['net_pnl']:>+7.2f} {r['profit_factor']:>5.2f}")

    with open("hypothesis_results.json", "w") as f:
        json.dump(results, f, indent=2)
