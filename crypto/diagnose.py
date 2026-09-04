"""Root Cause Analysis - Why does the strategy lose?
Tests hypotheses about failure modes."""
import json
import numpy as np
from datetime import datetime, timezone

# Load the detailed trade data from a full run
# First, let's run a diagnostic that captures detailed trade info

def run_diagnostic_backtest(start_date, end_date):
    """Run backtest with detailed trade capture."""
    import sys
    sys.path.insert(0, '.')
    from fast_backtest import FastHistoricalBacktester

    bt = FastHistoricalBacktester(start_date=start_date, end_date=end_date, initial_capital=10000.0)
    report = bt.run()

    # Analyze trades
    trades = bt.closed_trades

    if not trades:
        return {"error": "no trades"}

    # Gross edge (no costs)
    gross_pnl_no_costs = sum(t["pnl_gross"] for t in trades)
    wins = [t for t in trades if t["pnl_pct"] > 0]
    losses = [t for t in trades if t["pnl_pct"] <= 0]

    # Categorize exits
    exit_reasons = {}
    for t in trades:
        reason = t["exit_reason"]
        if reason not in exit_reasons:
            exit_reasons[reason] = []
        exit_reasons[reason].append(t)

    # Z-score analysis
    z_scores = [abs(t.get("entry_z", 0)) for t in trades if "entry_z" in t]

    # Holding time analysis
    hold_times = [t["bars_held"] for t in trades]

    analysis = {
        "period": f"{start_date} to {end_date}",
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(trades) if trades else 0,

        # Gross vs Net
        "gross_pnl_no_costs": round(gross_pnl_no_costs, 2),
        "total_fees": round(sum(t["fees"] for t in trades), 2),
        "net_pnl": round(gross_pnl_no_costs - sum(t["fees"] for t in trades), 2),

        # Exit analysis
        "exit_reasons": {k: len(v) for k, v in exit_reasons.items()},
        "avg_win_pct": round(np.mean([t["pnl_pct"] for t in wins]), 3) if wins else 0,
        "avg_loss_pct": round(np.mean([t["pnl_pct"] for t in losses]), 3) if losses else 0,

        # Trade frequency
        "trades_per_day": round(len(trades) / 7, 1),  # approximate

        # Z-score stats
        "avg_entry_z": round(np.mean(z_scores), 2) if z_scores else 0,

        # Holding time
        "avg_hold_bars": round(np.mean(hold_times), 1) if hold_times else 0,
    }

    return analysis


if __name__ == "__main__":
    # Run diagnostic on the main period
    result = run_diagnostic_backtest("2026-06-29", "2026-08-07")

    print("=== ROOT CAUSE ANALYSIS ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    # Save
    with open("diagnostic_results.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\nSaved to diagnostic_results.json")
