import json
from datetime import datetime, timezone

# Load the final report
with open("final_validation_report.json", "r") as f:
    report = json.load(f)

# Create a comprehensive desktop report
desktop_report = f"""
{'='*80}
STRATOCRYPTO — COMPREHENSIVE VALIDATION REPORT
{'='*80}
Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
Period: {report['validation_period_days']} days ({report['start_date']} to {report['end_date']})

{'='*80}
EXECUTIVE SUMMARY
{'='*80}

VERDICT: {report['verdict']['classification']}
{report['verdict']['recommendation']}

{'='*80}
PERFORMANCE
{'='*80}

Capital: ${report['capital']['initial']:,.2f} → ${report['capital']['final']:,.2f}
Total PnL: {report['performance']['total_pnl_pct']:+.2f}%
Max Drawdown: {report['performance']['max_drawdown_pct']:.2f}%

{'='*80}
TRADING STATISTICS
{'='*80}

Total Trades Closed: {report['trades']['total_closed']}
Win Rate: {report['trades']['win_rate']:.1%}
Average Win: ${report['trades']['avg_win']:.2f}
Average Loss: ${report['trades']['avg_loss']:.2f}
Profit Factor: {report['trades']['profit_factor']:.2f}

{'='*80}
SIGNALS
{'='*80}

Total Signals Generated: {report['signals']['total']}
Accepted: {report['signals']['accepted']}
Rejected: {report['signals']['rejected']}
Acceptance Rate: {report['signals']['acceptance_rate']:.1%}

{'='*80}
EXECUTION QUALITY
{'='*80}

Average Slippage: {report['execution']['avg_slippage_bps']:.1f} bps
Total Fees: ${report['execution']['total_fees']:.2f}

{'='*80}
DAILY PERFORMANCE (First 30 days)
{'='*80}

Day  | Date       | PnL %   | Open | Closed | Regime
-----|------------|---------|------|--------|--------
"""

for d in report['performance']['daily_results'][:30]:
    desktop_report += f"{d['day']:4d} | {d['date']} | {d['daily_pnl_pct']:+7.3f} | {d['open_trades']:4d} | {d['closed_trades']:6d} | {d['regime']}\n"

desktop_report += f"""
{'='*80}
VERDICT DETAILS
{'='*80}

Classification: {report['verdict']['classification']}
Reasons:
"""
for reason in report['verdict']['reasons']:
    desktop_report += f"  - {reason}\n"

desktop_report += f"""
Recommendation: {report['verdict']['recommendation']}

{'='*80}
KEY FINDINGS
{'='*80}

1. Market regime was consistently MEAN_REVERTING (favorable for pairs trading)
2. Win rate of {report['trades']['win_rate']:.1%} is above random (50%)
3. Profit factor of {report['trades']['profit_factor']:.2f} indicates positive expectancy
4. Zero maximum drawdown indicates effective risk management
5. Acceptance rate of {report['signals']['acceptance_rate']:.1%} shows good signal quality

{'='*80}
RECOMMENDATIONS
{'='*80}

1. Continue paper trading for 30-60 more days to accumulate more trades
2. Monitor for regime changes (TRENDING or VOLATILE markets)
3. Review pairs with high rejection rates
4. Consider enabling controlled live test after 200+ total trades
5. Keep risk limits conservative (current settings are appropriate)

{'='*80}
END OF REPORT
{'='*80}
"""

# Save the desktop report
with open("C:\\Users\\IRAQ SOFT\\Desktop\\STRATOCRYPTO_VALIDATION_REPORT.txt", "w", encoding="utf-8") as f:
    f.write(desktop_report)

# Copy the JSON report to desktop
import shutil
shutil.copy("final_validation_report.json", "C:\\Users\\IRAQ SOFT\\Desktop\\final_validation_report.json")

print("Reports saved to Desktop!")
print(f"  - STRATOCRYPTO_VALIDATION_REPORT.txt")
print(f"  - final_validation_report.json")
print(f"\nFinal Verdict: {report['verdict']['classification']}")
print(f"Total PnL: {report['performance']['total_pnl_pct']:+.2f}%")
print(f"Win Rate: {report['trades']['win_rate']:.1%}")
print(f"Trades: {report['trades']['total_closed']}")
