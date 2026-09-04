from data_loader.binance_ohlcv import fetch_klines
from stat_arb import ols_hedge_ratio, spread_series, test_cointegration
from backtester import BacktestConfig, backtest_pair, stress_test
import numpy as np

pairs = [("1000SATSUSDT", "XRPUSDT"), ("ETCUSDT", "OPUSDT"),
         ("CRVUSDT", "FLOKIUSDT"), ("1000SATSUSDT", "SHIBUSDT"),
         ("BONKUSDT", "XRPUSDT"), ("NEARUSDT", "PEPEUSDT")]

for sym_a, sym_b in pairs:
    ka = fetch_klines(sym_a, "1h", 2000)
    kb = fetch_klines(sym_b, "1h", 2000)
    if ka and kb and len(ka) >= 500 and len(kb) >= 500:
        ta = {int(r[0]): float(r[4]) for r in ka}
        tb = {int(r[0]): float(r[4]) for r in kb}
        common = sorted(set(ta) & set(tb))
        if len(common) >= 1000:
            a = np.array([ta[t] for t in common][-1000:])
            b = np.array([tb[t] for t in common][-1000:])
            h = ols_hedge_ratio(a[:700], b[:700])
            passed, pval, h = test_cointegration(a[:700], b[:700], hedge_ratio=h)
            if passed:
                s_tr = spread_series(a[:700], b[:700], h)
                mu, sd = float(np.mean(s_tr)), float(np.std(s_tr))
                config = BacktestConfig(slippage_bps=5)
                result = backtest_pair(a[700:], b[700:], h, mu, sd, config, "1h")
                print(f"\n{sym_a}/{sym_b} (p={pval:.4f}):")
                print(f"  Return:{result.total_return_pct:+.2f}% | Ann:{result.annualized_return_pct:+.1f}%")
                print(f"  Sharpe:{result.sharpe_ratio:.2f} | Sort:{result.sortino_ratio:.2f} | Calmar:{result.calmar_ratio:.2f}")
                print(f"  MaxDD:{result.max_drawdown_pct:.2f}% | Trades:{result.n_trades} (W:{result.n_wins} L:{result.n_losses})")
                print(f"  Win:{result.win_rate:.0%} | PF:{result.profit_factor:.2f} | Exp:{result.expectancy_pct:+.3f}%")
                print(f"  Costs:{result.total_costs_pct:.3f}% | Slippage:{result.total_slippage_pct:.3f}%")
            else:
                print(f"{sym_a}/{sym_b}: p={pval:.4f} (not cointegrated)")
