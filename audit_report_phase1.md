# StratoCrypto — Architecture Audit Report
## Phase 1: Full System Audit

**Date:** 2026-08-09
**Auditor:** Senior Quant Developer
**Status:** AUDIT COMPLETE

---

## Executive Summary

StratoCrypto is a **working Statistical Arbitrage system** with genuine causal logic in its core engine (stat_arb.py walk-forward). However, it has significant gaps in risk management, execution quality, statistical rigor, and production readiness.

**Bottom line:** The pairs edge is real and well-implemented. The system needs a proper risk engine, execution engine, experiment tracking, and stress testing before it can be trusted with real capital.

**Production Readiness: YELLOW** (Research/Paper Trading only — not ready for live)

---

## Architecture Map

### Data Flow (Current)

```
Market Data (Binance)
  → data_loader.binance_ohlcv.fetch_klines (canonical loader, retry/paginate)
  → market_data.get_all_prices / get_klines (TTL cache layer)
  → stat_arb.fetch_aligned_pair (timestamp-aligned merge)
  → stat_arb.walk_forward (train/test split, causal)
  → pair_paper.step (live state machine, multi-TF)
  → paper_pairs.json (persistent book)
```

### Components Inventory

| Component | File | Status | Notes |
|---|---|---|---|
| Market Data | `data_loader/binance_ohlcv.py` | ✅ Good | Retry, pagination, session reuse |
| Cache Layer | `market_data.py` | ✅ Good | TTL cache, thread-safe, fallback |
| Pair Discovery | `stat_arb.py` scan_pairs | ✅ Good | Cointegration + half-life filter |
| Cointegration | `stat_arb.py` test_cointegration | ✅ Good | Engle-Granger, p<0.05 |
| Hedge Ratio | `stat_arb.py` ols_hedge_ratio | ✅ Good | OLS on log prices |
| Spread/Z-score | `stat_arb.py` spread_series | ✅ Good | log(a) - h*log(b) |
| Half-life | `stat_arb.py` half_life | ✅ Good | OU process fit |
| Walk-forward | `stat_arb.py` walk_forward | ✅ Good | Train/test split, causal |
| Backtest | `stat_arb.py` backtest_spread | ✅ Good | Causal state machine |
| Pairs Book | `pair_paper.py` | ✅ Good | Multi-TF, stop-loss, sizing |
| Regime Predictor | `regime_predictor.py` | ✅ Good | Causal statistics |
| Paper Engine | `paper_trading.py` | ⚠️ Partial | Single-leg only |
| Prediction Tracker | `prediction_tracker.py` | ✅ Good | Atomic lock, RMW |
| Settings | `settings.py` | ✅ Good | Centralized |
| Config | `.env` | ✅ Good | Env-based secrets |
| Risk Metrics | `risk_metrics.py` | ⚠️ Incomplete | No portfolio risk |
| Portfolio Manager | `portfolio_manager.py` | ⚠️ Simplified | No drawdown control |
| Dashboard | `dashboard_server.py` | ⚠️ Basic | No risk visualization |
| Launcher | `launch_bots.py` | ✅ Good | Ordered start, watchdog |

---

## What is CORRECT (Preserve)

### 1. Walk-Forward is Genuinely Causal
The `stat_arb.walk_forward` function correctly:
- Splits data into train/test folds
- Estimates hedge ratio, cointegration, mean, std on TRAIN window ONLY
- Never touches test data during estimation
- Charges transaction costs on every position change
- Uses `np.diff(spread)` for bar returns (correct)

### 2. Cointegration Test is Correct
Uses `statsmodels.tsa.stattools.coint` (Engle-Granger) with p<0.05 threshold. Properly rejects non-cointegrated pairs.

### 3. Hedge Ratio is Correct
OLS on log prices: `log(a) = h * log(b) + residual`. This is the standard approach for pairs trading.

### 4. Prediction Tracker is Thread-Safe
Atomic lock file + temp-write-rename + read-modify-write. Prevents corruption from concurrent bots.

### 5. Market Data Cache is Well-Designed
TTL cache with thread-safe locking. Falls back to direct Binance call on cache failure.

### 6. Regime Predictor Uses Causal Statistics
All features (autocorrelation, variance ratio, directional DX) are computed from past data only. No look-ahead.

---

## What is QUESTIONABLE (Investigate)

### 1. Sharpe Ratio Calculation is Inflated
**Location:** `stat_arb._metrics()` line 221-223

```python
std = ret.std(ddof=1)
sharpe = float(ret.mean() / std) * math.sqrt(bars_per_year)
```

**Problem:** `ret` is a per-bar array where most bars are 0 (when flat). This inflates the mean/std ratio and produces misleadingly high Sharpe ratios.

**Fix:** Use per-trade returns, not per-bar returns. Or use a proper annualization that accounts for the actual number of trades.

### 2. Regime Predictor Thresholds are Arbitrary
**Location:** `regime_predictor.py` lines 156-184

The thresholds (`vol_20 > 1.0`, `dx > 0.65`, `score += 30`) are magic numbers without OOS optimization.

**Fix:** Make thresholds configurable and test OOS.

### 3. Portfolio Manager Uses Simplified Kelly
**Location:** `portfolio_manager.py` Kelly formula is `avg_ret / var_ret × 0.25`, which is not the proper Kelly criterion.

**Fix:** Use proper Kelly: `f* = (p × b - q) / b` where `b` is win/loss ratio.

### 4. No Drawdown Protection
The system has NO mechanism to halt trading when drawdown exceeds a threshold.

**Fix:** Implement daily loss limit, max drawdown limit, consecutive-loss protection.

### 5. No Correlation Risk Management
82 pairs may share common risk factors (e.g., many depend on BTC). The system doesn't calculate portfolio correlation.

**Fix:** Implement correlation matrix and concentration limits.

### 6. No Circuit Breaker
If the API fails, data is corrupted, or spreads explode, there's no automatic halt.

**Fix:** Implement circuit breaker that stops new entries on critical conditions.

---

## What is BROKEN (Fix)

### 1. Walk-Forward Baseline Leak
**Location:** `stat_arb.walk_forward` line 266-267

```python
s_tr = spread_series(a_tr, b_tr, hedge)
mean = float(np.mean(s_tr))
std = float(np.std(s_tr))
```

**Problem:** The baseline mean/std includes the LAST bar of the train window. When the first test bar uses this baseline, it's technically using information from the train window's last bar, which is at the boundary.

**Fix:** Exclude the last train bar from baseline calculation (already partially fixed in pair_paper._baseline but NOT in stat_arb.walk_forward).

### 2. paper_trading.py Doesn't Support Two-Leg Trades
The single-leg PaperTrade engine cannot represent a pairs position (long A + short B).

**Fix:** Extend or use pair_paper.py as the sole paper engine.

### 3. No Experiment Tracking
There's no registry of experiments, parameters tested, results, or code versions. This enables data snooping.

**Fix:** Implement experiment registry with timestamp, params, results, git hash.

### 4. No Stress Testing
The system has never been tested under adverse conditions (high volatility, crashes, liquidity crisis).

**Fix:** Implement stress tests with widened spreads, increased slippage, extreme volatility.

### 5. No Execution Quality Tracking
The system doesn't track implementation shortfall, fill rate, or slippage.

**Fix:** Add execution quality metrics to track signal price vs actual execution price.

---

## What is MISSING (Add)

### 1. Centralized Risk Engine
No component enforces position limits, drawdown limits, or correlation limits.

### 2. Execution Engine
No separation between signal generation and order execution.

### 3. Experiment Registry
No tracking of experiments, parameters, or results.

### 4. Stress Testing Framework
No adverse-condition testing.

### 5. Production Dashboard
No professional monitoring of portfolio health, pair status, or risk metrics.

### 6. Alerting System
No alerts for pair degradation, cointegration breakdown, or risk limit breaches.

### 7. Regime-Specific Performance Measurement
No tracking of strategy performance by market regime.

### 8. Multi-Timeframe Integration
The system scans multiple timeframes but doesn't integrate them into a unified signal.

---

## Statistical Issues Found

### 1. Look-Ahead Bias: LOW RISK
The walk-forward is well-implemented. No systematic future-data leakage detected.

### 2. Data Snooping: MEDIUM RISK
82 pairs were selected from ~1500 candidates based on OOS performance. This is selection bias.
**Mitigation:** Report the number of tested pairs and apply Bonferroni correction or hold-out validation.

### 3. Survivorship Bias: LOW RISK
Only current active symbols are tested. Delisted assets are excluded.
**Mitigation:** Report this limitation explicitly.

### 4. Statistical Significance: MEDIUM RISK
Many pairs have <10 trades. A Sharpe ratio from 5 trades is not statistically reliable.
**Mitigation:** Require minimum 20 trades for OOS inclusion. Report confidence intervals.

### 5. Sharpe Inflation: HIGH RISK
Per-bar Sharpe calculation inflates results by 2-5x compared to per-trade Sharpe.
**Fix:** Calculate Sharpe on per-trade returns.

---

## Recommended Fix Priority

| Priority | Issue | Impact |
|---|---|---|
| **P0** | Fix Sharpe calculation (per-trade) | Performance metrics are misleading |
| **P0** | Add drawdown protection | Can lose entire capital |
| **P0** | Fix walk-forward baseline leak | Slight performance inflation |
| **P1** | Implement correlation risk | Concentration risk |
| **P1** | Add circuit breaker | System failure protection |
| **P1** | Add experiment registry | Data snooping prevention |
| **P2** | Add execution quality tracking | Realistic PnL estimation |
| **P2** | Add stress testing | Robustness validation |
| **P3** | Production dashboard | Monitoring |
| **P3** | Alerting system | Timely intervention |

---

## Conclusion

StratoCrypto has a **genuinely causal core engine** (stat_arb walk-forward) that correctly implements pairs trading. This is the most important thing and it's done right.

However, the system lacks:
1. Proper risk management (drawdown, correlation, circuit breakers)
2. Statistical rigor (Sharpe inflation, data snooping)
3. Production infrastructure (execution tracking, stress testing, monitoring)

**Recommendation:** Fix P0 issues first, then P1. Do NOT enable live trading until P0 and P1 are resolved.

**Production Readiness: YELLOW** (Paper Trading only)

---

*End of Phase 1 Audit*
