# ARBITRAGE ARCHITECTURE AUDIT

**Date:** 2026-08-10
**Scope:** Full STRATOCRYPTO codebase audit in preparation for the Arbitrage Research & Simulation Engine.
**Status:** COMPLETE — no code modified.

---

## 1. Executive Summary

STRATOCRYPTO is a **bar-based (OHLCV) directional/statistical-arbitrage research system**.
It has no components suitable for arbitrage research as-is, but its research discipline
(point-in-time safety, OOS walk-forward, honest reporting) is exactly the standard the
new arbitrage subsystem must inherit.

**Verdict:** The arbitrage engine must be built as a **new, separate, event-driven
subsystem**. Reuse: data-fetch pattern, risk metrics, fee/slippage simulation concepts,
reporting utilities. Do NOT reuse: bar-based backtesters, signal generators.

---

## 2. Current Architecture Map

```
DATA LAYER
  data_loader/binance_ohlcv.py      -> REST klines (fetch_klines) [used by download_historical.py]
  download_historical.py            -> batch downloader -> historical_data/*_1h.json
  historical_data/                  -> 10 symbols x 1h OHLCV, 1000 bars each (2026-06-28 -> 2026-08-09)
  market_data.py / crypto_market_data.py / multi_exchange.py -> live REST/WS data
  onchain_data.py, binance_liquidity.py, crypto_sentiment_analyzer.py

SIGNAL LAYER
  stat_arb.py                       -> mean-reversion z-score pair strategy (the main engine)
  regime_gate.py / regime_predictor.py -> regime filter (proven harmful as hard gate)
  predictor.py / foundation_models.py / ml_trainer.py / consensus.py -> ML direction prediction
  reinforcement/ (finrl, sb3_ppo, walkforward) -> RL trading layer
  anomaly_detector.py, meta_labeling.py

BACKTEST LAYER
  fast_backtest.py                  -> the ONLY validated point-in-time-safe candle backtester
  backtester.py / real_historical_backtester.py / backtest/ (vectorbt) -> legacy engines
  run_full_system.py                -> ⚠ INVALID: random PnL + repeated same-day data (do not use)

RESEARCH LAYER
  research_framework.py, research_v2.py, research_v3.py, alpha_research.py
  multi_period_test.py, hypothesis_test.py, z_entry_test.py, cost_analysis.py
  gate_ab_test.py, phase3_validation.py, quant_audit.py

EXECUTION LAYER
  execution_engine.py               -> paper execution: fixed slippage (5bps default),
                                       maker/taker fees (10bps), fills at close
  pair_paper.py / paper_trading.py / live_paper.py -> paper trading state
  portfolio_manager.py / risk_engine.py / risk_metrics.py

REPORTING LAYER
  utils/reporter.py, quant_x/report.py, desktop_report.py, dashboard.py

TESTS
  tests/ : test_stat_arb, test_risk, test_leakage, test_calibration, test_pair_paper,
           test_phase3, test_quant_x, test_regime, test_regime_gate, test_tracker,
           test_state_store, test_integration  (107 passing previously)
```

## 3. Reusable Components (keep / adapt)

| Component | Location | Reuse in arbitrage |
|---|---|---|
| klines fetch pattern (chunked, dedupe, sorted) | `download_historical.py` | Yes — same pattern for aggTrades/klines downloads |
| REST session/rate-limit handling | `data_loader/binance_ohlcv.py` | Yes — extend to trades/funding endpoints |
| Risk metrics (Sharpe, Sortino, MaxDD, PF, Kelly) | `risk_metrics.py` | Yes — statistical analysis layer |
| Fee + slippage simulation concept | `execution_engine.py` (taker 10bps, base slip 5bps) | Yes — but must become depth-aware, not fixed |
| Point-in-time discipline | `fast_backtest.py`, research scripts | Yes — hard rule with assertions |
| OOS / walk-forward methodology | `research_v3.py`, `multi_period_test.py` | Yes — mandatory for every arb model |
| Reporter pattern (JSON + MD) | `utils/reporter.py`, `quant_x/report.py` | Yes — leaderboard/experiment reports |

## 4. Missing Components (must build)

| Required for arbitrage | Status |
|---|---|
| Tick / aggTrades data | ❌ absent |
| Order book (L2/L3) data | ❌ absent |
| Futures data (basis, funding) | ❌ absent |
| Multi-exchange data | ❌ absent (Binance only) |
| Event-driven replay engine | ❌ absent |
| Depth-aware execution simulator | ❌ absent |
| Latency model | ❌ absent |
| Fill probability / partial fills | ❌ absent |
| Two-leg execution risk | ❌ absent |
| Capital transfer model | ❌ absent |
| Fee engine per exchange | ⚠ partial (fixed 10bps only) |
| Minimum order / precision constraints | ❌ absent |
| Opportunity persistence (half-life) | ❌ absent |
| Cycle graph (triangular) search | ❌ absent |

## 5. Data Limitations

1. Only **OHLCV 1h**, 1000 bars/symbol, 10 symbols, ~42 days (2026-06-28 → 2026-08-09).
2. Bar granularity makes arbitrage simulation impossible at this resolution.
3. **Binance Vision (free)** offers: `aggTrades`, `trades`, 1s klines (spot),
   futures `bookTicker` (best bid/ask), `fundingRate`, mark/index klines (USD-M).
   NO spot order-book history is available for free.
4. Order-book data must be **recorded live** (WebSocket) or **purchased** (Tardis.dev, Databento).
5. Multi-exchange history requires additional sources (ccxt REST bulk download possible
   but rate-limited; other exchanges' free archives are scarce).

## 6. Execution Limitations

- No latency modeling anywhere.
- Fills assumed at close of candle or fixed slippage — no depth, no queue position.
- No partial-fill modeling.
- No cross-exchange fund transfer modeling.
- No market impact model.

## 7. Performance Bottlenecks

- Pure-Python loops in legacy backtesters (vectorbt exists but unused at scale).
- For tick-level replay: 1s klines / aggTrades for 3-5 symbols over days = millions of rows.
  → Solution: numpy-vectorized batch simulation (feasible for triangular/basis without L2),
    Numba JIT if order-book simulation becomes necessary.

## 8. Sources of Bias to Avoid (from lessons learned)

1. Random PnL generation (`run_full_system.py` bug) — **NEVER** fabricate fills.
2. Repeated same-day data masquerading as multi-day history.
3. Future timestamps (`datetime.now()` offsets).
4. Optimizing on the evaluation period — DEV/OOS split mandatory.
5. Assuming displayed price == executable price.
6. Assuming infinite liquidity at best bid/ask.

## 9. Duplicated Systems

- 5+ backtesters (`backtester.py`, `real_historical_backtester.py`, `fast_backtest.py`,
  `backtest/vectorbt_backtester.py`, `run_full_system.py`).
- 3+ paper-trading engines (`paper_trading.py`, `pair_paper.py`, `live_paper.py`).
- → For arbitrage: ONE simulator, ONE registry, ONE leaderboard.

## 10. Security Concerns

- `.env` exists with API keys (paper-trading only — no live keys should ever be added).
- No secrets should be committed; arbitrage subsystem is research-only (NO live orders).

## 11. Environment

- Python 3.14.5, pandas 3.0.3, numpy 2.4.6, scipy, numba, ccxt 4.5.71, websockets.
- `hftbacktest` NOT installed — not required; we build a focused numpy-driven engine.

## 12. Recommended Structure

```
crypto/arbitrage/                <- NEW subsystem (isolated, does not touch existing code)
    __init__.py
    data/                        <- data acquisition & storage
        vision_downloader.py     # aggTrades, 1s klines, futures bookTicker, funding
        storage.py               # parquet/json storage with schema
    engine/
        market_replay.py         # event-driven replay over trades/klines
        order_book_model.py      # best bid/ask + depth proxy (from bookTicker/trades)
        execution.py             # fees, slippage, latency, fill probability, partial fills
        capital.py               # prefunded / single-wallet / limited models
    strategies/
        triangular_v1.py         # cycle search on cross rates
        basis_v1.py              # spot vs futures + funding
        cross_exchange_v1.py     # (when multi-exchange data exists)
    research/
        experiments.py           # versioned experiments (ARBITRAGE_V1...)
        leaderboard.py           # results registry
        robustness.py            # fee/slippage/latency/liquidity sweeps
        stats.py                 # reuse risk_metrics + arb-specific metrics
    reports/                     # generated ARBITRAGE_*.md reports
```

## 13. Acceptance Gates for the New Subsystem

1. Point-in-time assertions fail the run on any violation.
2. Every experiment has an ID + version; nothing is overwritten.
3. No live trading, no real capital, no order submission.
4. Every result reported as GROSS vs NET (fees+spread+slippage+latency).
5. Promising models must pass OOS + robustness sweeps before status=VALIDATED.
6. If no arbitrage edge exists → state it clearly in the final report.
