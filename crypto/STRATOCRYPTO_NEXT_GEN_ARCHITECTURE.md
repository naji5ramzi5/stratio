# STRATOCRYPTO — NEXT-GENERATION ARCHITECTURE
## v2.0: From Bar-Based Bot to Quant Research Platform

**Date:** 2026-08-09
**Status:** Proposed (not implemented)
**Author:** Senior Quant Developer

---

## PRINCIPLES

**No Bullshit:** Every design decision must answer "does this help us find real alpha?"
**Evidence-First:** Only build components that research shows are valuable.
**Modular:** Every layer can be tested independently.
**Realistic:** Never assume perfect execution or instant fills.
**Reproducible:** Same inputs always produce same output.

---

## CURRENT SYSTEM (What We Have)

```
Binance API
   ↓
OHLCV Candles (1h bars only)
   ↓
stat_arb (cointegration + z-score)
   ↓
paper trading (single-leg)
   ↓
Telegram Alerts
```

**Problems:**
- Only 1h bars (too slow for real mean-reversion)
- No order book data
- No realistic execution
- No microstructure
- Signals tested on same data they're tuned on

---

## PROPOSED SYSTEM (What We're Building)

### Layer 1: Data Plane (New)
```
┌─────────────────────────────────────────────────────────────┐
│  DATA PLANE (collected once, used many times)               │
├─────────────────────────────────────────────────────────────┤
│  Raw Sources:                                               │
│  1. Binance OHLCV (existing, 1h/4h)                        │
│  2. Tick trades (from Binance stream)                      │
│  3. Order book snapshots (every 100ms)                     │
│  4. Funding rates (periodic)                               │
│  5. Volume profile (VWAP, cumulative volume)              │
│  6. On-chain metrics (optional, from onchain_data.py)    │
└─────────────────────────────────────────────────────────────┘
```

### Layer 2: Feature Engine (New)
```
Raw Data → Features → Store
├── Time-based: returns, volatility, momentum
├── Market-based: order imbalance, spread, liquidity
├── Cross-asset: correlation, beta, relative strength
├── Cross-pair: cointegration vector, hedge ratio
└── Regime: trend strength, volatility regime
```

**Key insight from our research:** Features must be computed from POINT-IN-TIME data only. No leakage.

### Layer 3: Signal Engine (Modified)
```
Features → Candidate Signals
           ↓
    Signal Filter (existing: AI score, regime, liquidity)
           ↓
    Entry Signal (specific asset pair, direction, size)
           ↓
    Execution Plan (limit order? market? sizing?)
           ↓
    Execution Engine (new)
```

**New requirements:**
- Every signal must have a stop-loss and take-profit computed at signal time
- Position size determined by Kelly or volatility-based sizing
- Risk engine gates every position before execution

### Layer 4: Risk Engine (Enhanced)
```
Signal → Risk Check → Approve/Reject
           ↓
    [Portfolio Level]
    - Correlation with existing positions
    - Total exposure across all pairs
    - Volatility regime sizing
    - Drawdown protection
    - Daily/weekly loss limits
           ↓
    [Asset Level]
    - Position size limits (Kelly, max %)
    - Volatility adjustment
    - Funding cost awareness
```

**New features:**
- Multi-leg risk (a pair has 2 positions)
- Correlation-aware sizing (don't stack similar trades)
- Dynamic sizing based on regime (smaller in VOLATILE, larger in MEAN_REVERTING)

### Layer 5: Execution Engine (NEW)
```
Execution Request → Execution Planner → Order Router
                           ↓
    1. Choose order type (market vs limit)
    2. Compute expected slippage from order book
    3. Simulate fill at mid-price +/- slippage
    4. Record actual fill price (with latency)
    5. Update position
                           ↓
    Track: slippage, fill ratio, latency, spread at execution
```

**Critical:** We must model the difference between signal price and execution price. This was our biggest mistake in the mean-reversion tests.

### Layer 6: Backtest / Simulation (Modified)
```
Historical Events (tick or bar)
   ↓
Point-in-Time Gate (only see data ≤ T)
   ↓
Feature Engine (compute features from allowed data)
   ↓
Signal Engine (generate candidate)
   ↓
Risk Engine (approve/reject)
   ↓
Execution Simulator (fill at realistic price)
   ↓
Update Position / Equity
   ↓
Record results
```

**Key change:** The simulation must not process all pairs at once. Process pairs sequentially, and within each pair, only process bars up to the decision time.

**Salvation for accuracy:**
- Use tick data instead of bars when available
- If using bars, process each bar one at a time, not all of them at once
- Respect the latency between signal and fill

---

## DATA STRUCTURES (Proposed)

### Historical Data Schema

```python
Candle = {
    "timestamp": int,
    "open": float,
    "high": float,
    "low": float,
    "close": float,
    "volume": float,
    "quote_volume": float,
    "trades_count": int,
    "taker_buy_volume": float,
    "taker_buy_quote_volume": float,
}

# Additional data we need
OrderBookSnapshot = {
    "timestamp": int,
    "bids": List[Tuple[float, float]],  # price, quantity
    "asks": List[Tuple[float, float]],
    "bid_depth": float,
    "ask_depth": float,
    "spread_pct": float,
}

TickTrade = {
    "timestamp": int,
    "price": float,
    "quantity": float,
    "side": "BUY" | "SELL",  # whether maker or taker
    "is_buyer_maker": bool,
}
```

### Research Database Schema

```python
Experiment = {
    "id": str,
    "name": str,
    "version": str,
    "hypothesis": str,
    "parameters": dict,
    "data_period": str,
    "train_test_split": dict,
    "results": dict,
    "status": "discovered" | "validated" | "rejected",
    "created_at": str,
    "validated_at": str,
    "notes": str,
}
```

---

## IMPLEMENTATION PLAN

### Phase 0: What Already Works (Keep)

| Component | Location | Status |
|---|---|---|
| Point-in-time safety | `real_historical_backtester.py` | ✅ Working |
| Historical data provider | `real_historical_backtester.py` | ✅ Working |
| Stat-arb baseline | `stat_arb.py` | ❌ Baseline keeps as reference, not for production |
| Risk engine | `risk_engine.py` | ⚠️ Needs correlation awareness |
| Regime detection | `regime_predictor.py` | ⚠️ Needs to use causal features only |
| Execution engine | `execution_engine.py` | ⚠️ Needs bid/ask simulation |
| Paper trading | `paper_trading.py` | ✅ Working for single-leg |
| Pair paper trading | `pair_paper.py` | ✅ Working for pairs |
| Alerting | `alerting.py` | ✅ Working |
| Dashboard | `dashboard.py` | ⚠️ Needs data and metrics |

### Phase 1: Core Alpha Research (Week 1-2)
- [ ] Build the unified data pipeline
- [ ] Implement realistic execution simulator with bid/ask
- [ ] Add point-in-time feature calculations
- [ ] Create experiment tracking database
- [ ] Write unit tests for all data flows

### Phase 2: Strategy Research (Week 2-4)
- [ ] Test mean-reversion signals with proper execution
- [ ] Test momentum on 4h bars with microstructure filters
- [ ] Test volatility breakouts with regime detection
- [ ] Test cross-asset lead-lag relationships
- [ ] Test order book imbalance predictive power
- [ ] Evaluate every strategy with OOS, walk-forward, and Monte Carlo

### Phase 3: Risk & Execution (Week 4-6)
- [ ] Build the event-driven execution engine
- [ ] Implement queue position awareness
- [ ] Add latency modeling (actual exchange ping)
- [ ] Test fill price vs signal price discrepancies
- [ ] Add funding rate awareness for perpetuals

### Phase 4: Portfolio & Validation (Week 6+)
- [ ] Multi-pair portfolio manager with correlation
- [ ] Drawdown and volatility targeting
- [ ] Continuous validation framework
- [ ] Production-quality reporting

---

## RESOURCE REQUIREMENTS

### Data
- **Historical candles:** 1h bars for all symbols in our universe (we have this)
- **Tick trades:** Would need additional download from Binance or other exchange
- **Order book:** Snapshots every 100ms for our target symbols

### Storage
- Candles: ~100MB for 10 symbols × 1000 hours × 1h bars (we have ~300MB already)
- Ticks: ~50MB for 10 symbols × 1000 hours × avg 50k trades (would add ~500MB)
- Order book: ~100MB for 10 symbols × 1000 hours × snapshots every 100ms (would add ~1GB)

### Compute
- Backtests: Process in parallel, each pair independently
- Research: Run multiple experiments in background jobs

---

## DESIGN DECISIONS

### Use Existing vs New?

**Keep:**
- Historical data downloader (works)
- Point-in-time check (works)
- Risk engine with Kelly sizing (good for single-leg)
- Regime predictor (useful but simplify)
- Telegram alerts (works)
- Dashboard (works)

**Rebuild:**
- Execution engine (currently assumes instant fill)
- Features computation (currently uses some non-causal functions)
- Data pipeline (should be event-driven)
- Research framework (should track all experiments and hypotheses)

### External Libraries?

**Use existing open-source (MIT/Apache 2.0):**
- `vectorbt` for fast backtesting (Apache 2.0)
- `backtrader` for reference architecture (GPL)
- `statsmodels` for statistics (BSD)
- `pandas/numpy` for data ops (BSD)

**Build ourselves:**
- Data provider abstraction (we have the raw data)
- Feature engine (custom logic that must be causal)
- Execution simulator (specific to our fees/microstructure model)
- Risk engine (needs to integrate with our correlation model)

---

## EXPECTED OUTCOME

By the end of this work:**

1. **We can test any hypothesis** on our data without surprises from the current system
2. **Every result is trustworthy** because it's point-in-time safe and uses realistic execution
3. **We know what works** because we have a structured research process
4. **We know why it works** because we track hypothesis → experiment → result
5. **We don't waste resources** on ideas that have no evidence

---

## FINAL JUDGMENT

**Status:** PROPOSED (not yet implemented)

**Priority:** P0 (high) — this is the foundation for any serious research

**Effort estimate:** 2-3 days of focused work

---

**Report Status:** Complete
**Next Step:** Begin implementing the Data Plane
