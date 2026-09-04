# STRATOCRYPTO ARBITRAGE ROADMAP

**Date:** 2026-08-10 · **Phase:** RESEARCH ONLY (no live trading, no real capital, no order submission)

---

## Research Status After Full Build-Out

### TAKER arbitrage — definitively REJECTED (all honest, all negative)
| Type | Evidence | Status |
|---|---|---|
| Triangular spot (BTC/ETH, BTC/BNB) | 0 executable trades in 15.7M real buckets after 30bps | ❌ |
| Spot vs futures basis | basis never ≥10bps; 1 trade, net −15.7bps | ❌ |
| Funding carry (6 symbols, 166 days) | funding never ≥3bps/8h; all variants lose to 30bps entry costs | ❌ |
| Viral claim $0.90→$408k | arithmetically impossible + empirically unsupported | ❌ |

### MAKER triangular on perps — **first validated edge candidate (OOS_VALIDATED, conditional)**
- Cycle BTCUSDT/ETHUSDT/ETHBTC perps, maker fees 2bps/leg (6bps/cycle)
- 9 pre-registered configs positive in BOTH DEV (10d) and OOS (6d): e.g. thr8bps/1s/p0.5 → DEV +$17.05, OOS +$8.37 per $100
- Robustness boundary: breaks at per-leg fill prob p < 0.2 or 500ms windows at low p
- **The edge exists in the data; its extractability depends on passive-fill execution quality**

---

## Live Infrastructure Running

### Order-book recorder (VALIDATED — running now)
- `crypto/arbitrage/recorder.py`: futures depth@100ms L2 for btcusdt/ethusdt/ethbtc + optional trades
- Auto-reconnect, daily rotation, status file; spot depth unsupported in this environment, futures OK
- Start: `python crypto/arbitrage/recorder.py --hours 336 --depth 20`
- ~2.9k depth events/min; ≈15-25 MB/day/symbol
- **Purpose: calibrate the real queue-position fill probability (p_leg) the maker backtest depends on**

## Next Steps (priority order)

| # | Step | Why | Data needed |
|---|---|---|---|
| 1 | **Run recorder ≥ 2 weeks** | Calibrate p_leg, verify depth-constrained executable size | recorder output |
| 2 | **Order-book re-run (ARBITRAGE_ORDERBOOK_V1)** | Re-run maker backtest with REAL queue model + depth caps | recorder output |
| 3 | **Extend maker cycle to more pairs** (SOL/BNB/XRP cycles on perps) | Diversify the edge, check breadth | Vision futures aggTrades |
| 4 | **Cross-exchange maker** | Bybit/OKX perp depth (same recorder pattern, different wss) | live recording |
| 5 | **Paper-execute the maker strategy** (paper only, no real orders) | Confirm live signal latency/competition assumptions | live feeds |
| 6 | **Monthly re-validation** | The edge may decay as more makers compete — re-run V4E monthly | monthly funding + trades |

## Success Criteria Status

1. Positive net edge after all costs — ✅ (maker, conditional on execution quality)
2. Realistic execution — ⏳ depth calibration pending
3. Adequate liquidity — ⏳ depth constraint measurement pending
4. Cost/latency robustness — ✅ swept (fees 2-30bps, latency 0-500ms, fill prob 0.1-1.0)
5. Multiple periods + OOS — ✅ (DEV/OOS both positive; monthly re-validation scheduled)
6. No leakage — ✅ (PIT assertions by construction)
7. Repeatability — ✅ (89 versioned experiments, leaderboard)

## Final Rule (unchanged)

Research only. No live orders. Every number traceable to real tape. If the edge dies in real depth data — we report it and stop.
