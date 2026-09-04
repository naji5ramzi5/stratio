# ARBITRAGE BACKTEST REPORT

**Date:** 2026-08-10 · **Data:** Real Binance aggTrades (spot + USD-M futures), 2026-07-25 → 2026-08-09 (16 days, 605.8 MB), funding history 166 days × 6 symbols · **Engine:** `crypto/arbitrage/` (PIT-safe, numpy-vectorized)

---

## 1. Experimental Setup

| Parameter | Value |
|---|---|
| Signal granularity | 100ms buckets (triangular), 1s (basis) |
| Fees | spot taker 10bps/leg; futures taker 5bps/leg; futures maker 2bps/leg |
| Cycle costs | taker: 30bps; maker: 6bps |
| Latency scenarios | 0 / 50 / 100 / 250 / 500 ms (+ 200ms decision latency in V4D) |
| Notional | $100/cycle, prefunded, no compounding |

## 2. TAKER Strategies — ALL REJECTED (22/22 configs)

| Experiment | Candidates | Executable | Result |
|---|---|---|---|
| TRIANGULAR_V1 (BTC/ETH, BTC/BNB spot, taker 30bps) | 1 | **0** | REJECTED |
| BASIS_V1 (spot vs futures, both directions, 30bps) | 1 | 0 | REJECTED (−15.7bps on the only trade) |
| FUNDING_V5 (cash-and-carry, 6 symbols, 166 days) | 0 @ thr3bps | 0 | REJECTED (funding never ≥3bps/8h; all variants negative after 30bps costs) |

**Cross-rate characterization (spot, 15.7M buckets):** mean −1.4…−2.0bps gross; ≥5bps = 0.14–0.94%; ≥10bps ≈ 0 (2 buckets in 7.8M for BTC/ETH).
**Basis:** futures persistently −4.4bps below spot; |basis| ≥10bps = 0% of time.
**Funding:** max observed +1.96bps/8h (BNB); annualized 1.2–2.8% APR — far below entry costs.

## 3. MAKER Strategy on Perps — PROMISING (first positive edge in the project)

Cycle: BTCUSDT + ETHUSDT + ETHBTC **perps**, maker fee 2bps/leg → **6bps/cycle**. Passive fills at the touch, sequential 3-leg model with fill-window and unwind-at-taker on failure.

| Config | DEV (10d) PnL | OOS (6d) PnL | Verdict |
|---|---|---|---|
| thr5bps, win 500ms, p0.5 | +$38.5 | +$16.7 | ✅ both positive |
| thr5bps, win 1s, p0.5 | +$56.0 | +$23.6 | ✅ both positive |
| thr5bps, win 1s, p0.2 | +$5.0 | +$1.4 | ⚠ marginal |
| thr8bps, win 1s, p0.2 | +$7.9 | +$2.2 | ✅ both positive |
| thr8bps, win 1s, p0.5 | +$17.1 | +$8.4 | ✅ both positive |
| thr10bps, win 1s, p0.5 | +$7.3 | +$3.1 | ✅ both positive |
| thr5bps, win 500ms, p0.2 | −$32.8 | −$13.7 | ❌ breaks at p<0.2 |

**Robustness boundary:** profitable when per-leg fill probability p ≥ ~0.2/bucket with 1s windows (P(all 3 legs) ≈ 0.49 at p=0.2 over 10 buckets); negative at p=0.1 or 500ms windows with p≤0.2. Latency 200ms has minor impact. **This is execution-quality dependent — p must be calibrated from real depth (recorder).**

**Gross distribution (13M perp buckets):** net-after-6bps mean −8.5bps (structurally negative — perps cross cycles cheap), but tail ≥5bps = 0.9% (116k buckets), ≥8bps = 0.16% (21k), ≥10bps = 0.057% (7.4k), ≥15bps = 0.003% (366).

## 4. Verdict

1. **Taker arbitrage on Binance (spot triangular, basis, funding): DEAD.** Consistent with industry reality.
2. **Maker triangular on perps: genuine, repeatable in DEV+OOS, but dependent on passive-fill execution quality.** The 9 OOS_VALIDATED configs are the project's first validated edge candidates.
3. **Unknowns:** real queue-position fill probability (→ live depth recorder), competition decay, market impact above $100/cycle.

## 5. Data/Model Limitations (honest)

- Order-book depth NOT in historical data → p_fill is a calibration assumption, not a measurement (recorder running to fix this)
- No multi-exchange data → cross-exchange arb untested
- Funding interval fixed 8h; mark-price vs index premium unmodeled in V5
- 16-day window; single cycle pair (BTC/ETH)
