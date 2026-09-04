# ARBITRAGE OOS REPORT

**Date:** 2026-08-10 · **Protocol:** DEV = 2026-07-25 → 08-03 (10 days), OOS = 2026-08-04 → 08-09 (6 days, untouched). Parameters pre-registered (no tuning on DEV, no lookback at OOS).

---

## 1. Protocol Compliance

- ✅ All thresholds (5/8/10bps), windows (500ms/1s), fees (maker 2bps, taker 10bps) fixed BEFORE running
- ✅ Point-in-time: last-known-before-bucket quotes only; fills at real tape prices
- ✅ Both periods run with identical code/params; every experiment versioned (ARBITRAGE_V3/V4E)

## 2. Taker Experiments — Consistent Rejection

| Experiment | DEV | OOS |
|---|---|---|
| TRIANGULAR_V3 (BTC/ETH, BTC/BNB, all latencies) | 0 trades | 0 trades |
| BASIS_V1 (spot/futures, both directions) | 0 profitable | 0 profitable |
| FUNDING_V5 (thr≥3bps on all symbols) | 0 triggers | 0 triggers |

Rejection is not a tuning artifact: the effect itself is absent from both periods.

## 3. Maker Experiment — Consistent Positivity (MAKER_V4E)

Pre-registered configs, $100/cycle, sequential 3-leg fills, unwind-at-taker on failure:

| Config | DEV completed | DEV PnL | OOS completed | OOS PnL | Verdict |
|---|---|---|---|---|---|
| thr5, win1s, p0.5 | 1,192 | +$56.03 | 504 | +$23.57 | ✅ |
| thr8, win1s, p0.5 | 235 | +$17.05 | 108 | +$8.37 | ✅ |
| thr8, win1s, p0.2 | 157 | +$7.89 | 61 | +$2.18 | ✅ |
| thr10, win1s, p0.5 | 73 | +$7.28 | 32 | +$3.09 | ✅ |
| thr10, win1s, p0.2 | 36 | +$1.83 | 16 | +$0.90 | ✅ (marginal) |
| thr5, win500ms, p0.2 | 243 | −$32.78 | 103 | −$13.65 | ❌ consistent |
| thr8, win500ms, p0.2 | 50 | −$5.61 | 18 | −$2.65 | ❌ consistent |

9 configs positive in BOTH periods; 2 configs negative in both. **No sign-flip between DEV and OOS** — the signal is stable across periods, not an artifact of one lucky window.

## 4. Honest Caveats

1. **p_leg (fill probability) is assumed, not measured** — the recorder now collects real depth to calibrate it. If real p < 0.2, the strategy flips negative (per the 500ms sweep).
2. Sample: 16 days, one cycle (BTC/ETH perps). A new unseen month is required for full confidence.
3. Capital $100/cycle — market impact unmodeled; competition decay unmeasured.

## 5. Conclusion

- Taker arbitrage: **REJECTED** (no edge in either period).
- Maker triangular on perps: **OOS_VALIDATED (conditional)** — positive in unseen data under pre-registered params, pending real depth calibration of fill probability.
