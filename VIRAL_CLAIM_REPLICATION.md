# VIRAL CLAIM REPLICATION — $0.90 → $408,000 in ~2 days

**Date:** 2026-08-10 · **Method:** pure math + measured market data (ARBITRAGE_V1_DIAGNOSTIC, ARBITRAGE_V2_BASIS). The claim was treated ONLY as a hypothesis; nothing was optimized to reach the target.

---

## 1. The Arithmetic

- Target ratio: 408,000 / 0.90 = **x453,333**
- Required cumulative log-return over 48h: **13.02**

Required average net return per cycle at plausible frequencies:

| Frequency | Cycles in 48h | Required net edge/cycle |
|---|---|---|
| 1 trade/min | 2,880 | 45.3 bps |
| 2 trades/min | 5,760 | 22.6 bps |
| 5 trades/min | 14,400 | 9.0 bps |
| 10 trades/min | 28,800 | 4.5 bps |
| 60 trades/min | 172,800 | 0.75 bps |

Cost floor on Binance spot (3 legs × taker 10bps): **30 bps/cycle** → even at 1 trade/min you need 45bps net = 15bps over costs; at 10/min you need 4.5bps net = **negative gross**. The claim is arithmetically incompatible with measured market behavior.

## 2. What the Real Market Offered (16 days, 15.7M quote buckets)

| Metric | BTC/ETH cycle | BTC/BNB cycle |
|---|---|---|
| Gross cross-rate mean | -1.98 bps | -1.43 bps |
| Buckets ≥ 5bps gross | 11,275 (0.14%) | 74,281 (0.94%) |
| Buckets ≥ 10bps gross | **2 (≈0%)** | 770 (0.0097%) |
| Executable after 30bps fees | **0** | **0** |

Spot-futures basis: max +0.96bps (BTC), max +4.75bps (ETH) — **never ≥ 10bps**; futures persistently BELOW spot by ~4.4bps (unprofitable direction for cash-and-carry too).

## 3. Additional Constraints

1. **Minimum order sizes:** $0.90 cannot trade BTCUSDT (min notional $5-10; step sizes). The claim's starting capital is below exchange minimums.
2. **Fees:** even if $0.90 deployed, 30bps/cycle on a growing notional would need liquidity that doesn't exist: at 1bps net (unachieved) you'd need 130,244 profitable cycles in 48h ≈ **45/min** with no depth cap — but top-of-book depth caps executable notional at a few hundred dollars.
3. **Order-book depth:** not available in free history; nothing in the tape suggests sustained depth at mispriced levels.
4. **No persistence:** max consecutive window with ≥5bps gross edge: 26.8s (BTC/ETH), 81.5s (BTC/BNB) — in 16 days.

## 4. VERDICT

> **"THE CLAIM IS NOT SUPPORTED BY THE AVAILABLE HISTORICAL DATA."**
> **STATUS: CLAIM NOT REPRODUCIBLE UNDER CURRENT MARKET CONDITIONS.**

The combination of (a) sub-10bps measured gross deviations, (b) 30bps cost floor, (c) min-order constraints on $0.90, and (d) vanishing liquidity at depth makes x453,333 in 48h mathematically impossible from Binance spot/futures market data.
