# ARBITRAGE DATA REQUIREMENTS

**Date:** 2026-08-10
**Status:** Based on verified data.binance.vision availability + ccxt market metadata.

---

## 1. What Arbitrage Research Needs

| Research question | Data required | Granularity | Available free? |
|---|---|---|---|
| Triangular arbitrage (spot) | trades / 1s klines per leg (BTC/USDT, ETH/BTC, ETH/USDT…) | per-trade | ✅ Binance Vision aggTrades |
| Best bid/ask spread dynamics | bookTicker or trades (buyer/seller aggressor) | per-event | ✅ futures bookTicker; spot via trades aggressor flag |
| Depth / executable quantity | L2/L3 order book | per-event | ❌ NOT free — record live or buy |
| Spot↔futures basis | spot klines + futures klines + mark/index | 1s-1m | ✅ Vision (klines for both) |
| Funding rate capture | fundingRate history | 8h/4h/1h | ✅ Vision (monthly files) |
| Cross-exchange arb | per-exchange trades/book | per-event | ⚠ Binance only free; other exchanges scarce |

## 2. Data Inventory (current)

| Source | Content | Location | Format |
|---|---|---|---|
| Binance Vision | aggTrades (spot+futures) | daily/monthly zips | CSV |
| Binance Vision | 1s/1m klines (spot+futures) | daily/monthly zips | CSV |
| Binance Vision | USD-M bookTicker (best bid/ask+qty) | daily zips | CSV |
| Binance Vision | USD-M fundingRate | monthly zips | CSV |
| ccxt | market metadata (precisions, limits, fees) | live API | JSON |

## 3. What We MUST Record Ourselves (not free)

- **Spot order book (L2/L3):** WebSocket diff.stream recording (needs a live recorder running days/weeks).
- **Multi-exchange data:** other venues (Bybit, OKX, Kraken) have limited/no free archives;
  Tardis.dev / Databento commercial archives exist but are paid.

## 4. Simulation Strategy Given Constraints

1. **Trade-feed execution model:** use aggTrades to build best-bid/ask proxy:
   - buy = last trade price when aggressor=SELLER (market sell hits bid → that trade is the bid execution)
   - sell = last trade price when aggressor=BUYER (market buy lifts ask)
   - depth proxy: cumulative volume of consecutive same-side aggressive trades
2. **Pessimistic fill modeling:** taker fills at the trade prices of the OPPOSITE aggressor side;
   half-spread assumption derived from tick size where trade data is insufficient.
3. **1s klines as fallback** for spread estimation (high/low within second).
4. **Futures basis + funding:** fully reconstructable from Vision klines + fundingRate — no gaps.
5. **Order-book-only strategies:** NOT possible on free historical data → marked "DATA UNAVAILABLE"
   in the leaderboard; provide live recorder blueprint for future.

## 5. Download Plan (this phase)

| Dataset | Symbols | Period | Files |
|---|---|---|---|
| aggTrades spot | BTC, ETH, BNB (for BTC/USDT, ETH/USDT, ETH/BTC…) | 2026-07-01 → 2026-08-09 | ~40 daily zips |
| klines 1s spot | BTC, ETH (validation) | 2026-07-01 → 2026-08-09 | ~80 zips |
| klines 1m spot + USD-M | BTC, ETH | same | ~80 zips |
| USD-M fundingRate | BTC, ETH | 2026-06 → 2026-08 | 3 monthly zips |
| USD-M bookTicker | BTC, ETH (depth/BBO study) | sample days | ~10 zips |

Size estimate: BTCUSDT aggTrades ≈ 1-3M trades/day (roughly 20-80 MB/day). Storage is fine.

## 6. Known Gaps (documented honestly)

- No spot L2 depth history → executable quantity at best price cannot be proven historically.
- No multi-exchange historical feed → cross-exchange arb can only be SIMULATED with
  conservative assumptions or recorded live later.
- Funding data only monthly for full history; fine for 40-day window.
