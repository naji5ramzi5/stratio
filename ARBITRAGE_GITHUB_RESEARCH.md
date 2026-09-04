# ARBITRAGE GITHUB RESEARCH

**Date:** 2026-08-10
**Method:** GitHub API (stars current), READMEs, official docs, direct probing of data.binance.vision S3 bucket.
**License note:** concepts borrowed, NO code copied, NO license violations.

---

## 1. Mature Engines

### hftbacktest — nkaz001/hftbacktest (≈4,350★ · MIT · Rust core + Python API)
- **Data:** full tick feed, L2 market-by-price AND L3 market-by-order, trades; multi-asset/exchange.
- **Order book:** full L2/L3 replay; queue-position models (RiskAverseQueueModel, ProbQueueModel).
- **Execution:** feed + order-entry + response latency (Constant/Intp models); NoPartialFillExchange /
  PartialFillExchange; explicit "no market impact" assumption; taker fills at best price.
- **Backtest:** event-driven tick-by-tick replay, latency-aware — reference standard for HFT backtests.
- **Converters:** data.binance.vision, Tardis, Databento, Bybit, Hyperliquid, MEXC, diff orderbook snapshots.
- **Weaknesses:** no arbitrage-cycle abstraction; Rust/Numba learning curve.
- **Integration:** HIGH — reuse its concepts (event stream, latency models, queue models); not a dependency here.

### NautilusTrader — nautechsystems/nautilus_trader (≈25,400★ · LGPL-3.0 · Rust core + Python API)
- Deterministic event-driven engine; L2/L3 book with delta replay; per-venue latency models; partial fills; FOK/GTX.
- **Weaknesses:** LGPL (modifications must stay LGPL); heavier abstraction.
- **Integration:** reference only.

### Hummingbot — hummingbot/hummingbot (≈19,400★ · Apache-2.0 · Python)
- **Cross-exchange arb done right:** `strategy_v2/` controllers+executors incl. arbitrage_executor,
  xemm_executor; per-connector L2 order-book trackers; budget checks; fee schemas per exchange.
- Backtest: event-driven replay of recorded book/trade data; fill timing coarser than hftbacktest.
- **Integration:** HIGH (Apache-2.0) — borrow fee-schema + executor concepts.

### Freqtrade — freqtrade/freqtrade (≈53,100★ · GPL-3.0)
- Bar-based; no order book in backtest; no latency/partial-fill modeling; `lookahead-analysis` audit tool is a great concept.
- **Integration:** concepts only (GPL blocks code reuse).

### ccxt — ccxt/ccxt (≈43,600★ · MIT)
- Unified API over 100+ exchanges: fetchTrades, fetchOrderBook (L2), fetchFundingRate, market metadata
  (precisions, limits, per-market fees) via `load_markets()`.
- **Integration:** USE AS-IS for exchange metadata + live/paper data.

### backtrader / vectorbt / jesse / CryptoSignal
- All bar-based, no microstructure. Design concepts only (sizers, significance testing, vectorized sweeps).
- vectorbt license (Commons Clause) not business-friendly — skip as dependency.

## 2. Triangular / Cross-Exchange Arb Projects (all live-scan bots)

| Project | ★ | License | Notes |
|---|---|---|---|
| zlq4863947/triangular-arbitrage | 671 | GPL-3.0 | Binance cycle scanner, depth-aware sizing |
| tiagosiebler/TriangularArbitrage | 620 | no license | detection from best bid/ask tickers |
| ericjang/cryptocurrency_arbitrage | 859 | no license | classic multi-exchange bot (2017) |
| kelvinau/crypto-arbitrage | 845 | MIT | cross-exchange arb, okx/binance |
| eugenioclrc/binance-crypto-triangular-arbitrage | 289 | no license | includes depth/fee simulation docs |
| karthik947/BinanceTriangularArbitrage_v2 | 268 | MIT | live v2 |

**Key finding:** common pattern = graph-of-markets cycle detection on best bid/ask + profit threshold after fees.
**GAP: NONE of them provide event-driven HISTORICAL arbitrage backtesting.** This is our original contribution space.

## 3. Binance Free Historical Data — VERIFIED

| Data | Spot | USD-M Futures | Granularity |
|---|---|---|---|
| aggTrades | ✅ | ✅ | per trade (µs timestamps since 2025-01-01) |
| trades | ✅ | ✅ | per trade |
| klines | ✅ | ✅ | **1s to 1mo** |
| bookTicker (best bid/ask + qty) | ❌ | ✅ | daily files, event-level |
| bookDepth (aggregated depth % from mid) | ❌ | ✅ | daily files |
| fundingRate | ❌ | ✅ | monthly only |
| index/mark/premium klines | ❌ | ✅ | standard intervals |
| **Order book (L2/L3)** | ❌ | ❌ | **NOT available free** |

- Endpoint: `https://data.binance.vision` (S3 bucket), daily + monthly zips, `.CHECKSUM` sha256 files.
- Helper: `binance/binance-public-data` (MIT, ≈2,400★).
- Spot order-book history must be recorded live (WebSocket diff streams) or purchased (Tardis.dev, Databento).

## 4. Engineering Conclusions

1. **Use:** ccxt (metadata), data.binance.vision (bulk history: aggTrades + 1s klines + futures bookTicker/funding).
2. **Borrow as design (no code):** hftbacktest latency/queue concepts; Hummingbot fee-schema/executor structure;
   freqtrade lookahead-audit idea; cycle-enumeration idea from triangular bots.
3. **Build ourselves (the gap):** numpy-vectorized, event-driven, trade-feed arbitrage simulator with
   fee + slippage + latency + fill-probability + two-leg-risk + capital models, fully point-in-time safe.
4. **Cython/LOB libs:** no meaningful OSS exists; if speed becomes critical use Numba (already installed).
