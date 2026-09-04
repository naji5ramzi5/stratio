# PHASE 2 — RESEARCH AUDIT

**Date:** 2026-08-10 · **Scope:** full inspection of the codebase before
building the feature-research layer. No code modified.

## 1. What already exists

| Layer | Location | State |
|---|---|---|
| Event model + validation | `crypto/market_events/{model,timestamps,validation}.py` | STABLE (Phase 1, approved) |
| Order-book engine | `crypto/market_events/orderbook.py` | STABLE |
| PIT replay engine | `crypto/market_events/replay.py` | STABLE, fail-loud |
| Event store (npz + manifests) | `crypto/market_events/store.py` | STABLE; 48 day-files cached |
| Feature interface (protocol + 3 example calculators) | `crypto/market_events/features.py` | STABLE — the interface Phase 2 must use |
| Trade-tape quote proxy (PIT-safe) | `crypto/arbitrage/replay.py` (`build_quote_series`, `reindex_quotes` — searchsorted `side='right' - 1`) | Reusable as-is |
| V4D/V4E maker-triangular simulator | `crypto/arbitrage/maker_seq.py` | Read-only reference for reproduction |
| Cost model | `crypto/arbitrage/execution.py` | `FeeConfig`: futures taker 5 bps/leg, futures maker 2 bps/leg, spot taker 10 bps/leg |
| V4D/V4E results (DEV/OOS protocol) | `crypto/ARBITRAGE_V4D_MAKER_SEQ.json`, `crypto/ARBITRAGE_V4E_MAKER_DEVOOS.json` | DEV = 07-25..08-03, OOS = 08-04..08-09 |
| Existing "feature system" | `crypto/quant_x/` | kline heuristics (fractals/FVG/Wyckoff) — NOT event-level; documented heuristics, not reusable as a research feature engine |
| Backtester | `crypto/fast_backtest.py` (+ `backtester.py`) | OHLCV-level, 1h bars; PIT via `ts <= end_ts` mask |
| Tests | `crypto/tests/` (207, all passing) | Phase 1 + legacy |

## 2. Data availability (verified)

| Dataset | Symbols | Period | Events | Type | In Phase-1 store? |
|---|---|---|---|---|---|
| Futures aggTrades (USDT-M) | BTCUSDT, ETHUSDT, ETHBTC | 2026-07-25 → 2026-08-09 (16 d) | BTC 12.27M, ETH 11.52M, ETHBTC ~1.9M | TRADE (µs) | BTC/ETH yes; **ETHBTC not yet ingested** |
| Spot aggTrades | BTCUSDT | 16 d | 9.18M | TRADE (µs) | yes |
| Order-book depth (top-20, 10 Hz) | BTCUSDT, ETHUSDT, ETHBTC | **2026-08-10 only, ≈28 min** | 16.6k / 16.6k / 1.5k | BOOK (partial) | recorder JSONL (not npz) |

Timestamp resolution: **microseconds** (trades), ms→µs (depth). Research grid: 60 s (1440 samples/day/symbol).

## 3. What can be reused

1. Phase 1 store (load once per day, never re-parse zips) — performance step 20.
2. Phase 1 `ReplayEngine` (PIT guarantees) for event-level consumption; vectorized PIT slicing for the 60 s grid (both paths, cross-checked).
3. `arbitrage/replay.build_quote_series` + `reindex_quotes` for V4D reproduction (same PIT-safe proxy, new data path).
4. `arbitrage/maker_seq.simulate` for V4D/V4E re-run (identical params; new data path).
5. `execution.FeeConfig` costs; V4E DEV/OOS split protocol (07-25..08-03 / 08-04..08-09).

## 4. What is missing (must be built)

- Stateful, deterministic, vectorized feature calculators on the Phase 1 interface
- Separate future-return label engine (labels MAY use future; features MUST NOT)
- Feature/label alignment with explicit no-leak tests
- Statistical engine (IC, Spearman, directional accuracy, conditional returns, monotonicity)
- Quantile engine (fixed quintiles, no boundary tuning)
- Predictive-decay analysis (H = 60/300/900/1800 s, all reported)
- Cost engine (gross vs net; round-trip taker 5 bps/side + spread + stress slippage)
- Regime classifier (vol terciles; trend/ranging via return sign consistency) — from trade returns, 60 s grid
- Cross-asset analysis (BTC↔ETH futures, same grid)
- DEV/OOS harness + experiments ledger (multiple-testing accounting)
- V4D/V4E replay-path reproduction + equivalence check vs old loader path

## 5. Leakage risks identified (to be guarded)

1. **Forward-fill inside the sample bucket**: feature at T must use only trades with `ts <= T` (`searchsorted(..., side='right')` on the FULL event array; slice `[0, idx)`). Labels use `ts <= T+H` in a separate engine.
2. **Same-bucket reversal**: last-trade price at T must not see trades at `T+ε` (bucket-aligned grid; boundary tests required).
3. **Equal timestamps**: duplicate-ts trades must not be split between feature/label sides inconsistently — tested.
4. **Overnight continuity**: labels crossing day boundaries use the next day's data (all 16 days loaded as one concatenated series per symbol); documented.
5. **ETHBTC thinness**: forward-fill gaps; missing-value policy = NaN, never fabricated.
6. **Old loader path**: `load_aggtrades` sorts by ts — safe; new store also sorted — equivalence test.

## 6. Computational constraints

- Pure-Python dataclass replay ≈ 105k events/s (Phase 1 benchmark). 33M events → ~5 min full pass.
- Feature generation must be vectorized over per-day arrays (cumsum / rolling via numpy) on the 60 s grid: 16 d × 3 symbols ≈ 69k samples × ~20 features.
- Load-once discipline: one `store.load_day` per (symbol, day) per run; results cached to parquet/JSON on disk for the analysis stage.

## 7. Pre-registered decisions (locked before any analysis)

- **Grid**: 60 s UTC-aligned samples; feature window ≤ 3600 s; labels H ∈ {60, 300, 900, 1800} s.
- **Split**: DEV = 07-25..08-03 (10 d) · OOS = 08-04..08-09 (6 d) — identical to V4E protocol.
- **Quantiles**: fixed quintiles Q1..Q5 by feature value; boundaries never selected post-hoc.
- **Promising rule (DEV-only, locked before OOS)**: |Spearman IC| ≥ 0.02 at H=300 s with **consistent sign on ≥2 of 3 trade symbols**, or |IC| ≥ 0.03 on any single symbol — else REJECTED.
- **Costs** (futures, project values): round-trip = 2 × taker 5 bps + spread proxy (measured per day) + stress slippage 1 bps/side. Reported gross and net separately.
- **No ML, no parameter optimization, no V4D/V4E changes.**
