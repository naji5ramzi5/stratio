# PHASE 1 — ARCHITECTURE AUDIT

**Date:** 2026-08-10 · **Scope:** Data-layer inspection of STRATOCRYPTO before introducing the event-driven market-data foundation. No code modified.

---

## 1. Data Flow Today

```
Binance REST (api.binance.com / fapi.binance.com)
   │
   ├─ fetch_klines() ──► historical_data/<SYM>_1h.json   (raw Binance kline arrays, ms)
   ├─ market_data.py  ──► market_cache.json              (TTL JSON cache, unix s)
   ├─ crypto_market_data.py ─► funding/OI/long-short      (uncached REST)
   │
Binance Vision (S3 zips)
   │
   ├─ arbitrage/vision.py + storage.py ──► arbitrage/data/<market>/aggTrades/<SYM>/<day>.zip
   │
Live WebSocket (fstream.binance.com)
   │
   └─ arbitrage/recorder.py ──► arbitrage/data/orderbook/<date>/<sym>_depth.jsonl  (ms, U/u)
```

## 2. Current Market-Data Models (exact schemas)

| Layer | Model | Timestamp unit | Sequence | Notes |
|---|---|---|---|---|
| OHLCV file | JSON array of 12-field Binance klines, positional access | **ms** | none | 1000 bars/symbol only (downloader loop bug: 4× same latest chunk) |
| OHLCV backtest | numpy `[ts,o,h,l,c,vol]` float64 | **ms** | none | `fast_backtest._load_data`; PIT via `ts <= end_ts` mask (`fast_backtest.py:50-57`) |
| Live cache | dict `{"key": {"ts": sec, "data": ...}}` | **unix s** | none | non-atomic rewrite (`market_data.py:50`) |
| aggTrades (spot) | CSV: `agg_id,price,qty,first_id,last_id,ts_us,buyer_maker,best_match` | **µs** (raw) | agg_id | no header |
| aggTrades (futures) | CSV w/ header: `...,transact_time,is_buyer_maker` | **ms** (converted →µs) | agg_id | header + lower-case bools |
| Recorder depth | JSONL: `{t,E-ish ms, U, u, b:[[p,q]], a:[[p,q]]}` | **ms** | **U/u** | partial top-20 book |
| Execution | `Order` dataclass (`execution_engine.py:31-46`) | str timestamp | order_id | no book model |

**Key finding — three different timestamp units (ms / µs / unix-s) and no canonical event schema.** The arbitrage layer already normalizes trade data to µs; this phase makes that normalization explicit and platform-wide.

## 3. Point-in-Time Discipline Today

- `fast_backtest.get_candles_up_to` masks `ts <= end_ts` (ad hoc, per-call re-filter over full array — `fast_backtest.py:50-57`).
- `arbitrage/replay.reindex_quotes` uses `searchsorted(side="right") - 1` (PIT by construction).
- Leakage tests exist for the feature pipeline (`tests/test_leakage.py`); no platform-wide guarantee layer.

## 4. Storage / Caching Today

- Plain JSON everywhere; one atomic-write helper (`utils/state_store.save_json`, tmp+replace).
- No DB. `market_cache.json` is a process-local TTL dict with non-atomic full rewrites.
- Expensive pattern: backtester re-filters entire datasets per event loop iteration (`get_candles_up_to` full mask each call) — the exact inefficiency Phase 1 should remove for the event layer.

## 5. Tests Today

- Framework: **unittest** (no pytest installed). Discovery: `python -m unittest discover -s tests` from `crypto/`.
- 12 files ≈ 90+ tests covering: stat_arb, risk, leakage, regime, pairs, phase3, quant_x, state_store, tracker, calibration, integration.
- Standalone root-level `test_*.py` files exist outside `tests/` (not part of suite).

## 6. Constraints for Phase 1

1. Do NOT touch: `fast_backtest.py`, `stat_arb.py`, strategy params, execution engine, any OHLCV code path.
2. Do NOT add DBs or heavy deps (numpy/pandas/JSONL suffice; numba available if needed).
3. Do NOT create live-trading functionality.
4. New layer must coexist alongside OHLCV; existing tests must keep passing.

## 7. Phase 1 Design Decisions (pre-registered)

| Decision | Choice | Rationale |
|---|---|---|
| Canonical timestamp | **int64 µs since epoch, UTC** | matches arbitrage layer; converts exactly from ms (×1000) and unix-s (×1e6); no precision loss |
| Event time vs receive time | separate fields; receive never auto-filled with `now()` — flagged if missing | spec Step 3 |
| Event model | typed dataclasses (Trade/Quote/BookSnapshot/BookUpdate) + common envelope | exchange-neutral, no Binance JSON leakage |
| Symbols | canonical uppercase `BTCUSDT` | existing convention |
| Exchange IDs | `binance_spot`, `binance_futures_um` | distinct, extensible |
| Event store | per-symbol-per-day `.npz` derived cache + manifest; raw zips remain authoritative | no DB; simple; partitioned; sorted; deterministic |
| Order book | in-memory state engine: partial-book (replace) + diff (level-update) modes; sequence guard; crossed-book detection | recorder emits partial top-20; diffs needed for generality |
| Replay | `EventReplayEngine` with monotonic clock + PIT assertion (fail loud) | spec Steps 8-9 |
| Features | interface only (`MarketState → FeatureCalculator → FeatureSet`) + 3 example calculators to validate the interface | spec Step 13 |
| Tests | unittest, in `crypto/tests/`, following repo convention | matches suite |
