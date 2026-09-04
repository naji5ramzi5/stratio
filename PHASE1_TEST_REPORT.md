# PHASE 1 — TEST REPORT

Run: `python -m unittest discover -s tests` from `crypto/`

## Result

**207 tests ran — OK (1 skipped)** · skipped = pre-existing calibration test (unrelated to Phase 1).

## New test files (7)

| File | Coverage |
|---|---|
| `test_market_events_model.py` | event construction, frozen immutability, factory defaults, timestamp conversions (unit + None + bad unit) |
| `test_market_events_validation.py` | 20 checks: envelope, timestamp types, missing-receive flag (and proof it is not filled), price/qty NaN/zero/negative, crossed quote, book sort order, crossed book, seq validity |
| `test_market_events_adapters.py` | spot row (µs, no header), futures row (ms→µs, header), bool variants, recorder depth JSONL, bookTicker→QUOTE |
| `test_market_events_orderbook.py` | full snapshots (replace, top-N, qty-0 delete), partial top-N semantics (replace/keep/prune), diff updates, sequence rewind/duplicate, wrong-symbol, non-book event, determinism (same events ⇒ same state), version counter |
| `test_market_events_replay.py` | advance boundaries (boundary event not lost), future events unconsumed, rewind rejected, non-monotonic rejected, duplicate trade seq rejected, equal-timestamp ordering, advance_by, book built through replay, latest-state-only visibility, crossed book → ReplayPITError, structural no-peek guarantee |
| `test_market_events_store.py` | ingest (spot + futures header), sorting, idempotency (same checksum), changed source re-ingest, unknown market, empty zip, range queries ([start,end) incl. exact-end exclusion), missing day, list_days, manifest, replay |
| `test_market_events_features.py` | MarketState from book / from None, 3 calculators, NaN when no book, compute_all keys, replay→feature end-to-end |

Plus `test_market_events_integration.py` — real-data tests against the
downloaded datasets (skip when absent, so a fresh checkout still passes):

- spot BTCUSDT day (243,382 agg trades): all events valid, monotonic ts,
  trade semantics, full-day replay, 1-minute PIT bar-walk, rewind rejection
- futures day (header/ms parse): 2026-range sanity, all valid
- recorder depth day (2026-08-10 BTCUSDT): all valid, book replay,
  determinism (identical snapshot on second replay)
- event store vs direct stream: identical replay counts, range query correctness

## Bugs found & fixed by the tests

1. **Replay boundary loss** — `for ... break` on the underlying iterator
   pulled the next event before the break, silently dropping it (243,381/243,382
   events consumed). Fixed with `_peek`/`_take` lookahead buffer.
2. **Stale level index** — `OrderBook._reindex()` ran after `_check_crossed()`,
   so best-level reads used stale lists (IndexError / wrong best). Reindex now
   precedes crossed-book checks.
3. **Partial-snapshot prune reference** — pruned against the whole book's
   best level instead of the event's own visible top; wrong levels removed.
4. **Range query end bound** — `searchsorted(..., side="right")` included
   events at the end boundary; `[start, end)` semantics fixed with `side="left"`.

## Compatibility

All pre-existing tests (stat_arb, risk, leakage, regime, pairs, phase3,
quant_x, state_store, tracker, calibration, integration) pass unchanged.
No existing file was modified.
