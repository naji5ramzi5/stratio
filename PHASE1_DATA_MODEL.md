# PHASE 1 — DATA MODEL

Canonical, exchange-neutral event schema for the STRATOCRYPTO data foundation.

## 1. Timestamp Model

| Concept | Field | Unit | Rule |
|---|---|---|---|
| event time | `event_timestamp` | **int64 µs UTC** | the timestamp the exchange attributes to the event |
| receive time | `receive_timestamp` | int64 µs UTC (or None) | when WE received/ingested it — **never auto-filled**; missing is a validation defect |
| sequence | `sequence_number` | exchange seq (int) | trade agg id, book U/u, bookTicker update id; None if source has none |

Precision is lossless for every unit Binance publishes (s → µs ×1e6, ms → µs ×1e3, µs identity). Conversions live in `market_events/timestamps.py`; adapters convert at the boundary, so the model only ever sees µs.

## 2. MarketEvent Envelope

```
MarketEvent(
    event_type        EventType           TRADE | QUOTE | ORDER_BOOK_SNAPSHOT | ORDER_BOOK_UPDATE
    exchange          str                 "binance_spot" | "binance_futures_um"
    symbol            str                 canonical uppercase, e.g. "BTCUSDT"
    event_timestamp   int                 µs
    receive_timestamp int | None          µs (never invented)
    sequence_number   int | None          exchange sequence
    source            str                 provenance, e.g. "binance_vision_aggtrades"
    payload           object              one of the payloads below
)
```

## 3. Payloads

| Payload | Fields | Notes |
|---|---|---|
| `TradePayload` | `price`, `quantity`, `is_buyer_maker`, `trade_id` | agg-trade level |
| `QuotePayload` | `bid`, `ask`, `bid_size`, `ask_size`, `quote_id` | best bid/ask snapshot (bookTicker) |
| `OrderBookSnapshotPayload` | `bids`, `asks`, `first_update_id`, `final_update_id`, `partial` | `partial=True` → top-N replacement (recorder depth); `False` → full book (REST depth) |
| `OrderBookUpdatePayload` | `bid_updates`, `ask_updates`, `first_update_id`, `final_update_id` | diff semantics: qty 0 = delete level |

All prices/qty are float64-canonical floats. Events are immutable (`frozen=True`).

## 4. Binance → Canonical Adapter Map

| Source | Raw unit | Event | Conversion |
|---|---|---|---|
| Vision aggTrades spot (no header) | µs in col 5 | TRADE | identity |
| Vision aggTrades futures (header) | ms in `transact_time` | TRADE | ×1000 |
| Recorder depth JSONL | ms in `t`, U/u seqs | ORDER_BOOK_SNAPSHOT (partial) | ×1000 |
| Futures bookTicker | ms in `T`, `u` | QUOTE | ×1000 |
| REST depth | ms | ORDER_BOOK_SNAPSHOT (full) | ×1000 |

Adapters are pure functions `raw row → MarketEvent`; the raw row is never re-interpreted elsewhere.

## 5. Validation Rules (never silently repair)

- envelope: typed event, non-empty exchange/symbol
- `event_timestamp` positive int (float/bool/negative rejected)
- `receive_timestamp` missing → defect `missing_receive_timestamp`; never filled from wall clock
- price > 0 finite; quantity >= 0 finite (both payload sides)
- QUOTE: ask > bid
- book: levels price > 0, qty >= 0; bids descending; asks ascending; best bid < best ask
- sequence >= 0 when present

Every defect carries a machine-readable reason (`market_events/validation.py`).

## 6. Order-Book State Semantics

- SNAPSHOT (full): clear both sides, apply levels.
- SNAPSHOT (partial/top-N): upsert levels present; **drop stale levels better than the new visible top** (the exchange did not show them, so they cannot exist); keep levels worse than the top (unknown → previous state).
- UPDATE (diff): upsert; qty 0 deletes the level.
- Sequence guard: rewind or duplicate sequence → `OrderBookSequenceError`.
- Crossed book after apply → `OrderBookCrossedError`.
- Deterministic: identical event sequence ⇒ identical final state (tested).

## 7. Replay / Point-in-Time Contract

`ReplayEngine` (market_events/replay.py):

- the **only** read path is `advance_to(ts)` / `advance_by(delta)`; events are consumed (not peeked) — a buffered lookahead guarantees no event is lost at a boundary
- `current_event_time`, `current_receive_time`, `current_market_state`, `event_sequence` are the only exposed state
- rewind → `ReplayClockError`; non-monotonic stream → `ReplayClockError`; duplicate trade seq → `ReplayPITError`; book rejection → `ReplayPITError`
- it is structurally impossible to read an event whose timestamp is beyond the clock

## 8. Event Store

- Derived cache only; raw Vision zips / recorder JSONL stay authoritative
- Layout: `market_events/cache/{market}/{SYMBOL}/{day}.npz` + `.manifest.json`
- npz arrays: `ts_us` (int64, sorted), `seq` (int64), `price` (f64), `qty` (f64), `buyer_maker` (bool)
- manifest: source basename, sha256 of source zip, content checksum, row count, first/last ts, receive ts, created_at
- ingestion idempotent; changed source detected via checksum (re-ingest with `--force`)
- queries: `range_query(start_us, end_us)` via `searchsorted` — **[start, end)** semantics; reads never touch the source zip
- order books are NOT re-cached: recorder JSONL already satisfies sequential/sorted/partitioned requirements

## 9. Feature Interface (interface only)

```
MarketState ──► FeatureCalculator (Protocol: name, calculate) ──► FeatureSet (dict[str, float])
```

- calculators are stateless pure functions; no cross-event memory in Phase 1
- example calculators (`MidPrice`, `SpreadBps`, `BookImbalance`) exist only to validate the interface
