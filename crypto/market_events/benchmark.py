"""Benchmark: event store vs the status-quo pandas re-parse path.

Compares, on the SAME day of real data:
1. BEFORE  — arbitrage-layer path: pandas re-parse of the Vision zip
   (this is what the existing research layer does per query).
2. AFTER   — derived npz load, range query, and sequential replay.

No assumptions: measured on the machine, on real files, printed as tables.
"""

from __future__ import annotations

import gc
import os
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MARKET = "spot"
SYMBOL = "BTCUSDT"
DAY = "2026-08-09"


def bench_zip_parse(zip_path: str) -> dict:
    import pandas as pd
    import zipfile
    import io

    t0 = time.perf_counter()
    tracemalloc.start()
    with zipfile.ZipFile(zip_path) as z:
        raw = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None, usecols=[1, 2, 5, 6],
                     names=["price", "qty", "ts", "bm"])
    df = df.sort_values("ts")
    ts = df["ts"].to_numpy(dtype=np.int64)
    price = df["price"].to_numpy(dtype=np.float64)
    qty = df["qty"].to_numpy(dtype=np.float64)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "label": "BEFORE: zip -> pandas -> sort",
        "seconds": time.perf_counter() - t0,
        "peak_mb": peak / 1e6,
        "rows": len(ts),
        "range_us": (time.perf_counter() - t0) * 1e6,
    }


def bench_store(store) -> dict:
    t0 = time.perf_counter()
    data = store.load_day(MARKET, SYMBOL, DAY)
    load_sec = time.perf_counter() - t0
    ts = data["ts_us"]
    n = len(ts)

    start_us, end_us = int(ts[0]), int(ts[-1])
    lo = int(np.searchsorted(ts, start_us, side="left"))
    hi = int(np.searchsorted(ts, end_us, side="right"))

    t1 = time.perf_counter()
    sliced = store.range_query(MARKET, SYMBOL, DAY, start_us, end_us)
    range_sec = time.perf_counter() - t1

    t2 = time.perf_counter()
    engine = store.replay(MARKET, SYMBOL, DAY)
    data0 = store.load_day(MARKET, SYMBOL, DAY)
    first_ts = int(data0["ts_us"][0])
    engine.advance_to(first_ts)
    count = 0
    clock = 0
    while True:
        ev = engine.advance_by(60_000_000)  # 1-minute bars
        if ev is None:
            break
        count = engine.event_sequence
        clock = engine.current_event_time
    replay_sec = time.perf_counter() - t2

    # pure sequential replay (no per-bar overhead)
    t3 = time.perf_counter()
    eng2 = store.replay(MARKET, SYMBOL, DAY)
    eng2.advance_to(int(data0["ts_us"][-1]))
    seq_sec = time.perf_counter() - t3
    seq_evs = n / seq_sec if seq_sec else float("inf")

    del data0, data, sliced
    gc.collect()
    return {
        "label": "AFTER: npz load + range + replay",
        "seconds": load_sec + range_sec + replay_sec,
        "rows": n,
        "load_ms": load_sec * 1e3,
        "range_us": range_sec * 1e6,
        "bar_walk_evs": n / replay_sec if replay_sec else float("inf"),
        "seq_replay_evs": seq_evs,
        "clock_end": clock,
    }


def main() -> int:
    from market_events.store import DEFAULT_ROOT, EventStore

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "arbitrage", "data",
    )
    zip_path = os.path.join(data_dir, MARKET, "aggTrades", SYMBOL, f"{DAY}.zip")
    store = EventStore(DEFAULT_ROOT)

    if not store.has_day(MARKET, SYMBOL, DAY):
        print(f"ingesting {DAY} first...")
        store.ingest_trade_day(zip_path, MARKET, SYMBOL, DAY,
                               receive_timestamp=int(os.path.getmtime(zip_path)) * 1_000_000)

    if not os.path.exists(zip_path):
        print(f"no data for benchmark: {zip_path}", file=sys.stderr)
        return 2

    print(f"benchmark data: {zip_path}")
    a = bench_zip_parse(zip_path)
    b = bench_store(store)

    print(f"\n{'case':<42}{'rows':>12}{'wall(s)':>10}{'range(µs)':>12}")
    print("-" * 78)
    for r in (a, b):
        print(f"{r['label']:<42}{r['rows']:>12,}{r['seconds']:>10.3f}{r['range_us']:>12,.0f}")
    print(f"\nAFTER components: load {b['load_ms']:.1f}ms | range {b['range_us']:,.0f}µs")
    print(f"AFTER sequential replay (full day, no bar loop): {b['seq_replay_evs']:,.0f} events/s")
    print(f"AFTER 1-min bar walk (PIT-constrained): {b['bar_walk_evs']:,.0f} events/s")
    print(f"BEFORE path peak memory: {a['peak_mb']:,.1f} MB")
    print(f"\nper-query speedup (range vs full re-parse): {a['range_us'] / b['range_us']:.1f}x")
    print(f"total wall (BEFORE vs load+range+replay): {a['seconds'] / b['seconds']:.1f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
