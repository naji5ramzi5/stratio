"""V4D/V4E reproduction through the Phase 1 data path (Phase 2, STEP 14).

Objective: determine whether the V4D/V4E candidate's inputs can be
reproduced from the Phase 1 event store, and whether the old vs new data
handling produces identical results.

Method (no optimization, no parameter changes):
1. Load the SAME daily futures data via the Phase 1 store (npz caches,
   sorted arrays, µs) instead of the old pandas zip re-parse.
2. Feed the EXACT V4D/V4E simulator (`arbitrage/maker_seq.simulate`,
   `arbitrage/maker_devoos.simulate` — imported read-only) with the
   identical pre-registered parameter grid and the identical rng stream
   (seed 11, same call order) as the original runs.
3. Compare every number against the saved results
   (ARBITRAGE_V4D_MAKER_SEQ.json / ARBITRAGE_V4E_MAKER_DEVOOS.json).

Deterministic equality of p=1.0 runs proves input equivalence; equality of
the full table (same rng stream) proves data-path equivalence.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, timedelta

import numpy as np

from ..store import EventStore

CRYPTO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, CRYPTO)

from arbitrage import maker_devoos, maker_seq  # noqa: E402  (read-only reuse)

SYMBOLS = ["BTCUSDT", "ETHUSDT", "ETHBTC"]
MARKET = "futures_um"


def load_series_from_store(store: EventStore, day_list) -> dict:
    """{symbol: [day_dict, ...]} with the SAME dict schema as the old
    loader (load_aggtrades): ts_us (µs int64, sorted), price, qty,
    buyer_maker."""
    series = {s: [] for s in SYMBOLS}
    for day in day_list:
        for s in SYMBOLS:
            if not store.has_day(MARKET, s, day):
                raise FileNotFoundError(f"store missing {MARKET}/{s}/{day}")
            data = store.load_day(MARKET, s, day)
            series[s].append({
                "ts_us": data["ts_us"],
                "price": data["price"],
                "qty": data["qty"],
                "buyer_maker": data["buyer_maker"],
            })
    return series


def replicate_v4e(store: EventStore) -> dict:
    """Exact re-run of maker_devoos.main() with the store as data source."""
    dev = [(date(2026, 7, 25) + timedelta(days=i)).isoformat() for i in range(10)]
    oos = [(date(2026, 8, 4) + timedelta(days=i)).isoformat() for i in range(6)]
    s_dev, s_oos = load_series_from_store(store, dev), load_series_from_store(store, oos)
    rng = np.random.default_rng(11)
    out = {"protocol": "replica via Phase 1 store", "results": {}}
    for thr in (5.0, 8.0, 10.0):
        for win in (5, 10):
            for p in (0.5, 0.2):
                key = f"thr{thr}_win{win*100}ms_p{p}"
                d = maker_devoos.simulate(s_dev, thr, win, p, rng)
                o = maker_devoos.simulate(s_oos, thr, win, p, rng)
                out["results"][key] = {"dev": d, "oos": o}
    return out


def replicate_v4d(store: EventStore) -> dict:
    """Exact re-run of maker_seq.main() (48-config grid, rng 11)."""
    series = load_series_from_store(store, maker_seq.DAYS)
    rng = np.random.default_rng(11)
    out = {"fee_net_bps": maker_seq.MAKER_FEE_LEG_BPS * 3,
           "notional_usd": maker_seq.NOTIONAL, "results": {}}
    for thr in (5.0, 8.0, 10.0):
        for win in (5, 10):
            for p in (1.0, 0.5, 0.2, 0.1):
                for lat in (0, 2):
                    key = f"thr{thr}_win{win*100}ms_p{p}_lat{lat*100}ms"
                    out["results"][key] = maker_seq.simulate(
                        thr, win, p, rng, series, lat)
    return out


def compare_tables(replica: dict, original: dict, key_chain: list) -> dict:
    """Compare replica vs original JSON, return per-key diffs."""
    mismatches = []
    checked = 0
    for key, val in original["results"].items():
        if key not in replica["results"]:
            mismatches.append({"key": key, "issue": "missing in replica"})
            continue
        r, o = replica["results"][key], val
        for field in key_chain:
            checked += 1
            if r.get(field) != o.get(field):
                mismatches.append({
                    "key": key, "field": field,
                    "replica": r.get(field), "original": o.get(field),
                })
    return {"checked": checked, "mismatches": mismatches,
            "identical": len(mismatches) == 0}
