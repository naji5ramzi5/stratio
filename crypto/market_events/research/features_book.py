"""Order-book feature pilot (Phase 2).

Data reality check (from PHASE2_RESEARCH_AUDIT): the ONLY order-book
dataset in this environment is the 2026-08-10 recorder session
(~28 minutes, top-20 @10 Hz, BTCUSDT + ETHUSDT futures). Features are
therefore computed at the depth-event grid on that single day.

Every feature is read from the Phase 1 OrderBook state engine
(market_events/orderbook.py), which is PIT by construction (each event
updates state using only itself + prior state). Labels are future mid
returns over the SAME depth series (label engine may look ahead).

This is a pilot: with ~28 min of data no statistical claim is possible;
the outputs feed the report as DATA_LIMITED evidence.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..adapters import recorder_depth_row_to_event
from ..model import EventType
from ..orderbook import OrderBook, OrderBookError

BOOK_FEATURES = ["best_bid", "best_ask", "mid", "spread_bps", "bid_depth_quote",
                 "ask_depth_quote", "depth_imbalance", "depth_imbalance_top3",
                 "microprice", "microprice_offset_bps", "bid_depth_5qty",
                 "ask_depth_5qty"]

DEPTH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "arbitrage", "data", "orderbook", "2026-08-10",
)

HORIZONS_US = [5_000_000, 15_000_000, 30_000_000]  # 5/15/30 s (pilot only)


def load_depth_events(symbol: str, dir_path: str = DEPTH_DIR):
    path = os.path.join(dir_path, f"{symbol.lower()}_depth.jsonl")
    if not os.path.exists(path):
        return []
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(recorder_depth_row_to_event(
                row, "binance_futures_um", symbol, receive_timestamp=None))
    return events


def book_features_at(book: OrderBook) -> Dict[str, float]:
    """Feature vector from the current book state."""
    bb, ba, mid = book.best_bid, book.best_ask, book.mid
    if bb is None or ba is None or mid is None or mid <= 0:
        return {k: float("nan") for k in BOOK_FEATURES}
    bid5, ask5 = book.top_n("bid", 5), book.top_n("ask", 5)
    bdp = sum(p * q for p, q in bid5)
    adp = sum(p * q for p, q in ask5)
    wsum = sum(p * q for p, q in bid5 + ask5)
    qsum = sum(q for _, q in bid5 + ask5)
    micro = wsum / qsum if qsum > 0 else mid
    bid3, ask3 = book.top_n("bid", 3), book.top_n("ask", 3)
    bd3 = sum(p * q for p, q in bid3)
    ad3 = sum(p * q for p, q in ask3)
    spread_bps = (ba - bb) / mid * 1e4
    imb = (bdp - adp) / (bdp + adp) if (bdp + adp) > 0 else 0.0
    imb3 = (bd3 - ad3) / (bd3 + ad3) if (bd3 + ad3) > 0 else 0.0
    return {
        "best_bid": bb,
        "best_ask": ba,
        "mid": mid,
        "spread_bps": spread_bps,
        "bid_depth_quote": bdp,
        "ask_depth_quote": adp,
        "depth_imbalance": imb,
        "depth_imbalance_top3": imb3,
        "microprice": micro,
        "microprice_offset_bps": (micro - mid) / mid * 1e4,
        "bid_depth_5qty": sum(q for _, q in bid5),
        "ask_depth_5qty": sum(q for _, q in ask5),
    }


def build_book_dataset(symbol: str, dir_path: str = DEPTH_DIR) -> pd.DataFrame:
    """Feature rows at the depth-event grid + future mid-return labels."""
    events = load_depth_events(symbol, dir_path)
    if not events:
        return pd.DataFrame()
    book = OrderBook("binance_futures_um", symbol)
    rows: List[dict] = []
    ts = []
    mids = []
    for ev in events:
        if ev.event_type is not EventType.ORDER_BOOK_SNAPSHOT:
            continue
        try:
            book.apply(ev)
        except OrderBookError:
            continue
        ts.append(ev.event_timestamp)
        mids.append(book.mid)
        row = {"ts_us": ev.event_timestamp}
        row.update(book_features_at(book))
        rows.append(row)
    df = pd.DataFrame(rows)
    t = np.array(ts, dtype=np.int64)
    m = np.array(mids, dtype=np.float64)
    for h in HORIZONS_US:
        end_idx = np.searchsorted(t, t + h, side="right")
        lp0, lp1 = np.log(m), np.full(len(m), np.nan)
        valid = end_idx < len(m)
        lp1[valid] = np.log(m[end_idx[valid]])
        gross = lp1 - lp0
        label_valid = valid & np.isfinite(gross) & (end_idx > np.arange(len(m)))
        df[f"label_gross_{int(h/1_000_000)}s"] = gross
        df[f"label_valid_{int(h/1_000_000)}s"] = label_valid
    df["symbol"] = symbol
    df["day"] = "2026-08-10"
    return df
