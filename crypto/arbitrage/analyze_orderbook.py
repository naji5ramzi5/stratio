"""Analyze recorded order-book data: touch depth, spread, and the depth constraint
on passive fills. Answers: 'if I quote at the touch, how often would a market
order of size X have consumed my quantity?' (queue-depth proxy from L2 tape).
Run: python analyze_orderbook.py [--date YYYY-MM-DD]
"""
import argparse
import glob
import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "data", "orderbook")


def load_day(day: str, symbol: str):
    depth, trades = [], []
    dp = os.path.join(OUT, day, f"{symbol}_depth.jsonl")
    tp = os.path.join(OUT, day, f"{symbol}_trades.jsonl")
    if os.path.exists(dp):
        for line in open(dp):
            r = json.loads(line)
            bids = {float(p): float(q) for p, q in (r.get("b") or [])}
            asks = {float(p): float(q) for p, q in (r.get("a") or [])}
            depth.append((r["t"], bids, asks))
    if os.path.exists(tp):
        for line in open(tp):
            r = json.loads(line)
            trades.append((r["t"], float(r["p"]), float(r["q"]), r.get("m")))
    depth.sort(key=lambda x: x[0])
    trades.sort(key=lambda x: x[0])
    return depth, trades


def analyze(day: str, symbol: str):
    depth, trades = load_day(day, symbol)
    if not depth:
        return None
    best_bid = np.array([max(b.keys()) if b else np.nan for _, b, _ in depth])
    best_ask = np.array([min(a.keys()) if a else np.nan for _, _, a in depth])
    bb_qty = np.array([b[max(b.keys())] if b else 0 for _, b, _ in depth])
    ba_qty = np.array([a[min(a.keys())] if a else 0 for _, _, a in depth])
    spread_bps = (best_ask - best_bid) / best_bid * 1e4

    # depth constraint: fraction of trades with qty <= touch qty on their side
    touch = {"b": 0, "a": 0, "ok": 0}
    for t, p, q, m in trades:
        ts = depth
        idx = np.searchsorted([d[0] for d in depth], t, side="right") - 1
        if idx < 0:
            continue
        _, bids, asks = depth[idx]
        if m:  # buyer maker = seller aggressive -> hits bid
            qty = bids.get(p)
            touch["b"] += 1
        else:
            qty = asks.get(p)
            touch["a"] += 1
        if qty is not None and q <= qty:
            touch["ok"] += 1

    return {
        "events": len(depth),
        "trades": len(trades),
        "best_bid": round(float(np.nanmean(best_bid)), 2),
        "best_ask": round(float(np.nanmean(best_ask)), 2),
        "spread_bps_mean": round(float(np.nanmean(spread_bps)), 4),
        "spread_bps_p50": round(float(np.nanmedian(spread_bps)), 4),
        "touch_bid_qty_mean": round(float(np.nanmean(bb_qty)), 3),
        "touch_ask_qty_mean": round(float(np.nanmean(ba_qty)), 3),
        "touch_bid_qty_btc": round(float(np.nanmean(bb_qty)), 4),
        "touch_ask_qty_btc": round(float(np.nanmean(ba_qty)), 4),
        "depth_constraint_ok_pct": round(100 * touch["ok"] / touch["b"] if touch["b"] else 0, 2),
        "trade_side_counts": touch,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None)
    args = ap.parse_args()
    days = [args.date] if args.date else sorted(os.listdir(OUT))
    for day in days:
        day_dir = os.path.join(OUT, day)
        if not os.path.isdir(day_dir):
            continue
        print(f"=== {day} ===")
        for f in sorted(glob.glob(os.path.join(day_dir, "*_depth.jsonl"))):
            sym = os.path.basename(f).replace("_depth.jsonl", "")
            r = analyze(day, sym)
            if r:
                print(f"  {sym}: events={r['events']} trades={r['trades']} "
                      f"spread={r['spread_bps_mean']}bps touch_bid={r['touch_bid_qty_btc']} "
                      f"depth_ok={r['depth_constraint_ok_pct']}%")


if __name__ == "__main__":
    main()
