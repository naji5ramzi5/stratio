"""
verify_tracker.py - Honest horizon-matched backfill of the prediction tracker.

The live bot records thousands of predictions but the running verification loop
requires a live price feed at the right moment; records older than a 48h window
were skipped forever, so the tracker had 8000+ predictions and ZERO verified
samples and therefore no real calibration.

This tool fixes that by re-deriving the actual outcome from Binance HISTORY:
for every unverified record it fetches the close of the bar that STARTS at
``created_at + timeframe`` (the exact forecast horizon) and computes the true
forward return. It then rebuilds the honest calibration table.

Run: python verify_tracker.py [--max-verify 1000] [--tf 24]
"""
import argparse
import bisect
import json
import os
import sys
import time
import warnings
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader.binance_ohlcv import get_session
from settings import BINANCE_BASE

TF_INTERVAL = {2: "5m", 6: "15m", 8: "30m", 24: "1h"}
HORIZON_MS = {2: 2 * 3600_000, 6: 6 * 3600_000, 8: 8 * 3600_000, 24: 24 * 3600_000}
INTERVAL_MS = {"5m": 5 * 60_000, "15m": 15 * 60_000, "30m": 30 * 60_000, "1h": 3600_000}


def _to_ms(iso):
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return None


def fetch_range(symbol, interval, start_ms, end_ms):
    """Fetch all klines in [start_ms, end_ms] using paginated startTime."""
    out = []
    cur = start_ms
    while cur <= end_ms:
        resp = get_session().get(BINANCE_BASE + "/api/v3/klines",
                                 params={"symbol": symbol, "interval": interval,
                                         "startTime": cur, "endTime": end_ms,
                                         "limit": 1000}, timeout=10)
        if resp.status_code != 200:
            return None
        batch = resp.json()
        if not isinstance(batch, list) or not batch:
            break
        out.extend(batch)
        if len(batch) < 1000:
            break
        cur = int(batch[-1][0]) + INTERVAL_MS[interval]
    return out


def close_at_or_after(rows, target_ms):
    """First close whose bar timestamp >= target_ms (bisect on open time)."""
    ts = [r[0] for r in rows]
    i = bisect.bisect_left(ts, target_ms)
    if i >= len(rows):
        return None, None
    return rows[i][4], ts[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="prediction_accuracy.json")
    ap.add_argument("--max-verify", type=int, default=100000)
    ap.add_argument("--tfs", default="2,6,8,24")
    args = ap.parse_args()

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.file)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    preds = data.get("predictions", [])
    tfs = {int(x) for x in args.tfs.split(",") if x.strip()}

    unverified = [p for p in preds if not p.get("verified")
                  and p.get("timeframe_hours") in tfs]
    print(f"tracker: {len(preds)} records, {len([p for p in preds if p.get('verified')])} verified, "
          f"{len(unverified)} to backfill")

    by_sym = {}
    for p in unverified:
        by_sym.setdefault(p["symbol"], []).append(p)

    done = 0
    skipped = 0
    for sym, recs in sorted(by_sym.items()):
        tf = recs[0]["timeframe_hours"]
        interval = TF_INTERVAL[tf]
        horizon = HORIZON_MS[tf]
        times = sorted(_to_ms(r["timestamp"]) for r in recs)
        if not times:
            continue
        lo = times[0] - INTERVAL_MS[interval]
        hi = times[-1] + horizon + INTERVAL_MS[interval]
        rows = fetch_range(sym, interval, lo, hi)
        if rows is None or not rows:
            print(f"  {sym}: klines unavailable, {len(recs)} skipped")
            skipped += len(recs)
            continue
        hits = misses = 0
        for p in recs:
            t0 = _to_ms(p["timestamp"])
            entry = p.get("current_price")
            if t0 is None or not entry:
                skipped += 1
                continue
            close, bar_ts = close_at_or_after(rows, t0 + horizon)
            if close is None or bar_ts - (t0 + horizon) > 2 * INTERVAL_MS[interval]:
                misses += 1
                continue
            p["actual_change_pct"] = round((float(close) / float(entry) - 1.0) * 100, 4)
            p["verified"] = True
            p["_backfilled"] = True
            p["_bar_open_utc"] = datetime.fromtimestamp(bar_ts / 1000, tz=timezone.utc).isoformat()
            hits += 1
            done += 1
        if hits or misses:
            print(f"  {sym} ({interval}): verified {hits}, horizon-unresolved {misses}")
        if done >= args.max_verify:
            break
        time.sleep(0.15)

    if done:
        from prediction_tracker import PredictionTracker
        tracker = PredictionTracker()
        tracker.data = data
        tracker._update_stats()
        tracker._save()

    verified = [p for p in data.get("predictions", []) if p.get("verified")]
    if verified:
        cal = {}
        for k, v in (data.get("stats", {}).get("calibration") or {}).items():
            cal[k] = v["realized_acc"]
        cal["overall"] = round(data["stats"]["direction_accuracy_pct"] / 100.0, 3)
        cal["n"] = len(verified)
        cal["source"] = "verify_tracker horizon-matched backfill"
        cal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "models", "calibration.json")
        with open(cal_path, "w", encoding="utf-8") as f:
            json.dump(cal, f, indent=2)
        print(f"wrote {cal_path}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "prediction_accuracy_verified.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"verified dump -> {out}")

    print(f"\nbackfill complete: {done} verified this run, {skipped} skipped (missing data)")
    s = data.get("stats", {})
    print(f"total_verified={s.get('total_verified')}  "
          f"direction_accuracy_pct={s.get('direction_accuracy_pct')}  "
          f"calibration={s.get('calibration')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
