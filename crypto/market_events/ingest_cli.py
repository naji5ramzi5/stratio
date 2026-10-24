"""CLI: ingest Vision aggTrades days into the derived event store.

Usage:
    python -m market_events.ingest_cli --market spot --symbol BTCUSDT \
        --days 2026-08-01..2026-08-09 [--receive-ts <us>] [--force]

The --receive-ts is the honest "when we received this data" timestamp
(download time); the store will never invent one. Default: unix time of
the source zip file mtime (a truthful proxy for ingestion time).

Writes nothing except the derived cache + manifests.
"""

from __future__ import annotations

import argparse
import os
import sys
import time


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market", required=True, choices=["spot", "futures_um"])
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--days", required=True, help="YYYY-MM-DD or YYYY-MM-DD..YYYY-MM-DD")
    ap.add_argument("--receive-ts", type=int, default=None, help="µs; default = zip mtime")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if ".." in args.days:
        start, end = args.days.split("..")
        from datetime import date, timedelta

        d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
        if d1 < d0:
            print("day range inverted", file=sys.stderr)
            return 2
        days = [(d0 + timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]
    else:
        days = [args.days]

    from .store import DEFAULT_ROOT, EventStore

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "arbitrage", "data"
    )
    store = EventStore(DEFAULT_ROOT)
    t0 = time.time()
    total_rows = 0
    for day in days:
        zip_path = os.path.join(
            data_dir, args.market, "aggTrades", args.symbol, f"{day}.zip"
        )
        if not os.path.exists(zip_path):
            print(f"SKIP {day}: {zip_path} missing")
            continue
        receive_ts = args.receive_ts
        if receive_ts is None:
            receive_ts = int(os.path.getmtime(zip_path)) * 1_000_000
        manifest = store.ingest_trade_day(
            zip_path,
            market=args.market,
            symbol=args.symbol,
            day=day,
            receive_timestamp=receive_ts,
            force=args.force,
        )
        total_rows += manifest["rows"]
        print(
            f"OK {day}: rows={manifest['rows']:,} "
            f"ts={manifest['first_ts_us']}..{manifest['last_ts_us']} "
            f"checksum={manifest['content_checksum'][:10]}"
        )
    print(
        f"DONE {len(days)} days, {total_rows:,} rows in {time.time() - t0:.1f}s "
        f"-> {DEFAULT_ROOT}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
