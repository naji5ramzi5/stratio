"""CLI:  python -m quant_x BTCUSDT [--interval 1h] [--lookback 200] [--no-ml] [--json out.json]
"""
import argparse
import json
import sys
from datetime import datetime, timezone

from quant_x.report import build_report, render_markdown

VALID_INTERVALS = ("5m", "15m", "30m", "1h")


def main():
    ap = argparse.ArgumentParser(prog="quant_x", description="QUANT-X research engine")
    ap.add_argument("symbol", nargs="?", default="BTCUSDT")
    ap.add_argument("--interval", default="1h", choices=VALID_INTERVALS)
    ap.add_argument("--lookback", type=int, default=200)
    ap.add_argument("--no-ml", action="store_true", help="skip ML ensemble input")
    ap.add_argument("--json", default=None, help="also write structured report to this path")
    args = ap.parse_args()

    report = build_report(args.symbol, args.interval, args.lookback,
                          use_ml=not args.no_ml,
                          now_hour_utc=datetime.now(timezone.utc).hour)

    print(render_markdown(report))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, default=str, indent=2)
        print(f"\n[quant_x] structured report -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
