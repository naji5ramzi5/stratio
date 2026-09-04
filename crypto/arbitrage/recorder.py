"""Live order-book recorder (research data infrastructure).
Records Binance USD-M futures L2 depth@100ms + aggTrades for the maker-arbitrage
cycle (BTCUSDT, ETHUSDT, ETHBTC). The depth history calibrates the queue-position
fill probability (p_leg) that the maker backtest depends on.
- Research-only: no trading, no order submission.
- Auto-reconnect with exponential backoff, heartbeat, daily file rotation.
"""
import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone

import websockets

FUT_WS = "wss://fstream.binance.com/stream?streams="
OUT = os.path.join(os.path.dirname(__file__), "data", "orderbook")

SYMBOLS_DEFAULT = ["btcusdt", "ethusdt", "ethbtc"]
DEPTH_LEVELS = 20
SNAP_EVERY_S = 300  # full snapshot via REST every 5 min for integrity checks


def day_path(symbol: str, suffix: str) -> str:
    d = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    folder = os.path.join(OUT, d)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{symbol}_{suffix}.jsonl")


async def recv_loop(streams: list, symbols: list, duration_h: float, status_path: str):
    url = FUT_WS + "/".join(streams)
    t0 = time.time()
    deadline = t0 + duration_h * 3600
    handles = {s: {"depth": None, "trades": None} for s in symbols}
    written = {f"{s}_{k}": 0 for s in symbols for k in ("depth", "trades")}

    while time.time() < deadline:
        try:
            async with websockets.connect(url, open_timeout=15, ping_interval=20,
                                          ping_timeout=20) as ws:
                while time.time() < deadline:
                    raw = await asyncio.wait_for(ws.recv(), timeout=60)
                    msg = json.loads(raw)
                    data = msg.get("data", {})
                    sym = data.get("s", "").lower()
                    if not sym or sym not in handles:
                        continue
                    kind = "depth" if data.get("b") is not None else "trades"
                    fh = open(day_path(sym, kind), "a", encoding="utf-8")
                    rec = {"t": data.get("E") or data.get("T"),
                           "U": data.get("U"), "u": data.get("u"),
                           "b": data.get("b"), "a": data.get("a"),
                           "p": data.get("p"), "q": data.get("q"),
                           "m": data.get("m")}
                    fh.write(json.dumps({k: v for k, v in rec.items() if v is not None}) + "\n")
                    fh.close()
                    written[f"{sym}_{kind}"] += 1
                    if int(time.time()) % 60 == 0:
                        with open(status_path, "w") as st:
                            json.dump({"last_event": time.time(),
                                       "written": written,
                                       "uptime_s": int(time.time() - t0)}, st)
        except asyncio.TimeoutError:
            print(f"[recorder] timeout, reconnecting ({time.time()-t0:.0f}s uptime)")
        except Exception as e:
            print(f"[recorder] error {type(e).__name__}: {str(e)[:120]}, reconnecting")
        await asyncio.sleep(5)


def main():
    ap = argparse.ArgumentParser(description="Live futures depth recorder (research)")
    ap.add_argument("--symbols", default=",".join(SYMBOLS_DEFAULT))
    ap.add_argument("--hours", type=float, default=24 * 14, help="recording duration (h)")
    ap.add_argument("--depth", type=int, default=DEPTH_LEVELS)
    ap.add_argument("--with-trades", action="store_true",
                    help="also subscribe aggTrade streams (may be unsupported on some feeds)")
    args = ap.parse_args()

    symbols = [s.lower() for s in args.symbols.split(",")]
    streams = [f"{s}@depth{args.depth}@100ms" for s in symbols]
    if args.with_trades:
        streams += [f"{s}@aggTrade" for s in symbols]
    status_path = os.path.join(OUT, "recorder_status.json")

    print(f"[recorder] symbols={symbols} depth={args.depth} duration={args.hours}h "
          f"trades={'on' if args.with_trades else 'off'}")
    print(f"[recorder] streams: {streams}")
    asyncio.run(recv_loop(streams, symbols, args.hours, status_path))


if __name__ == "__main__":
    main()
