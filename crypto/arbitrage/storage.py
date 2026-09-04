"""Load Binance Vision aggTrades zips into numpy arrays.
Spot format (no header): agg_id, price, qty, first_id, last_id, ts_us, buyer_maker, best_match
Futures format (header):  agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time_ms,is_buyer_maker
Auto-detects header + timestamp unit (µs vs ms) and normalizes to µs int64.
"""
import os
import zipfile
import io
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _read_zip(path):
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            raw = f.read()
    return io.BytesIO(raw), name


def load_aggtrades(symbol: str, day: str, market: str = "spot") -> dict:
    """Load one daily aggTrades file. Returns dict:
    ts_us (int64 µs), price (float64), qty (float64), buyer_maker (bool)."""
    local = os.path.join(DATA_DIR, market, "aggTrades", symbol, f"{day}.zip")
    if not os.path.exists(local):
        raise FileNotFoundError(local)
    buf, name = _read_zip(local)
    head = buf.readline().decode("utf-8", "replace")
    buf.seek(0)
    if head.lstrip().startswith("agg_trade_id") or head.lstrip().startswith("agg_id"):
        df = pd.read_csv(buf, usecols=["price", "quantity" if "quantity" in head else "qty",
                                       "transact_time", "is_buyer_maker"])
        df = df.rename(columns={"quantity": "qty", "transact_time": "ts", "is_buyer_maker": "bm"})
        ms = True
    else:
        df = pd.read_csv(buf, header=None, usecols=[1, 2, 5, 6],
                         names=["price", "qty", "ts", "bm"])
        ms = False
    df = df.sort_values("ts")
    ts = df["ts"].to_numpy(dtype=np.int64)
    if ms:
        ts = ts * 1000  # ms -> µs
    return {
        "ts_us": ts,
        "price": df["price"].to_numpy(dtype=np.float64),
        "qty": df["qty"].to_numpy(dtype=np.float64),
        "buyer_maker": df["bm"].astype(str).str.lower().isin(["true", "1"]).to_numpy(dtype=bool),
    }


def list_days(market: str, symbol: str) -> list:
    d = os.path.join(DATA_DIR, market, "aggTrades", symbol)
    if not os.path.exists(d):
        return []
    return sorted(f.replace(".zip", "") for f in os.listdir(d) if f.endswith(".zip"))


def available_days(market: str) -> dict:
    out = {}
    d = os.path.join(DATA_DIR, market, "aggTrades")
    if not os.path.exists(d):
        return out
    for sym in os.listdir(d):
        out[sym] = list_days(market, sym)
    return out
