"""Trade-feed replay: build best-bid/ask proxies from real aggTrades,
vectorized and strictly point-in-time (only trades at ts <= bucket are used).

Model:
- buyer_maker=True  -> aggressive SELL hits the bid  -> trade price ~ bid
- buyer_maker=False -> aggressive BUY lifts the ask  -> trade price ~ ask
- Last known bid/ask per time bucket, forward-filled.
- Trades occur AT the touch -> execution at our proxy == real taker execution price.
- Depth is unknown from tape alone -> execution qty is constrained by
  conservative assumptions (stress-tested in execution layer).
"""
import numpy as np


def _bucketize(ts_us: np.ndarray, bucket_us: int) -> np.ndarray:
    return ts_us // bucket_us


def _last_quote_per_bucket(bucket: np.ndarray, price: np.ndarray, uniq: np.ndarray) -> np.ndarray:
    """For each bucket in uniq (sorted), the price of the LAST trade in that bucket.
    NaN where no trade in bucket (caller forward-fills)."""
    last_pos = np.searchsorted(bucket, uniq + 1, side="left") - 1
    out = np.full(len(uniq), np.nan)
    valid = last_pos >= 0
    out[valid] = price[last_pos[valid]]
    return out


def _ffill(a: np.ndarray) -> np.ndarray:
    mask = np.isfinite(a)
    idx = np.where(mask, np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    return a[idx]


def build_quote_series(trades: dict, bucket_us: int = 100_000) -> dict:
    """Return dict: bucket (us), bid, ask (last known per bucket, ffill)."""
    ts = trades["ts_us"]
    price = trades["price"]
    bm = trades["buyer_maker"]

    uniq = np.unique(ts // bucket_us)

    bid_idx = np.flatnonzero(bm)
    ask_idx = np.flatnonzero(~bm)

    bid = np.full(len(uniq), np.nan)
    if len(bid_idx):
        bid = _last_quote_per_bucket(ts[bid_idx] // bucket_us, price[bid_idx], uniq)
    ask = np.full(len(uniq), np.nan)
    if len(ask_idx):
        ask = _last_quote_per_bucket(ts[ask_idx] // bucket_us, price[ask_idx], uniq)

    bid = _ffill(bid)
    ask = _ffill(ask)
    return {"bucket_us": uniq, "bid": bid, "ask": ask}


def reindex_quotes(q: dict, target_buckets: np.ndarray) -> tuple:
    """Reindex per-symbol quote series onto target bucket grid (last-known, PIT-safe)."""
    pos = np.searchsorted(q["bucket_us"], target_buckets, side="right") - 1
    bid = np.full(len(target_buckets), np.nan)
    ask = np.full(len(target_buckets), np.nan)
    valid = pos >= 0
    if valid.any():
        bid[valid] = q["bid"][pos[valid]]
        ask[valid] = q["ask"][pos[valid]]
    return bid, ask


def merge_cycle_quotes(quotes: dict, cycle: list, bucket_us: int = 100_000) -> dict:
    """Merge quote series for a cycle's legs onto a single bucket grid.
    quotes: {symbol: {bucket_us, bid, ask}}; cycle: [sym1, sym2, sym3]"""
    all_buckets = np.unique(np.concatenate([q["bucket_us"] for q in quotes.values()]))
    out = {"bucket_us": all_buckets}
    for sym in cycle:
        bid, ask = reindex_quotes(quotes[sym], all_buckets)
        out[f"{sym}_bid"] = bid
        out[f"{sym}_ask"] = ask
    return out
