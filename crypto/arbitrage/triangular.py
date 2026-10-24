"""Triangular arbitrage cycle detector (vectorized, point-in-time safe).

Cycle USDT -> A -> B -> USDT (via A/USDT, B/A, B/USDT):
    r1 = bid_BUSDT / (ask_AUSDT * ask_BA)
Cycle USDT -> B -> A -> USDT:
    r2 = bid_AUSDT * bid_BA / ask_BUSDT
where bid_BA / ask_BA are on the B/A pair (in units of A).

All quotes are last-known-before-bucket real trade executions -> strictly PIT.
"""
import numpy as np
from .replay import build_quote_series, merge_cycle_quotes


def cycle_rates(merged: dict, a: str, b: str) -> tuple:
    """Return (r1, r2) arrays for the two cycle directions.
    a = quote-asset on the cross pair (e.g. BTC in ETH/BTC), b = USDT-paired asset (e.g. ETH)."""
    bid_a_usdt = merged[f"{a}USDT_bid"]
    ask_a_usdt = merged[f"{a}USDT_ask"]
    bid_b_usdt = merged[f"{b}USDT_bid"]
    ask_b_usdt = merged[f"{b}USDT_ask"]
    bid_ba = merged[f"{b}{a}_bid"]
    ask_ba = merged[f"{b}{a}_ask"]
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = bid_b_usdt / (ask_a_usdt * ask_ba)     # USDT->A->B->USDT
        r2 = bid_a_usdt * bid_ba / ask_b_usdt       # USDT->B->A->USDT
    return r1, r2


def _qualifying_runs(net: np.ndarray, latency_buckets: int):
    """Vectorized run detection: buckets where net>0 for >= latency_buckets consecutive buckets.
    Returns entry bucket indices of qualifying runs."""
    above = net > 0
    d = np.diff(np.concatenate(([0], above.astype(np.int8), [0])))
    start = np.flatnonzero(d == 1)
    end = np.flatnonzero(d == -1)
    lengths = end - start
    return start[lengths >= latency_buckets]


def simulate_latency_fills(net: np.ndarray, latency_buckets: int,
                           rng: np.random.Generator, fill_prob: float = 1.0) -> dict:
    """Given per-bucket net edge per unit (rate-1-fee), require persistence >= latency,
    then fill at t+latency if still profitable (else cancelled)."""
    n = len(net)
    entry = _qualifying_runs(net, latency_buckets)
    fill_idx = entry + latency_buckets
    ok = fill_idx < n
    entry, fill_idx = entry[ok], fill_idx[ok]

    trades = []
    for ei, fi in zip(entry, fill_idx):
        fill_net = net[fi]
        if fill_net <= 0:
            continue
        if rng.random() > fill_prob:
            continue
        trades.append({"entry_bucket": int(ei), "fill_bucket": int(fi),
                       "edge_bps": float(fill_net * 1e4)})
    return {"trades": trades, "candidates": len(entry)}


def run_triangular_v1(trades: dict, a: str, b: str, fee_net: float, exec_cfg,
                      verbose: bool = False) -> dict:
    """trades: {symbol: loaded aggTrades} for AUSDT, BUSDT, BA symbols."""
    pair_a_usdt, pair_b_usdt, pair_ba = f"{a}USDT", f"{b}USDT", f"{b}{a}"
    if verbose:
        print(f"  cycle legs: {pair_a_usdt} {pair_b_usdt} {pair_ba} | "
              f"fee_net={fee_net*1e4:.1f}bps latency={exec_cfg.latency_ms}ms")

    quotes = {s: build_quote_series(t, exec_cfg.bucket_us) for s, t in trades.items()}
    merged = merge_cycle_quotes(quotes, [pair_a_usdt, pair_b_usdt, pair_ba],
                                exec_cfg.bucket_us)
    r1, r2 = cycle_rates(merged, a, b)
    net1 = r1 - 1 - fee_net
    net2 = r2 - 1 - fee_net

    latency_buckets = max(1, int(exec_cfg.latency_ms * 1000 / exec_cfg.bucket_us))
    rng = np.random.default_rng(exec_cfg.seed)
    res1 = simulate_latency_fills(net1, latency_buckets, rng, exec_cfg.fill_prob_leg)
    res2 = simulate_latency_fills(net2, latency_buckets, rng, exec_cfg.fill_prob_leg)

    def stats(res):
        tr = res["trades"]
        if not tr:
            return {"trades": 0, "gross_bps": 0.0, "net_bps": 0.0,
                    "net_pnl_usd": 0.0, "total_fees_usd": 0.0}
        edges = np.array([t["edge_bps"] for t in tr])
        notional = exec_cfg.notional_usd
        fees_usd = (fee_net * 1e4 / 1e4) * notional * len(tr)
        return {
            "trades": len(tr),
            "gross_bps": float(edges.mean()),
            "net_bps": float(edges.mean()),
            "net_pnl_usd": float((edges / 1e4 * notional).sum()),
            "total_fees_usd": float(fees_usd),
        }

    s1, s2 = stats(res1), stats(res2)
    return {
        "cycle": f"USDT->{a}->{b}->USDT (r1) | USDT->{b}->{a}->USDT (r2)",
        "buckets": int(len(net1)),
        "fee_net_bps": fee_net * 1e4,
        "latency_ms": exec_cfg.latency_ms,
        "notional_usd": exec_cfg.notional_usd,
        "dir1": s1, "dir2": s2,
        "combined": {
            "trades": s1["trades"] + s2["trades"],
            "net_pnl_usd": s1["net_pnl_usd"] + s2["net_pnl_usd"],
            "net_bps": (s1["net_bps"] * s1["trades"] + s2["net_bps"] * s2["trades"]) /
                       (s1["trades"] + s2["trades"]) if (s1["trades"] + s2["trades"]) else 0,
        },
        "candidates_total": res1["candidates"] + res2["candidates"],
    }
