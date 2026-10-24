"""ARBITRAGE_V2: spot vs USD-M futures basis on real tick data.
basis_bps = (futures_last / spot_last - 1) * 1e4 per 1s bucket.
Strategy: when basis >= open_thr (futures rich): short futures + long spot;
close when basis <= close_thr or max_hold_seconds elapsed.
Pnl = (open_basis - close_basis) - round_trip_costs. Market-neutral cash-and-carry.
"""
import os
import sys
import json
from datetime import date, timedelta
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from arbitrage.storage import load_aggtrades  # noqa: E402
import numpy as np

START = date(2026, 7, 25)
END = date(2026, 8, 9)
DAYS = [(START + timedelta(days=i)).isoformat() for i in range((END - START).days + 1)]

BUCKET_US = 1_000_000  # 1s buckets for basis (slow-moving)


@dataclass
class BasisConfig:
    open_thr_bps: float = 15.0     # open when futures rich by >= open_thr
    close_thr_bps: float = 5.0     # close when basis compressed to <= close_thr
    max_hold_seconds: int = 3600   # force close after 1h
    round_trip_cost_bps: float = 30.0  # spot taker 10 + fut taker 5, open+close
    notional_usd: float = 100.0
    min_open_seconds: int = 5      # persistence required before opening


def load_pair(day: str, symbol: str) -> tuple:
    spot = load_aggtrades(symbol, day, "spot")
    fut = load_aggtrades(symbol, day, "futures_um")
    return spot, fut


def build_basis_series(spot: dict, fut: dict, bucket_us: int = BUCKET_US) -> dict:
    def last_price(t: dict) -> np.ndarray:
        b = t["ts_us"] // bucket_us
        uniq = np.unique(b)
        last = np.searchsorted(b, uniq + 1, side="left") - 1
        p = np.full(len(uniq), np.nan)
        v = last >= 0
        p[v] = t["price"][last[v]]
        return uniq, p

    bs, ps = last_price(spot)
    bf, pf = last_price(fut)
    uniq = np.unique(np.concatenate([bs, bf]))
    pos_s = np.searchsorted(bs, uniq, side="right") - 1
    pos_f = np.searchsorted(bf, uniq, side="right") - 1
    ok = (pos_s >= 0) & (pos_f >= 0)
    basis = np.full(len(uniq), np.nan)
    basis[ok] = (pf[pos_f[ok]] / ps[pos_s[ok]] - 1) * 1e4
    return {"bucket_us": uniq, "basis_bps": basis, "spot_last": np.where(ok, ps[pos_s], np.nan),
            "fut_last": np.where(ok, pf[pos_f], np.nan)}


def simulate_basis(b: dict, cfg: BasisConfig, direction: str = "short_fut",
                   rng: np.random.Generator | None = None) -> dict:
    """direction='short_fut': futures rich (basis >= open) -> short fut / long spot.
    direction='long_fut': futures cheap (basis <= -open) -> long fut / short spot.
    Pnl for long_fut = (close_basis - open_basis) - costs (basis rises = futures catch up)."""
    basis = b["basis_bps"]
    n = len(basis)
    open_thr, close_thr = cfg.open_thr_bps, cfg.close_thr_bps
    max_hold_buckets = cfg.max_hold_seconds * (BUCKET_US // 1_000_000) if BUCKET_US == 1_000_000 \
        else cfg.max_hold_seconds
    min_hold = cfg.min_open_seconds

    if direction == "short_fut":
        active = basis >= open_thr
        exited = basis <= close_thr
    else:
        active = basis <= -open_thr
        exited = basis >= -close_thr

    d = np.diff(np.concatenate(([0], active.astype(np.int8), [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    runs = [(s, e) for s, e in zip(starts, ends) if e - s >= min_hold]
    if not runs and active.any() and (n - np.flatnonzero(active)[0] >= min_hold):
        runs = [(np.flatnonzero(active)[0], n)]

    trades = []
    for s, e in runs:
        window = basis[s:e]
        idx = np.flatnonzero(exited[s:e])
        if len(idx):
            ci = s + idx[0]
        else:
            ci = min(s + max_hold_buckets, n - 1)
        if ci <= s:
            continue
        open_bps = basis[s]
        close_bps = basis[ci]
        if direction == "short_fut":
            gross_bps = open_bps - close_bps  # basis compresses
        else:
            gross_bps = close_bps - open_bps  # basis widens up
        cost_bps = cfg.round_trip_cost_bps
        net_bps = gross_bps - cost_bps
        notional = cfg.notional_usd
        trades.append({
            "open_bucket": int(s), "close_bucket": int(ci),
            "hold_seconds": int((ci - s) * (BUCKET_US // 1_000_000)),
            "open_bps": float(open_bps), "close_bps": float(close_bps),
            "gross_bps": float(gross_bps), "cost_bps": cost_bps, "net_bps": float(net_bps),
            "net_pnl_usd": float(net_bps / 1e4 * notional),
        })
    return {"trades": trades}


def main():
    out = {"days": DAYS, "results": {}}
    for symbol in ("BTCUSDT", "ETHUSDT"):
        print(f"=== {symbol} basis ===")
        stats = {"n_buckets": 0, "mean_bps": [], "std_bps": [], "pct_ge_10": [],
                 "pct_ge_25": [], "max_bps": 0, "min_bps": 0}
        all_trades = []
        for day in DAYS:
            try:
                spot, fut = load_pair(day, symbol)
            except FileNotFoundError:
                continue
            b = build_basis_series(spot, fut)
            base = b["basis_bps"]
            valid = np.isfinite(base)
            stats["n_buckets"] += int(valid.sum())
            stats["mean_bps"].append(float(base[valid].mean()))
            stats["std_bps"].append(float(base[valid].std()))
            stats["pct_ge_10"].append(100 * float((base[valid] >= 10).sum()) / int(valid.sum()))
            stats["pct_ge_25"].append(100 * float((base[valid] >= 25).sum()) / int(valid.sum()))
            stats["max_bps"] = max(stats["max_bps"], float(base[valid].max()))
            stats["min_bps"] = min(stats["min_bps"], float(base[valid].min()))
            rng = np.random.default_rng(42)
            for direction in ("short_fut", "long_fut"):
                res = simulate_basis(b, BasisConfig(), direction, rng)
                for t in res["trades"]:
                    t["day"] = day
                    t["direction"] = direction
                all_trades.extend(res["trades"])

        r = {
            "buckets": stats["n_buckets"],
            "mean_bps": round(float(np.mean(stats["mean_bps"])), 3),
            "std_bps": round(float(np.mean(stats["std_bps"])), 3),
            "pct_time_basis_ge_10bps": round(float(np.mean(stats["pct_ge_10"])), 4),
            "pct_time_basis_ge_25bps": round(float(np.mean(stats["pct_ge_25"])), 4),
            "min_bps": round(stats["min_bps"], 3),
            "max_bps": round(stats["max_bps"], 3),
            "trades": len(all_trades),
        }
        if all_trades:
            nets = np.array([t["net_bps"] for t in all_trades])
            pnls = np.array([t["net_pnl_usd"] for t in all_trades])
            r.update({
                "trades_net_bps_mean": round(float(nets.mean()), 3),
                "trades_net_bps_positive": int((nets > 0).sum()),
                "net_pnl_usd_total": round(float(pnls.sum()), 2),
                "avg_hold_seconds": round(float(np.mean([t["hold_seconds"] for t in all_trades])), 1),
                "open_avg_bps": round(float(np.mean([t["open_bps"] for t in all_trades])), 3),
                "close_avg_bps": round(float(np.mean([t["close_bps"] for t in all_trades])), 3),
            })
        print(json.dumps(r, indent=1))
        out["results"][symbol] = r

    path = os.path.join(os.path.dirname(__file__), "..", "ARBITRAGE_V2_BASIS.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
