"""Research leaderboard v2: aggregates ALL arbitrage experiments (V1-V5) + statuses."""
import json
import os

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")


def load(name):
    p = os.path.join(ROOT, name)
    if os.path.exists(p):
        return json.load(open(p))
    return None


def main():
    v1 = load("ARBITRAGE_V1_TRIANGULAR.json")
    v2 = load("ARBITRAGE_V2_BASIS.json")
    v3 = load("ARBITRAGE_V3_DEV_OOS.json")
    v4 = load("ARBITRAGE_V4_MAKER_SWEEP.json")
    v4d = load("ARBITRAGE_V4D_MAKER_SEQ.json")
    v4e = load("ARBITRAGE_V4E_MAKER_DEVOOS.json")
    v5 = load("ARBITRAGE_V5_FUNDING.json")

    rows = []

    # V1 triangular (taker)
    for k, r in (v1 or {}).get("results", {}).items():
        rows.append({
            "experiment": "TRIANGULAR_V1_TAKER", "cycle": k,
            "latency_ms": k.split("_")[-1][3:], "trades": r["trades"],
            "candidates": r["candidates"], "net_bps": r["net_bps"],
            "net_pnl_usd": r["net_pnl_usd"],
            "status": "REJECTED" if r["trades"] == 0 else "REVIEW",
        })

    # V2 basis
    for k, r in (v2 or {}).get("results", {}).items():
        rows.append({
            "experiment": "BASIS_V1_TAKER", "cycle": k, "latency_ms": "n/a",
            "trades": r["trades"], "candidates": r.get("trades", 0),
            "net_bps": r.get("trades_net_bps_mean", 0),
            "net_pnl_usd": r.get("net_pnl_usd_total", 0),
            "status": "REJECTED" if r["trades"] == 0 else "REVIEW",
        })

    # V3 DEV/OOS (taker triangular)
    for k, r in (v3 or {}).get("results", {}).items():
        rows.append({
            "experiment": "TRIANGULAR_V3_TAKER_DEVOOS", "cycle": k,
            "latency_ms": k.split("_")[-1][3:],
            "trades": r["dev"]["trades"] + r["oos"]["trades"],
            "candidates": r["dev"]["candidates"] + r["oos"]["candidates"],
            "net_bps": 0, "net_pnl_usd": r["dev"]["net_pnl_usd"] + r["oos"]["net_pnl_usd"],
            "dev_trades": r["dev"]["trades"], "oos_trades": r["oos"]["trades"],
            "status": "REJECTED",
        })

    # V4 maker sweep (distribution)
    if v4:
        rows.append({
            "experiment": "MAKER_V4_PERPS", "cycle": "BTC/ETH perps",
            "latency_ms": "0-500", "trades": "n/a", "candidates": v4["buckets"],
            "net_bps": v4["net_bps_mean"], "net_pnl_usd": "n/a",
            "note": "gross distribution, see sweep",
            "status": "EXPLORATORY",
        })

    # V4D sequential maker (robustness)
    for k, r in (v4d or {}).get("results", {}).items():
        rows.append({
            "experiment": "MAKER_V4D_SEQ", "cycle": "BTC/ETH perps",
            "latency_ms": k, "trades": r["completed"], "candidates": r["candidates"],
            "net_bps": r["net_bps_mean"], "net_pnl_usd": r["net_pnl_usd"],
            "failures": r["failures"],
            "status": "PROMISING" if r["net_pnl_usd"] > 0 and r["failures"] < r["completed"] * 2
                       else "REJECTED",
        })

    # V4E DEV/OOS maker
    for k, r in (v4e or {}).get("results", {}).items():
        d, o = r["dev"], r["oos"]
        ok = d["net_pnl_usd"] > 0 and o["net_pnl_usd"] > 0
        rows.append({
            "experiment": "MAKER_V4E_DEVOOS", "cycle": "BTC/ETH perps",
            "latency_ms": k, "trades": d["completed"] + o["completed"],
            "candidates": "n/a", "net_bps": 0,
            "net_pnl_usd": d["net_pnl_usd"] + o["oos"] if False else d["net_pnl_usd"] + o["net_pnl_usd"],
            "dev_pnl": d["net_pnl_usd"], "oos_pnl": o["net_pnl_usd"],
            "status": "OOS_VALIDATED" if ok else "REJECTED",
        })

    # V5 funding
    for sym, r in (v5 or {}).get("results", {}).items():
        t = r["trades"].get("thr3_0", {})
        always = r["trades"].get("always", {})
        rows.append({
            "experiment": "FUNDING_V5_CARRY", "cycle": sym, "latency_ms": "n/a",
            "trades": t.get("trades", 0), "candidates": "n/a", "net_bps": 0,
            "net_pnl_usd": t.get("net_pnl_usd_total", 0) + always.get("net_pnl_usd_total", 0),
            "apr_pct": r["stats"]["annual_apr_pct"],
            "status": "REJECTED" if t.get("trades", 0) == 0 else "REVIEW",
        })

    counts = {"total": len(rows), "rejected": 0, "promising": 0, "oos_validated": 0,
              "exploratory": 0, "review": 0}
    for r in rows:
        counts[r["status"].lower()] = counts.get(r["status"].lower(), 0) + 1
        counts["total"] = counts["total"]  # noqa

    lb = {
        "updated": "2026-08-10",
        "data": "Binance spot+futures aggTrades 16 days + funding 166 days + live depth recorder",
        "verdict": (
            "TAKER arbitrage (triangular spot, basis, funding): ALL REJECTED. "
            "MAKER triangular on perps: PROMISING in DEV and OOS at p_fill>=0.2, "
            "1s windows, thresholds >=8bps — execution-quality dependent; "
            "queue probability to be calibrated by live depth recorder."
        ),
        "counts": counts,
        "rows": rows,
    }
    out = os.path.join(ROOT, "ARBITRAGE_LEADERBOARD.json")
    json.dump(lb, open(out, "w"), indent=2)
    print(json.dumps(counts, indent=2))
    for r in rows:
        if r["status"] in ("PROMISING", "OOS_VALIDATED"):
            print(f"  {r['status']:14s} {r['experiment']:24s} {r['latency_ms']} pnl={r['net_pnl_usd']}")
    print("saved ->", out)


if __name__ == "__main__":
    main()
