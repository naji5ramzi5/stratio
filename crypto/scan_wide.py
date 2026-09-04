"""Extended robust scan: 50+ symbols, keep all pairs with Sharpe > 0.5 in >= 2 windows."""
from datetime import datetime, timezone
import json, os, numpy as np
import stat_arb

UNIVERSE = sorted(set([
    # majors
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT",
    "ADAUSDT","LTCUSDT","LINKUSDT","DOTUSDT","AVAXUSDT","TRXUSDT",
    # mid-caps
    "FILUSDT","MATICUSDT","ATOMUSDT","ETCUSDT","NEARUSDT","APTUSDT",
    "ARBUSDT","OPUSDT","SUIUSDT","TONUSDT","BCHUSDT","UNIUSDT",
    "AAVEUSDT","MKRUSDT","INJUSDT","LDOUSDT","RUNEUSDT","1000SATSUSDT",
    # memes / high vol
    "PEPEUSDT","SHIBUSDT","FLOKIUSDT","BONKUSDT","WIFUSDT","MEMEUSDT",
    "TURBOUSDT","MYROUSDT","BOMEUSDT",
    # layer 2 / new
    "STXUSDT","SEIUSDT","PYTHUSDT","JUPUSDT","WLDUSDT","STRKUSDT",
    "CELOUSDT","NEOUSDT","GMTUSDT","GALAUSDT","IMXUSDT","SNXUSDT",
    "CRVUSDT","COMPUSDT","YFIUSDT","ZRXUSDT","BALUSDT","ENSUSDT",
]))

WINDOWS = [(2500,1000),(3000,1000),(3500,1000)]
MIN_WINDOWS = 2
MIN_SHARPE = 0.5  # lower threshold to find more pairs
OUT = "pairs_active.json"

print(f"universe={len(UNIVERSE)} symbols -> {len(UNIVERSE)*(len(UNIVERSE)-1)//2} pairs", flush=True)

screen = stat_arb.scan_pairs(UNIVERSE, "1h", 3000, p_max=0.05, hl_min=4, hl_max=800)
short = [r for r in screen if r["status"] == "shortlisted"]
print(f"shortlisted: {len(short)}", flush=True)

results = []
for r in short:
    pair = r["pair"]
    a, b = pair.split("/")
    wins = {}
    for lb, tb in WINDOWS:
        try:
            df = stat_arb.fetch_aligned_pair(a, b, "1h", lb)
            if df is None or len(df) < tb + 50:
                continue
            m = stat_arb.walk_forward(df, "1h", train_bars=tb, step=tb//4,
                                      z_entry=2.0, z_exit=0.5, cost_bps=25)
            if m["cointegrated"] and m["n_trades"] > 0 and m["sharpe"] > 0:
                wins[f"w{lb}"] = (round(m["sharpe"], 2), m["n_trades"])
        except Exception:
            pass
    is_robust = len(wins) >= MIN_WINDOWS and all(s >= MIN_SHARPE for s, _ in wins.values())
    flag = "ROBUST" if is_robust else "weak"
    if is_robust:
        mean_s = round(np.mean([s for s, _ in wins.values()]), 2)
        results.append({"pair": pair, "windows": wins, "mean_sharpe": mean_s})
        print(f"  {pair:24s} {flag} {wins} mean_sharpe={mean_s}", flush=True)
    else:
        print(f"  {pair:24s} {flag} {wins}", flush=True)

results.sort(key=lambda x: -x["mean_sharpe"])
out = {
    "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "universe_size": len(UNIVERSE),
    "min_sharpe": MIN_SHARPE,
    "min_windows": MIN_WINDOWS,
    "windows": [f"{a}/{b}" for a, b in WINDOWS],
    "pairs": results,
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)

print(f"\nTotal robust pairs: {len(results)}")
for p in results:
    print(f"  {p['pair']:24s} mean_sharpe={p['mean_sharpe']}")
print(f"saved {OUT}")
