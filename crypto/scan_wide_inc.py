"""Incremental wide scan: save each pair result immediately, resume on rerun."""
from datetime import datetime, timezone
import json, os, numpy as np
import stat_arb, itertools

UNIVERSE = sorted(set([
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT",
    "ADAUSDT","LTCUSDT","LINKUSDT","DOTUSDT","AVAXUSDT","TRXUSDT",
    "FILUSDT","MATICUSDT","ATOMUSDT","ETCUSDT","NEARUSDT","APTUSDT",
    "ARBUSDT","OPUSDT","SUIUSDT","TONUSDT","BCHUSDT","UNIUSDT",
    "AAVEUSDT","MKRUSDT","INJUSDT","LDOUSDT","RUNEUSDT","1000SATSUSDT",
    "PEPEUSDT","SHIBUSDT","FLOKIUSDT","BONKUSDT","STXUSDT","SEIUSDT",
    "JUPUSDT","WLDUSDT","GALAUSDT","SNXUSDT","CRVUSDT","COMPUSDT",
]))

WINDOWS = [(2500,1000),(3000,1000),(3500,1000)]
MIN_WINDOWS = 2
MIN_SHARPE = 0.5
OUT = "pairs_wide.json"

def load_existing():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "universe_size": len(UNIVERSE), "pairs": []}

def save(data):
    with open(OUT, "w") as f:
        json.dump(data, f, indent=2)

data = load_existing()
done = {p["pair"] for p in data["pairs"]}
print(f"already done: {len(done)}", flush=True)

# stage-1 screen first (cheap coint + half-life)
screen = stat_arb.scan_pairs(UNIVERSE, "1h", 3000, p_max=0.05, hl_min=4, hl_max=800)
short = [r for r in screen if r["status"] == "shortlisted"]
print(f"shortlisted: {len(short)} (of {len(UNIVERSE)*(len(UNIVERSE)-1)//2})", flush=True)

count = 0
for r in short:
    pair = r["pair"]
    if pair in done:
        continue
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
    if is_robust:
        mean_s = round(np.mean([s for s, _ in wins.values()]), 2)
        data["pairs"].append({"pair": pair, "windows": wins, "mean_sharpe": mean_s})
        save(data)
        count += 1
        print(f"  ROBUST {pair:24s} {wins} mean={mean_s}", flush=True)

# final
data["pairs"].sort(key=lambda x: -x["mean_sharpe"])
save(data)
robust = data["pairs"]
print(f"\nTotal robust: {len(robust)} (new: {count})")
for p in robust:
    print(f"  {p['pair']:24s} {p['mean_sharpe']}")
print(f"saved {OUT}")
