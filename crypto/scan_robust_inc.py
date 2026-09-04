"""Incremental robust scan: append each pair result so interruptions don't lose data."""
from datetime import datetime, timezone
import json, os, numpy as np
import stat_arb

UNIVERSE = sorted(set([
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT",
    "ADAUSDT","LTCUSDT","LINKUSDT","DOTUSDT","AVAXUSDT","TRXUSDT",
    "FILUSDT","MATICUSDT","ATOMUSDT","ETCUSDT","NEARUSDT","APTUSDT",
    "ARBUSDT","OPUSDT","SUIUSDT","TONUSDT","PEPEUSDT","SHIBUSDT",
    "BCHUSDT","UNIUSDT","AAVEUSDT","MKRUSDT","INJUSDT","1000SATSUSDT",
    "LDOUSDT","RUNEUSDT",
]))
WINDOWS = [(2500,1000),(3000,1000),(3500,1000)]
MIN_WINDOWS = 2
OUT = "pairs_robust.json"

def load_existing():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "universe_size": len(UNIVERSE), "windows": [f"{a}/{b}" for a,b in WINDOWS],
            "pairs": []}

def save(data):
    with open(OUT, "w") as f:
        json.dump(data, f, indent=2)

data = load_existing()
done = {p["pair"] for p in data["pairs"]}
print(f"already done: {len(done)}", flush=True)

screen = stat_arb.scan_pairs(UNIVERSE, "1h", 3000, p_max=0.05, hl_min=4, hl_max=800)
short = [r for r in screen if r["status"] == "shortlisted"]
print(f"shortlisted: {len(short)}", flush=True)

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
    is_robust = len(wins) >= MIN_WINDOWS
    flag = "ROBUST" if is_robust else "unstable"
    print(f"  {pair:22s} {flag} {wins}", flush=True)
    if is_robust:
        data["pairs"].append({"pair": pair, "windows": wins,
                              "mean_sharpe": round(np.mean([s for s,_ in wins.values()]), 2)})
        save(data)

# final summary
robust = data["pairs"]
print(f"\nROBUST: {len(robust)}")
for p in sorted(robust, key=lambda x: -x["mean_sharpe"]):
    print(f"  {p['pair']:22s} mean_sharpe={p['mean_sharpe']}")
print("saved", OUT)
