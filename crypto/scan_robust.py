"""Robust multi-window pairs scan: keep only pairs positive in >=2 windows."""
from datetime import datetime, timezone
import json, numpy as np
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

print(f"universe={len(UNIVERSE)} stage-1 lookback=3000", flush=True)
screen = stat_arb.scan_pairs(UNIVERSE, "1h", 3000, p_max=0.05, hl_min=4, hl_max=800)
short = [r for r in screen if r["status"] == "shortlisted"]
print(f"shortlisted={len(short)}", flush=True)

stability = {}
for r in short:
    a, b = r["pair"].split("/")
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
    stability[r["pair"]] = wins
    flag = "ROBUST" if len(wins) >= MIN_WINDOWS else "unstable"
    print(f"  {r['pair']:22s} {flag} {wins}", flush=True)

robust = {p: w for p, w in stability.items() if len(w) >= MIN_WINDOWS}
print(f"\nROBUST pairs (positive Sharpe in >= {MIN_WINDOWS} windows): {len(robust)}")
for p, w in robust.items():
    sharpes = [s for s, _ in w.values()]
    print(f"  {p:22s} sharpes={sharpes} mean={np.mean(sharpes):.2f}")

out = {
    "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "universe_size": len(UNIVERSE),
    "windows": [f"{lb}/{tb}" for lb, tb in WINDOWS],
    "pairs": [{"pair": p, "windows": w,
               "mean_sharpe": round(np.mean([s for s, _ in w.values()]), 2)}
              for p, w in robust.items()],
}
with open("pairs_robust.json", "w") as f:
    json.dump(out, f, indent=2)
print("saved pairs_robust.json")
