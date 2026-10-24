"""Smart wide scan: pre-fetch all closes once, filter by correlation, then
only run expensive cointegration on correlated pairs. Much faster."""
from datetime import datetime, timezone
import json, os, numpy as np
import stat_arb
from data_loader.binance_ohlcv import fetch_klines

UNIVERSE = sorted(set([
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT",
    "ADAUSDT","LTCUSDT","LINKUSDT","DOTUSDT","AVAXUSDT","TRXUSDT",
    "FILUSDT","MATICUSDT","ATOMUSDT","ETCUSDT","NEARUSDT","APTUSDT",
    "ARBUSDT","OPUSDT","SUIUSDT","TONUSDT","BCHUSDT","UNIUSDT",
    "AAVEUSDT","MKRUSDT","INJUSDT","LDOUSDT","RUNEUSDT","1000SATSUSDT",
    "PEPEUSDT","SHIBUSDT","FLOKIUSDT","BONKUSDT","STXUSDT","SEIUSDT",
    "JUPUSDT","WLDUSDT","GALAUSDT","SNXUSDT","CRVUSDT","COMPUSDT",
]))

CORR_LOOKBACK = 500
CORR_MIN = 0.7  # only test pairs with |correlation| > 0.7
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

print(f"fetching {len(UNIVERSE)} close series ({CORR_LOOKBACK} bars)...", flush=True)
closes = {}
for sym in UNIVERSE:
    try:
        k = fetch_klines(sym, "1h", CORR_LOOKBACK)
        if k and len(k) >= 200:
            closes[sym] = np.array([float(r[4]) for r in k])
    except Exception:
        pass

syms = sorted(closes.keys())
print(f"got {len(syms)} series", flush=True)

# correlation pre-filter
pairs_to_test = []
for i in range(len(syms)):
    for j in range(i + 1, len(syms)):
        a, b = syms[i], syms[j]
        ca, cb = closes[a], closes[b]
        n = min(len(ca), len(cb))
        if n < 200:
            continue
        # log-return correlation
        ra = np.diff(np.log(ca[-n:] + 1e-12))
        rb = np.diff(np.log(cb[-n:] + 1e-12))
        if len(ra) < 50:
            continue
        corr = np.corrcoef(ra, rb)[0, 1]
        if abs(corr) > CORR_MIN:
            pairs_to_test.append((a, b, abs(corr)))

pairs_to_test.sort(key=lambda x: -x[2])
print(f"correlated pairs (|r|>{CORR_MIN}): {len(pairs_to_test)}", flush=True)

count = 0
for a, b, corr in pairs_to_test:
    pair = f"{a}/{b}"
    if pair in done:
        continue
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
        print(f"  ROBUST {pair:24s} corr={corr:.2f} {wins} mean={mean_s}", flush=True)

data["pairs"].sort(key=lambda x: -x["mean_sharpe"])
save(data)
print(f"\nTotal robust: {len(data['pairs'])} (new this run: {count})")
for p in data["pairs"]:
    print(f"  {p['pair']:24s} {p['mean_sharpe']}")
print(f"saved {OUT}")
