"""Ultra-fast scan: fetch all closes ONCE, align locally, test cointegration locally.
No repeated network calls — 10-50x faster than stat_arb.scan_pairs."""
from datetime import datetime, timezone
import json, os, numpy as np, time
from data_loader.binance_ohlcv import fetch_klines
from statsmodels.tsa.stattools import coint

UNIVERSE = sorted(set([
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT",
    "ADAUSDT","LTCUSDT","LINKUSDT","DOTUSDT","AVAXUSDT","TRXUSDT",
    "FILUSDT","MATICUSDT","ATOMUSDT","ETCUSDT","NEARUSDT","APTUSDT",
    "ARBUSDT","OPUSDT","SUIUSDT","TONUSDT","BCHUSDT","UNIUSDT",
    "AAVEUSDT","MKRUSDT","INJUSDT","LDOUSDT","RUNEUSDT","1000SATSUSDT",
    "PEPEUSDT","SHIBUSDT","FLOKIUSDT","BONKUSDT","STXUSDT","SEIUSDT",
    "JUPUSDT","WLDUSDT","GALAUSDT","SNXUSDT","CRVUSDT","COMPUSDT",
]))

MAX_BARS = 3500
WINDOWS = [(2500,1000),(3000,1000),(3500,1000)]
MIN_WINDOWS = 2
MIN_SHARPE = 0.5
OUT = "pairs_active.json"

def load_existing():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "universe_size": len(UNIVERSE), "pairs": []}

def save(data):
    with open(OUT, "w") as f:
        json.dump(data, f, indent=2)

def aligned_pair(dict_a, dict_b):
    """Return aligned (a,b) close arrays from timestamp-keyed dicts."""
    common = sorted(set(dict_a.keys()) & set(dict_b.keys()))
    if len(common) < 500:
        return None, None
    a = np.array([dict_a[t] for t in common], dtype=float)
    b = np.array([dict_b[t] for t in common], dtype=float)
    return a, b

def walk_forward_local(a, b, train_bars, step):
    """Local walk-forward backtest (no network)."""
    la = np.log(a)
    lb = np.log(b)
    n = len(a)
    start = train_bars
    rets = []
    trades = 0
    wins = 0
    while start < n:
        end = min(start + step, n)
        tr_lo = max(0, start - train_bars)
        a_tr, b_tr = a[tr_lo:start], b[tr_lo:start]
        la_tr, lb_tr = la[tr_lo:start], lb[tr_lo:start]
        if len(a_tr) < 100:
            start += step
            continue
        # hedge
        h = np.polyfit(lb_tr, la_tr, 1)[0]
        # coint test
        try:
            _, pval, _ = coint(la_tr, lb_tr)
        except Exception:
            start += step
            continue
        if pval >= 0.05:
            start += step
            continue
        # baseline
        s_tr = la_tr - h * lb_tr
        mu, sd = float(np.mean(s_tr)), float(np.std(s_tr))
        if sd <= 0:
            start += step
            continue
        # test window
        for i in range(start, min(end, n)):
            z = (la[i] - h * lb[i] - mu) / sd
            if abs(z) >= 2.0 and i + 1 < n:
                # one-bar return of the spread
                dz = (la[i + 1] - h * lb[i + 1]) - (la[i] - h * lb[i])
                direction = 1.0 if z < 0 else -1.0
                ret = direction * dz * 100 - 0.25  # 25bps cost
                rets.append(ret)
                trades += 1
                if ret > 0:
                    wins += 1
        start += step
    if not rets:
        return None
    rets = np.array(rets)
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-9)) * np.sqrt(24 * 365 / max(step, 1))
    return {"sharpe": round(sharpe, 2), "n_trades": trades, "win_rate": round(wins / trades, 2) if trades else 0}


# ---- fetch all once ----
data = load_existing()
done = {p["pair"] for p in data["pairs"]}
print(f"already done: {len(done)}", flush=True)

print(f"fetching {len(UNIVERSE)} series...", flush=True)
t0 = time.time()
closes = {}
for sym in UNIVERSE:
    try:
        k = fetch_klines(sym, "1h", MAX_BARS)
        if k and len(k) >= 1000:
            closes[sym] = {int(r[0]): float(r[4]) for r in k}
    except Exception:
        pass
print(f"fetched {len(closes)} series in {time.time() - t0:.1f}s", flush=True)

# ---- test all pairs locally ----
syms = sorted(keys := list(closes.keys()))
pairs_to_test = [(syms[i], syms[j]) for i in range(len(syms)) for j in range(i + 1, len(syms))]
print(f"testing {len(pairs_to_test)} pairs locally...", flush=True)

count = 0
tested = 0
for a, b in pairs_to_test:
    pair = f"{a}/{b}"
    if pair in done:
        continue
    ca, cb = aligned_pair(closes.get(a, {}), closes.get(b, {}))
    if ca is None:
        continue
    tested += 1
    wins = {}
    for lb, tb in WINDOWS:
        if len(ca) < lb:
            continue
        m = walk_forward_local(ca[-lb:], cb[-lb:], tb, tb // 4)
        if m and m["n_trades"] > 0 and m["sharpe"] > 0:
            wins[f"w{lb}"] = (m["sharpe"], m["n_trades"])
    is_robust = len(wins) >= MIN_WINDOWS and all(s >= MIN_SHARPE for s, _ in wins.values())
    if is_robust:
        mean_s = round(np.mean([s for s, _ in wins.values()]), 2)
        data["pairs"].append({"pair": pair, "windows": wins, "mean_sharpe": mean_s})
        save(data)
        count += 1
        print(f"  ROBUST {pair:24s} {wins} mean={mean_s}", flush=True)

data["pairs"].sort(key=lambda x: -x["mean_sharpe"])
save(data)
print(f"\nTested: {tested} | Robust: {len(data['pairs'])} (new: {count})")
for p in data["pairs"]:
    print(f"  {p['pair']:24s} {p['mean_sharpe']}")
print(f"saved {OUT}")
