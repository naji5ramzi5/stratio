"""Multi-timeframe pairs scanner + portfolio risk manager.
The core money-making engine: find pairs on EVERY timeframe, size by edge,
and control risk at the portfolio level."""
from datetime import datetime, timezone
import json, os, numpy as np, time
from data_loader.binance_ohlcv import fetch_klines
from statsmodels.tsa.stattools import coint

TIMEFRAMES = {
    "5m":  {"bar_hours": 5/60,  "lookback": 2000, "train": 800,  "step": 200, "max_hold": 288, "z_entry": 1.5, "z_exit": 0.3, "z_stop": 3.0},
    "15m": {"bar_hours": 0.25,   "lookback": 1500, "train": 600,  "step": 150, "max_hold": 96,  "z_entry": 1.8, "z_exit": 0.4, "z_stop": 3.2},
    "1h":  {"bar_hours": 1,      "lookback": 3000, "train": 1000, "step": 250, "max_hold": 168, "z_entry": 2.0, "z_exit": 0.5, "z_stop": 3.5},
    "4h":  {"bar_hours": 4,      "lookback": 1500, "train": 500,  "step": 125, "max_hold": 42,  "z_entry": 2.0, "z_exit": 0.5, "z_stop": 3.5},
    "1d":  {"bar_hours": 24,     "lookback": 800,  "train": 300,  "step": 75,  "max_hold": 14,  "z_entry": 2.2, "z_exit": 0.6, "z_stop": 3.8},
}

UNIVERSE = sorted(set([
    "BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT","XRPUSDT","DOGEUSDT",
    "ADAUSDT","LTCUSDT","LINKUSDT","DOTUSDT","AVAXUSDT","TRXUSDT",
    "FILUSDT","MATICUSDT","ATOMUSDT","ETCUSDT","NEARUSDT","APTUSDT",
    "ARBUSDT","OPUSDT","SUIUSDT","TONUSDT","BCHUSDT","UNIUSDT",
    "AAVEUSDT","MKRUSDT","INJUSDT","LDOUSDT","RUNEUSDT","1000SATSUSDT",
    "PEPEUSDT","SHIBUSDT","FLOKIUSDT","BONKUSDT","STXUSDT","SEIUSDT",
    "JUPUSDT","WLDUSDT","GALAUSDT","SNXUSDT","CRVUSDT","COMPUSDT",
]))

OUT_DB = "pairs_db.json"

def aligned(a_dict, b_dict, min_overlap=300):
    common = sorted(set(a_dict) & set(b_dict))
    if len(common) < min_overlap:
        return None, None
    return np.array([a_dict[t] for t in common]), np.array([b_dict[t] for t in common])

def wf_backtest(a, b, train_bars, step, z_entry, z_exit, z_stop, max_hold):
    la, lb = np.log(a), np.log(b)
    n = len(a)
    rets, trades_list = [], []
    pos, entry_spread, bars_held = 0, 0.0, 0
    start = train_bars
    while start < n:
        end = min(start + step, n)
        tr_lo = max(0, start - train_bars)
        if start - tr_lo < 80:
            start += step
            continue
        a_tr, b_tr = a[tr_lo:start], b[tr_lo:start]
        h = np.polyfit(np.log(b_tr), np.log(a_tr), 1)[0]
        try:
            _, pval, _ = coint(np.log(a_tr), np.log(b_tr))
        except Exception:
            start += step
            continue
        if pval >= 0.05:
            start += step
            continue
        s_tr = np.log(a_tr) - h * np.log(b_tr)
        mu, sd = float(np.mean(s_tr)), float(np.std(s_tr))
        if sd <= 0:
            start += step
            continue
        seg_ret, seg_trades, seg_wins = 0.0, 0, 0
        for i in range(start, min(end, n)):
            if pos != 0:
                bars_held += 1
            z = (la[i] - h * lb[i] - mu) / sd
            if pos == 0 and abs(z) >= z_entry:
                pos = 1 if z < 0 else -1
                entry_spread = la[i] - h * lb[i]
                bars_held = 0
            elif pos != 0:
                if abs(z) <= z_exit or abs(z) >= z_stop or bars_held >= max_hold:
                    exit_spread = la[i] - h * lb[i]
                    raw = pos * (exit_spread - entry_spread)
                    cost = 0.0025
                    net = raw - cost
                    seg_ret += net
                    seg_trades += 1
                    if net > 0:
                        seg_wins += 1
                    pos, bars_held = 0, 0
        if seg_trades > 0:
            rets.append(seg_ret)
            trades_list.append((seg_trades, seg_wins))
        start += step
    if not rets:
        return None
    tot_trades = sum(t for t, _ in trades_list)
    tot_wins = sum(w for _, w in trades_list)
    tot_ret = sum(rets)
    bars_per_year = 365 * 24 / TIMEFRAMES["1h"]["bar_hours"]  # normalized
    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-9)) * np.sqrt(max(tot_trades, 1))
    return {"sharpe": round(sharpe, 2), "n_trades": tot_trades, "win_rate": round(tot_wins / max(tot_trades, 1), 2),
            "total_ret_pct": round(tot_ret * 100, 2), "avg_ret_pct": round(np.mean(rets) * 100, 3)}


# ---- fetch all series per timeframe ----
cache = {}
print("Fetching all symbols across timeframes...", flush=True)
for tf in TIMEFRAMES:
    cache[tf] = {}
    for sym in UNIVERSE:
        try:
            k = fetch_klines(sym, tf, TIMEFRAMES[tf]["lookback"])
            if k and len(k) >= TIMEFRAMES[tf]["train"] + 50:
                cache[tf][sym] = {int(r[0]): float(r[4]) for r in k}
        except Exception:
            pass
    print(f"  {tf}: {len(cache[tf])} series", flush=True)

# ---- load existing DB ----
if os.path.exists(OUT_DB):
    with open(OUT_DB) as f:
        db = json.load(f)
    done = {(p["pair"], p["tf"]) for p in db["results"]}
else:
    db = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "results": []}
    done = set()

def save():
    with open(OUT_DB, "w") as f:
        json.dump(db, f, indent=2)

# ---- scan ----
results = []
tested = 0
for tf, cfg in TIMEFRAMES.items():
    syms = sorted(cache[tf].keys())
    print(f"\nScanning {tf} ({len(syms)} syms -> {len(syms)*(len(syms)-1)//2} pairs)...", flush=True)
    count = 0
    for i in range(len(syms)):
        for j in range(i + 1, len(syms)):
            a, b = syms[i], syms[j]
            key = (f"{a}/{b}", tf)
            if key in done:
                continue
            ca, cb = aligned(cache[tf][a], cache[tf][b], min_overlap=cfg["train"] + 50)
            if ca is None:
                continue
            tested += 1
            m = wf_backtest(ca, cb, cfg["train"], cfg["step"],
                            cfg["z_entry"], cfg["z_exit"], cfg["z_stop"], cfg["max_hold"])
            if m and m["n_trades"] >= 3 and m["sharpe"] > 0.5:
                rec = {"pair": f"{a}/{b}", "tf": tf, **m}
                results.append(rec)
                db["results"].append(rec)
                save()
                count += 1
                print(f"  ROBUST {a}/{b} {tf}: sharpe={m['sharpe']} trades={m['n_trades']} wr={m['win_rate']}", flush=True)
    print(f"  {tf}: {count} robust pairs")

# ---- summary ----
print(f"\n=== TOTAL: {len(db['results'])} robust pair-timeframe combos ===")
by_tf = {}
for r in db["results"]:
    by_tf.setdefault(r["tf"], []).append(r)
for tf in sorted(by_tf):
    pairs = sorted(by_tf[tf], key=lambda x: -x["sharpe"])
    print(f"\n{tf} ({len(pairs)} pairs):")
    for p in pairs[:10]:
        print(f"  {p['pair']:24s} sharpe={p['sharpe']:6.2f} trades={p['n_trades']:3d} wr={p['win_rate']:.0%} ret={p['total_ret_pct']:.1f}%")

print(f"\nTop 20 overall:")
for r in sorted(db["results"], key=lambda x: -x["sharpe"])[:20]:
    print(f"  {r['pair']:24s} {r['tf']:4s} sharpe={r['sharpe']:6.2f} trades={r['n_trades']:3d}")
print(f"saved {OUT_DB}")
