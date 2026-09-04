#!/usr/bin/env python
import sys, os, time
import numpy as np

sys.path.insert(0, r'C:\Users\IRAQ SOFT\Desktop\stratocrypto\crypto')
sys.path.insert(0, r'C:\Users\IRAQ SOFT\Desktop\stratocrypto\crypto\market_events\research')

from market_events.store import EventStore, DEFAULT_ROOT
from market_events.research.dataset import SymbolDataset, FEATURE_NAMES
from market_events.research import labels, regimes, experiments, stats as st, regimes as regs
from market_events.research import features_book as fb
from market_events.research.costs import build_cost_config
from market_events.research.run_research import _stats_for, _select_promising, _decay_table, _regime_table, _cross_asset, _k

OUT_DIR = r'C:\Users\IRAQ SOFT\Desktop\stratocrypto'
DATASETS_DIR = os.path.join(OUT_DIR, "research_data")
os.makedirs(DATASEDS_DIR, exist_ok=True)

HORIZONS_US = labels.HORIZONS_US
SYMBOLS = [("futures_um", "BTCUSDT"), ("futures_um", "ETHUSDT"), ("spot", "BTCUSDT")]
SYM_KEYS = [f"{m}:{s}" for m, s in SYMBOLS]
FUT_KEYS = SYM_KEYS[:2]
DEV_DAYS = regimes.DEV_DAYS
OOS_DAYS = regimes.OOS_DAYS

print("Step 0: Loading data...")
store = EventStore(DEFAULT_ROOT)
datasets = {}

t0 = time.time()
for market, sym in SYMBOLS:
    key = f"{market}:{sym}"
    first = store.load_day(market, sym, DEV_DAYS[0])
    cost_cfg = build_cost_config(first)
    ds = SymbolDataset(store, market, sym)
    df = ds.build(cost_cfg=cost_cfg, horizons_us=HORIZONS_US)
    datasets[key] = df
    print(f"  {key}: {len(df):,} samples built in {time.time()-t0:.1f}s")

print("\nStep 1: DEV statistics...")
ledger = experiments.Ledger()
dev_stats = {}

t1 = time.time()
for sym, df in datasets.items():
    sub = df[df["day"].isin(DEV_DAYS)]
    for feature in FEATURE_NAMES:
        for h in HORIZONS_US:
            fv = sub[feature].to_numpy(dtype=np.float64)
            lc = f"label_gross_{int(h/1_000_000)}s"
            valid = sub[lc].to_numpy(dtype=bool)
            y = sub[lc].to_numpy(dtype=np.float64)
            y[~valid] = np.nan
            s = st.compute_stats(fv, y, None, feature=feature, symbol=sym, horizon_us=h, split="DEV")
            key = (feature, sym, h)
            dev_stats[key] = s

print(f"  DEV stats: {len(dev_stats)} feature-symbol-horizon combos in {time.time()-t1:.1f}s")

print("\nStep 2: Select promising...")
promising = _select_promising(dev_stats, datasets)
print(f"  Promising features: {sorted(promising['features'])}")

print("\nStep 3: OOS statistics...")
oos_stats = {}
t2 = time.time()
for sym, df in datasets.items():
    sub = df[df["day"].isin(OOS_DAYS)]
    for feature in promising["features"]:
        for h in HORIZONS_US:
            fv = sub[feature].to_numpy(dtype=np.float64)
            lc = f"label_gross_{int(h/1_000_000)}s"
            valid = sub[lc].to_numpy(dtype=bool)
            y = sub[lc].to_numpy(dtype=np.float64)
            y[~valid] = np.nan
            s = st.compute_stats(fv, y, None, feature=feature, symbol=sym, horizon_us=h, split="OOS")
            key = (feature, sym, h)
            oos_stats[key] = s

print(f"  OOS stats: {len(oos_stats)} entries in {time.time()-t2:.1f}s")

print("\nStep 4: Decay table...")
t3 = time.time()
decay = _decay_table(dev_stats)
print(f"  Decay table: {len(decay)} features in {time.time()-t3:.1f}s")

print("\nStep 5: Regime statistics...")
t4 = time.time()
regimes_out = _regime_table(datasets, promising["features"])
print(f"  Regime stats: {len(regimes_out)} features in {time.time()-t4:.1f}s")

print("\nStep 6: Cross-asset analysis...")
t5 = time.time()
cross = _cross_asset(datasets)
print(f"  Cross-asset: {len(cross.get('pairs', {}))} pairs in {time.time()-t5:.1f}s")

print("\nAll steps completed!")