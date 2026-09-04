#!/usr/bin/env python
import sys
import os
import time
import numpy as np

# Minimal test
print("TEST START", flush=True)

sys.path.insert(0, r'C:\Users\IRAQ SOFT\Desktop\stratocrypto\crypto')
sys.path.insert(0, r'C:\Users\IRAQ SOFT\Desktop\stratocrypto\crypto\market_events\research')

from market_events.store import EventStore, DEFAULT_ROOT
print("import OK", flush=True)

store = EventStore(DEFAULT_ROOT)
print("store OK", flush=True)

HORIZONS_US = []
from market_events.research import labels
HORIZONS_US = labels.HORIZONS_US

SYMBOLS = [("futures_um", "BTCUSDT"), ("futures_um", "ETHUSDT"), ("spot", "BTCUSDT")]
SYM_KEYS = [f"{m}:{s}" for m, s in SYMBOLS]
FUT_KEYS = SYM_KEYS[:2]

OUT_DIR = r'C:\Users\IRAQ SOFT\Desktop\stratocrypto'
DATASETS_DIR = os.path.join(OUT_DIR, "research_data")
os.makedirs(DATASETS_DIR, exist_ok=True)

from market_events.research.dataset import SymbolDataset
from market_events.research import regimes, experiments, stats as st, features_book as fb
from market_events.research.costs import build_cost_config
from market_events.research.run_research import _stats_for, _select_promising, _decay_table, _regime_table, _cross_asset, _k

DEV_DAYS = regimes.DEV_DAYS
OOS_DAYS = regimes.OOS_DAYS

print("Step 0: Loading data...", flush=True)
datasets = {}
t0 = time.time()
for market, sym in SYMBOLS:
    key = f"{market}:{sym}"
    first = store.load_day(market, sym, DEV_DAYS[0])
    cost_cfg = build_cost_config(first)
    ds = SymbolDataset(store, market, sym)
    df = ds.build(cost_cfg=cost_cfg, horizons_us=HORIZONS_US)
    datasets[key] = df
    print(f"  {key}: {len(df):,} samples built in {time.time()-t0:.1f}s", flush=True)

print("\nStep 1: DEV statistics...", flush=True)
ledger = experiments.Ledger()
dev_stats = {}

t1 = time.time()
for sym, df in datasets.items():
    sub = df[df["day"].isin(DEV_DAYS)]
    for feature in []:  # FEATURE_NAMES
        for h in HORIZONS_US:
            pass  # simplified for testing

print(f"  DEV stats done in {time.time()-t1:.1f}s", flush=True)

print("\nAll basic steps completed!", flush=True)