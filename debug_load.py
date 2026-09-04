#!/usr/bin/env python
import sys
import os
import time

crypto_dir = r'C:\Users\IRAQ SOFT\Desktop\stratocrypto'
sys.path.insert(0, crypto_dir)
sys.path.insert(0, os.path.join(crypto_dir, 'crypto', 'market_events', 'research'))

from market_events.store import EventStore, DEFAULT_ROOT
from market_events.research.dataset import SymbolDataset
from market_events.research import labels
from market_events import regimes

print('Loading data...')
store = EventStore(DEFAULT_ROOT)
horizons_us = labels.HORIZONS_US

symbols = [('futures_um', 'BTCUSDT'), ('futures_um', 'ETHUSDT'), ('spot', 'BTCUSDT')]

for market, sym in symbols:
    key = f'{market}:{sym}'
    print(f'Loading {key}...')
    first = store.load_day(market, sym, regimes.DEV_DAYS[0])
    cfg = type('Cfg', (), {})()
    cfg.taker_bps = 5
    ds = SymbolDataset(store, market, sym)
    t0 = time.time()
    df = ds.build(cost_cfg=cfg, horizons_us=horizons_us)
    t1 = time.time()
    print(f'  Built {key} in {t1-t0:.1f}s, shape: {df.shape}')
    print(f'  Columns: {list(df.columns)[:10]}...')
print('Done')