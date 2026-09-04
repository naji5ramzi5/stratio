import sys
import os

# Get the crypto directory - use multiple possible locations
possible_paths = [
    r'C:\Users\IRAQ SOFT\Desktop\stratocrypto',
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
]

crypto_dir = None
for p in possible_paths:
    if os.path.isdir(p) and os.path.isdir(os.path.join(p, 'market_events')):
        crypto_dir = p
        break

if crypto_dir:
    sys.path.insert(0, crypto_dir)
    try:
        from market_events.research.features_trade import FEATURE_REGISTRY, TradeFeatureEngine
        print('OK - FEATURE_REGISTRY has', len(FEATURE_REGISTRY), 'entries')
    except Exception as e:
        print('Error importing:', e)
else:
    print('Could not find crypto directory')