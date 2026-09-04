import sys
import os
# Add the crypto directory to path
crypto_dir = r'C:\Users\IRAQ SOFT\Desktop\stratocrypto'
if crypto_dir not in sys.path:
    sys.path.insert(0, crypto_dir)
    
# Now try the import
try:
    from market_events.research.features_trade import FEATURE_REGISTRY, TradeFeatureEngine
    print('OK - FEATURE_REGISTRY has', len(FEATURE_REGISTRY), 'entries')
except Exception as e:
    print('Error:', e)