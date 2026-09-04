import sys
import os
import traceback

# Add crypto directory to path
crypto_dir = r'C:\Users\IRAQ SOFT\Desktop\stratocrypto'
if crypto_dir not in sys.path:
    sys.path.insert(0, crypto_dir)

# Import and run the research pipeline
try:
    from market_events.research.run_research import main
    result = main()
    print('Pipeline completed with result:', result)
except Exception as e:
    print('Error running pipeline:', e)
    traceback.print_exc()
    sys.exit(1)