#!/usr/bin/env python
"""Run the Phase 2 research pipeline."""
import sys
import os

# Add crypto directory to path
crypto_dir = r'C:\Users\IRAQ SOFT\Desktop\stratocrypto'
if crypto_dir not in sys.path:
    sys.path.insert(0, crypto_dir)

# Add the market_events research directory to path
research_dir = os.path.join(crypto_dir, 'market_events', 'research')
if research_dir not in sys.path:
    sys.path.insert(0, research_dir)

# Import and run the research pipeline
try:
    from run_research import main
    print("Running Phase 2 research pipeline...")
    result = main()
    print(f"Pipeline completed with result: {result}")
    
    # Check if results were generated
    results_path = os.path.join(crypto_dir, 'PHASE2_RESEARCH_RESULTS.json')
    if os.path.exists(results_path):
        print(f"Results saved to: {results_path}")
        # Print summary
        import json
        with open(results_path, 'r') as f:
            results = json.load(f)
        print(f"\nSummary:")
        if 'performance' in results:
            print(f"  Performance: {results['performance']}")
        if 'ledger' in results:
            print(f"  Ledger: {results['ledger']}")
        if 'multiple_testing' in results:
            mt = results['multiple_testing']
            print(f"  Multiple testing: {mt}")
    else:
        print(f"Results file not found at: {results_path}")
        
except Exception as e:
    print('Error running pipeline:', e)
    import traceback
    traceback.print_exc()
    sys.exit(1)