#!/usr/bin/env python
import sys
import os
import json
import traceback

crypto_dir = r'C:\Users\IRAQ SOFT\Desktop\stratocrypto'
if crypto_dir not in sys.path:
    sys.path.insert(0, crypto_dir)

research_dir = os.path.join(crypto_dir, 'market_events', 'research')
if research_dir not in sys.path:
    sys.path.insert(0, research_dir)

os.makedirs(crypto_dir, exist_ok=True)

try:
    from run_research import main
    print("Running Phase 2 research pipeline...")
    result = main()
    print(f"Pipeline completed with result: {result}")
    
    results_path = os.path.join(crypto_dir, 'PHASE2_RESEARCH_RESULTS.json')
    if os.path.exists(results_path):
        print(f"Results saved to: {results_path}")
        with open(results_path, 'r') as f:
            results = json.load(f)
        print(f"\nSummary:")
        for key in ['performance', 'ledger', 'multiple_testing', 'dev', 'oos', 'promising']:
            if key in results:
                print(f"  {key}: {json.dumps(results[key], indent=2)[:500]}")
    else:
        print(f"Results file not found at: {results_path}")
        # Try to find what was generated
        import glob
        for f in glob.glob(os.path.join(crypto_dir, 'PHASE2*')):
            print(f"  Found: {f}")
            
except Exception as e:
    print('Error running pipeline:', e)
    traceback.print_exc()
    # Save error details
    error_path = os.path.join(crypto_dir, 'pipeline_error.txt')
    with open(error_path, 'w') as f:
        traceback.print_exc(file=f)
    sys.exit(1)