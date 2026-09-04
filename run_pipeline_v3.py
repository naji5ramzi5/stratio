#!/usr/bin/env python
import sys
import os
import json
import traceback

# Add paths
crypto_dir = r'C:\Users\IRAQ SOFT\Desktop\stratocrypto'
research_dir = os.path.join(crypto_dir, 'market_events', 'research')

# Remove old cache that might cause issues
import shutil
pycache = os.path.join(crypto_dir, 'market_events', '__pycache__')
if os.path.exists(pycache):
    shutil.rmtree(pycache)
research_pcache = os.path.join(research_dir, '__pycache__')
if os.path.exists(research_pcache):
    shutil.rmtree(research_pcache)

sys.path.insert(0, crypto_dir)
sys.path.insert(0, research_dir)

# Open log file
log_path = os.path.join(crypto_dir, 'pipeline_run.log')
old_stdout = sys.stdout
old_stderr = sys.stderr
log_file = open(log_path, 'w')
sys.stdout = log_file
sys.stderr = log_file

try:
    print("=" * 60)
    print("PHASE 2 RESEARCH PIPELINE START")
    print("=" * 60)
    print(f"Python path: {sys.path[:3]}")
    print(f"Crypto dir: {crypto_dir}")
    print(f"Research dir: {research_dir}")
    
    from run_research import main
    print("\n--- About to call main() ---")
    result = main()
    print(f"\n--- main() returned: {result} ---")
    
    results_path = os.path.join(crypto_dir, 'PHASE2_RESEARCH_RESULTS.json')
    if os.path.exists(results_path):
        print(f"\n--- Results found at: {results_path} ---")
        with open(results_path, 'r') as f:
            results = json.load(f)
        print(f"\nSummary of results:")
        for key in ['performance', 'ledger', 'multiple_testing', 'dev', 'oos', 'promising']:
            if key in results:
                val = results[key]
                if isinstance(val, dict):
                    print(f"  {key}: dict with {len(val)} keys")
                elif isinstance(val, list):
                    print(f"  {key}: list of {len(val)} items")
                else:
                    print(f"  {key}: {str(val)[:200]}")
    else:
        print(f"\n--- Results file NOT found at: {results_path} ---")
        # List files starting with PHASE2
        for f in sorted(os.listdir(crypto_dir)):
            if f.startswith('PHASE2'):
                print(f"  Found: {f} ({os.path.getsize(os.path.join(crypto_dir, f))} bytes)")
                
except Exception as e:
    print(f"\n--- CRASH: {type(e).__name__}: {e} ---")
    traceback.print_exc()
finally:
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    log_file.close()
    
print(f"\n--- Pipeline run log written to: {log_path} ---")