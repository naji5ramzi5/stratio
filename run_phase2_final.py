#!/usr/bin/env python
"""Run Phase 2 research pipeline to completion."""
import sys
import os

# Setup paths
crypto_dir = r'C:\Users\IRAQ SOFT\Desktop\stratocrypto'
research_dir = os.path.join(crypto_dir, 'market_events', 'research')

# Add to path
sys.path.insert(0, crypto_dir)
sys.path.insert(0, research_dir)

# Remove cached pyc files that might cause issues
pycache_dirs = [
    os.path.join(crypto_dir, '__pycache__'),
    os.path.join(crypto_dir, 'market_events', '__pycache__'),
    os.path.join(research_dir, '__pycache__')
]
for pc in pycache_dirs:
    if os.path.exists(pc):
        import shutil
        shutil.rmtree(pc)
        print(f"Removed {pc}")

try:
    from run_research import main
    print("=" * 60)
    print("PHASE 2 RESEARCH PIPELINE - STARTING")
    print("=" * 60)
    
    result = main()
    
    results_path = os.path.join(crypto_dir, 'PHASE2_RESEARCH_RESULTS.json')
    if os.path.exists(results_path):
        print("\n" + "=" * 60)
        print("PHASE 2 RESEARCH PIPELINE - COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"\nResults saved to: {results_path}")
        
        import json
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Print summary
        print("\n=== SUMMARY ===")
        for key in ['performance', 'ledger', 'multiple_testing', 'dev', 'oos', 'promising']:
            if key in results:
                val = results[key]
                if isinstance(val, dict):
                    print(f"  {key}: dict with {len(val)} entries")
                elif isinstance(val, list):
                    print(f"  {key}: list of {len(val)} items")
                else:
                    print(f"  {key}: {str(val)[:200]}")
            else:
                print(f"  {key}: NOT PRESENT")
        
        # Determine verdict
        print("\n=== VERDİCT ANALYSIS ===")
        if 'promising' in results and 'selected' in results['promising']:
            selected = results['promising']['selected']
            print(f"  Promising features selected: {len(selected)}")
            print(f"  Features: {selected}")
            
            # Check if OOS was also computed
            if 'oos' in results:
                print("  OOS statistics: COMPLETED")
            else:
                print("  OOS statistics: NOT COMPLETED (pipeline stopped)")
                
        if 'decay' in results:
            print("  Decay table: COMPLETED")
        else:
            print("  Decay table: NOT REACHED")
            
        if 'regimes' in results:
            print("  Regime analysis: COMPLETED")
        else:
            print("  Regime analysis: NOT REACHED")
            
    else:
        print(f"\nResults file not found at: {results_path}")
        # List any PHASE2 files
        for f in sorted(os.listdir(crypto_dir)):
            if 'PHASE2' in f:
                print(f"  Found: {f}")
                
except Exception as e:
    print("\n" + "=" * 60)
    print("PHASE 2 RESEARCH PIPELINE - FAILED")
    print("=" * 60)
    print(f"\nError: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)