import json
from pair_selector import PairLifecycleManager, PairMetrics, PairState

mgr = PairLifecycleManager()
mgr.initialize_from_db()

# Debug first 5 pairs
for key, rec in list(mgr.pairs.items())[:5]:
    score = mgr.compute_stability_score(rec.metrics, rec.performance)
    p = rec.performance
    print(f"{key}: sharpe={p.get('sharpe')}, wr={p.get('win_rate')}, trades={p.get('n_trades')} -> score={score:.0f}")

# Count states
from collections import Counter
states = Counter(r.state.value for r in mgr.pairs.values())
print(f"\nStates: {dict(states)}")
