from pair_selector import PairLifecycleManager, PairMetrics, PairState, PairRecord

mgr = PairLifecycleManager()

metrics = PairMetrics(cointegration_pvalue=0.01, half_life=20.0, hedge_ratio_stability=0.1)
performance = {"sharpe": 10.0, "win_rate": 0.6, "n_trades": 25}

key = "TEST@1h"
mgr.pairs[key] = PairRecord(pair="TEST", timeframe="1h", state=PairState.DISCOVERED,
                              metrics=metrics, performance=performance)

print(f"Before: {mgr.pairs[key].state.value}")
score = mgr.compute_stability_score(metrics, performance)
print(f"Score: {score}")

mgr.update_pair(key, metrics, performance)
print(f"After 1st update: {mgr.pairs[key].state.value}")

mgr.update_pair(key, metrics, performance)
print(f"After 2nd update: {mgr.pairs[key].state.value}")
