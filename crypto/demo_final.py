import json
from regime_predictor import regime_score_for_pair
from data_loader.binance_ohlcv import fetch_klines

with open("pairs_active.json") as f:
    pairs = json.load(f)["pairs"]

print("=== 15 Active Pairs — Live Regime Assessment ===\n")
trade = skip = 0
print("pair                      sharpe  score  regimes           action")
print("-" * 70)
for p in pairs:
    pair = p["pair"]
    a, b = pair.split("/")
    r = regime_score_for_pair(a, b, fetch_klines)
    if r:
        act = r["action_a"] if r["action_a"] == r["action_b"] else "MIXED"
        print(f"{pair:24s} {p['mean_sharpe']:6.2f}  {r['score']:5.1f}  {r['regime_a']:>10}+{r['regime_b']:>4}  {act}")
        if r["score"] >= 60:
            trade += 1
        else:
            skip += 1

print(f"\n{trade} pairs ready to TRADE now | {skip} pairs SKIP (regime unfavourable)")
print(f"Total universe: {len(pairs)} robust OOS-validated pairs")
