import json
from regime_predictor import regime_score_for_pair
from data_loader.binance_ohlcv import fetch_klines

with open("pairs_active.json") as f:
    data = json.load(f)

print("pair                      score regimeA         regimeB         action")
print("-" * 70)
for p in data["pairs"]:
    pair = p["pair"]
    a, b = pair.split("/")
    r = regime_score_for_pair(a, b, fetch_klines)
    if r:
        act = r["action_a"] if r["action_a"] == r["action_b"] else "MIXED"
        print(f"{pair:24s} {r['score']:5.1f} {r['regime_a']:>14} {r['regime_b']:>14} {act:>8}")
