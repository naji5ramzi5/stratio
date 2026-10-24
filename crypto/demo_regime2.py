import json
from regime_predictor import regime_score_for_pair, predict_regime
from data_loader.binance_ohlcv import fetch_klines

print("=== Regime-Aware Pairs Universe (LIVE) ===\n")

btc = predict_regime(fetch_klines("BTCUSDT", "1h", 200))
print(f"BTC regime: {btc['regime']}  score={btc['score']}  action={btc['action']}\n")

with open("pairs_active.json") as f:
    pairs = json.load(f)["pairs"]

trade = skip = 0
print("pair                      score regimes           action")
print("-" * 60)
for p in pairs:
    pair = p["pair"]
    a, b = pair.split("/")
    r = regime_score_for_pair(a, b, fetch_klines)
    if r:
        act = r["action_a"] if r["action_a"] == r["action_b"] else "MIXED"
        print(f"{pair:24s} {r['score']:5.1f}  {r['regime_a']:>10}+{r['regime_b']:>4}  {act}")
        if r["score"] >= 60:
            trade += 1
        else:
            skip += 1
print(f"\nTRADE: {trade} pairs | SKIP: {skip} pairs (regime unfavourable)")
print(f"These {trade} pairs will receive capital when stretched.")
