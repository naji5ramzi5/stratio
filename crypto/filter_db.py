import json, numpy as np

with open("pairs_db.json") as f:
    db = json.load(f)

clean = []
for r in db["results"]:
    if r["sharpe"] < 100 and r["win_rate"] >= 0.4 and r["n_trades"] >= 5:
        clean.append(r)

best = {}
for r in clean:
    key = (r["pair"], r["tf"])
    if key not in best or r["sharpe"] > best[key]["sharpe"]:
        best[key] = r

final = sorted(best.values(), key=lambda x: -(x["win_rate"] * np.log1p(x["sharpe"]) * np.log1p(x["n_trades"])))

print(f"Clean robust opportunities: {len(final)}")
print()
print(f"{'pair':24s} {'tf':4s} {'sharpe':>8} {'trades':>6} {'win%':>6} {'ret%':>8}")
print("-" * 65)
for r in final:
    print(f"{r['pair']:24s} {r['tf']:4s} {r['sharpe']:8.2f} {r['n_trades']:6d} {r['win_rate']:5.0%} {r['total_ret_pct']:7.1f}%")

# save clean version
db["results"] = final
db["filtered_at"] = db.get("generated")
with open("pairs_db.json", "w") as f:
    json.dump(db, f, indent=2)
print(f"\nsaved {len(final)} clean opportunities")
