import json

with open("pairs_active.json") as f:
    data = json.load(f)

clean = []
for p in data["pairs"]:
    total_trades = sum(n for _, n in p["windows"].values())
    if p["mean_sharpe"] < 100 and total_trades >= 3:
        clean.append(p)

clean.sort(key=lambda x: -x["mean_sharpe"])
print(f"Clean robust pairs: {len(clean)}")
for p in clean:
    trades = sum(n for _, n in p["windows"].values())
    print(f"  {p['pair']:24s} sharpe={p['mean_sharpe']:6.2f}  trades={trades}")

data["pairs"] = clean
with open("pairs_active.json", "w") as f:
    json.dump(data, f, indent=2)
print("saved")
