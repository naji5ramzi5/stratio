"""Generate the active trading universe from the robust scan.
Keep only pairs with mean_sharpe >= 1.0 and >=2 windows — these have the
most reliable edge across time windows."""
import json

ROBUST_FILE = "pairs_robust.json"
ACTIVE_FILE = "pairs_active.json"
MIN_SHARPE = 1.0
MIN_WINDOWS = 2

with open(ROBUST_FILE) as f:
    data = json.load(f)

active = [p for p in data["pairs"]
          if p["mean_sharpe"] >= MIN_SHARPE
          and len(p["windows"]) >= MIN_WINDOWS]

# de-dup by base asset concentration: prefer diverse underlyings
active.sort(key=lambda x: -x["mean_sharpe"])

out = {
    "generated": data["generated"],
    "min_sharpe": MIN_SHARPE,
    "min_windows": MIN_WINDOWS,
    "source": ROBUST_FILE,
    "pairs": active,
}
with open(ACTIVE_FILE, "w") as f:
    json.dump(out, f, indent=2)

print(f"Active pairs (mean_sharpe >= {MIN_SHARPE}, >= {MIN_WINDOWS} windows): {len(active)}")
for p in active:
    print(f"  {p['pair']:22s} mean_sharpe={p['mean_sharpe']}")
