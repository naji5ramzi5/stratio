"""One-off: backfill relative_bias for old prediction_history entries.

Old entries carry market_regime + atr_pct (recorded at creation time), so the
same rule the predictor uses can be re-applied honestly:
  FADE_PUMP  if regime in (bullish, strong_bullish) or atr_pct >= 4
  LEAN_LONG  if regime == neutral
  NEUTRAL    otherwise
Entries missing BOTH fields are left without bias. Backup written once.
"""
import json
import os
import shutil
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
p = os.path.join(BASE, "prediction_history.json")

with open(p, encoding="utf-8") as f:
    h = json.load(f)

bak = p + ".bak_bias"
if not os.path.exists(bak):
    shutil.copy2(p, bak)

n = 0
for x in h:
    if x.get("relative_bias"):
        continue
    has_reg = x.get("market_regime") is not None
    has_atr = x.get("atr_pct") is not None
    if not has_reg and not has_atr:
        continue
    try:
        atr = float(x.get("atr_pct") or 0)
    except Exception:
        atr = 0.0
    reg = x.get("market_regime")
    if (reg in ("bullish", "strong_bullish")) or atr >= 4:
        x["relative_bias"] = "FADE_PUMP"
    elif reg == "neutral":
        x["relative_bias"] = "LEAN_LONG"
    else:
        x["relative_bias"] = "NEUTRAL"
    n += 1

tmp = p + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(h, f, ensure_ascii=False, indent=2, default=str)
os.replace(tmp, p)
print("backfilled:", n, "total:", len(h))
print("bias distribution:", dict(Counter(x.get("relative_bias", "None") for x in h)))
