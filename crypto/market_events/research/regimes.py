"""Regime classification (Phase 2, STEP 12) and DEV/OOS split (STEP 15).

Regimes are day-level labels computed from the 60 s grid returns (trade
data, PIT-safe): a regime is a property of the PAST, assigned to every
sample T within the day.

- vol regime: day realized vol (annualized) vs DEV-day terciles ->
  {low, mid, high}
- directional regime: t-stat of 300 s returns over the day ->
  {trend_up, trend_down, ranging}  (|t| < 1.5 => ranging)

The split replicates the project's pre-registered V4E protocol:
    DEV = 2026-07-25 .. 2026-08-03 (10 days)
    OOS = 2026-08-04 .. 2026-08-09 (6 days, untouched until locked)
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np

DEV_START = date(2026, 7, 25)
DEV_END = date(2026, 8, 3)
OOS_START = date(2026, 8, 4)
OOS_END = date(2026, 8, 9)


def day_list(start: date, end: date) -> List[str]:
    out = []
    d = start
    while d <= end:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


DEV_DAYS = day_list(DEV_START, DEV_END)
OOS_DAYS = day_list(OOS_START, OOS_END)
ALL_DAYS = day_list(DEV_START, OOS_END)


def classify_day(day: str, grid_returns: np.ndarray,
                 vol_terciles: Dict[str, float]) -> Dict[str, str]:
    """day: 'YYYY-MM-DD'; grid_returns: 300 s returns for that day."""
    r = np.asarray(grid_returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return {"vol": "unknown", "direction": "unknown"}
    day_vol = float(np.std(r) * np.sqrt(86400.0 / 300.0 * 365.25))
    lo, hi = vol_terciles["low"], vol_terciles["high"]
    vol = "low" if day_vol <= lo else ("high" if day_vol >= hi else "mid")
    tstat = float(np.mean(r) / (np.std(r) / np.sqrt(len(r))))
    direction = "ranging"
    if abs(tstat) >= 1.5:
        direction = "trend_up" if tstat > 0 else "trend_down"
    return {"vol": vol, "direction": direction}


def vol_terciles_from_days(days: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Tercile boundaries over the DEV days' 300 s grid returns."""
    vols = []
    for day, r in days.items():
        r = r[np.isfinite(r)]
        if len(r) >= 10:
            vols.append(float(np.std(r) * np.sqrt(86400.0 / 300.0 * 365.25)))
    vols = np.array(vols)
    if len(vols) == 0:
        return {"low": float("inf"), "high": float("inf")}
    return {
        "low": float(np.quantile(vols, 1 / 3)),
        "high": float(np.quantile(vols, 2 / 3)),
    }
