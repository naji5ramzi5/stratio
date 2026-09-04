"""Future-return label engine (Phase 2, STEP 6-7).

SEPARATION OF CONCERNS (the core discipline):
- the FEATURE engine (features_trade.py) may only see events with ts <= T
- the LABEL engine is the ONLY component allowed to look past T: a label is
  the OUTCOME of the interval (T, T+H], so it is computed from future data
  by definition

Guarantees implemented and tested:
- label(T, H) uses only trades with ts <= T + H
- label(T, H) requires at least one trade STRICTLY inside (T, T+H] — if the
  same trade would serve both sides the label is invalid (NaN), never 0
- feature timestamps == label timestamps on the same grid (alignment test)
- net labels = gross label − round-trip costs (costs module)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .pit import PitView

HORIZONS_US: List[int] = [
    60_000_000,     # 60 s
    300_000_000,    # 300 s
    900_000_000,    # 900 s
    1_800_000_000,  # 1800 s
]


class LabelEngine:
    """Computes forward log-return labels on the sample grid."""

    def __init__(self, view: PitView, grid_us: np.ndarray):
        self.view = view
        self.grid = np.asarray(grid_us, dtype=np.int64)
        p_log = np.log(np.asarray(view.arrays["price"], dtype=np.float64))
        self._P_logp = np.concatenate(([0.0], np.cumsum(p_log)))
        self._idx1 = np.array([view.end_index(int(t)) for t in self.grid])

    def _logp_at(self, ts_targets: np.ndarray) -> np.ndarray:
        i1 = np.searchsorted(self.view.ts, ts_targets, side="right")
        valid = i1 > 0
        out = np.full(len(ts_targets), np.nan)
        out[valid] = self._P_logp[i1[valid] - 1]
        return out

    def gross_labels(self, horizon_us: int, cost_bps: Optional[float] = None) -> dict:
        """Forward log returns T -> T+H for every grid sample.

        Returns dict with arrays: label, start_price_log, end_price_log,
        valid (bool), n_trades_in_period (int).
        """
        idx0 = self._idx1  # feature side: events <= T
        targets = self.grid + horizon_us
        end_idx = np.searchsorted(self.view.ts, targets, side="right")
        valid = end_idx > idx0  # at least one trade strictly in (T, T+H]
        lp0 = self._logp_at(self.grid)
        lp1 = self._logp_at(targets)
        label = lp1 - lp0
        valid = valid & np.isfinite(label)
        n_trades = (end_idx - idx0).astype(np.int64)
        out = {
            "label": label,
            "start_price_log": lp0,
            "end_price_log": lp1,
            "valid": valid,
            "n_trades_in_period": n_trades,
        }
        if cost_bps is not None and cost_bps > 0:
            out["label_net"] = label - cost_bps / 1e4
        return out

    def labels(self, horizons_us: List[int], cost_bps: Optional[float] = None) -> dict:
        out: dict = {}
        for h in horizons_us:
            out[h] = self.gross_labels(h, cost_bps)
        return out


def label_row(
    symbol: str, T: int, horizon_us: int, label: float, valid: bool,
    label_net: Optional[float] = None,
) -> dict:
    row = {
        "symbol": symbol,
        "ts_us": int(T),
        "horizon_us": int(horizon_us),
        "label_gross": float(label),
        "label_valid": bool(valid),
    }
    if label_net is not None:
        row["label_net"] = float(label_net)
    return row
