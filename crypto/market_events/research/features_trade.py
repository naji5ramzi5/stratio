"""Trade-flow and market-state feature calculators (Phase 2).

Design (contract with Phase 1):
- features are computed from the Phase 1 event store via `PitView`
  (research/pit.py) — every sample at time T uses ONLY events with ts <= T
- every feature record carries: name, symbol, timestamp, value,
  calculator/version, source, window metadata (see FeatureRecord)
- deterministic: same data + same config => identical output (tested)
- stateful = windowed aggregates; all windows are closed at T (never beyond)

Vectorized implementation: per-sample windows are answered from cumulative
(prefix) arrays; the sample grid is explicit (array of T timestamps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .pit import PitView

GRID_US = 60_000_000  # default research grid: 60 s


@dataclass(frozen=True)
class FeatureRecord:
    """One feature sample with full provenance metadata."""

    name: str
    symbol: str
    timestamp_us: int
    value: float
    calculator_version: str
    source: str
    window_us: int

    def as_row(self) -> dict:
        return {
            "feature": self.name,
            "symbol": self.symbol,
            "ts_us": int(self.timestamp_us),
            "value": float(self.value),
            "calculator_version": self.calculator_version,
            "source": self.source,
            "window_us": int(self.window_us),
        }


CALCULATOR_VERSION = "phase2.v1"


# ---------------------------------------------------------------------------
# Feature registry: maps feature names to (method_name, window_us_or_tuple)
# The method_name corresponds to methods on TradeFeatureEngine (which already
# have the 'f_' prefix). The code in build() uses getattr(fe, method_name)
# (no extra "f_" added).
# ---------------------------------------------------------------------------

FEATURE_REGISTRY: dict = {
    "trade_count": ("f_trade_count", 60_000_000),
    "trade_volume": ("f_trade_volume", 60_000_000),
    "buy_volume": ("f_buy_volume", 60_000_000),
    "sell_volume": ("f_sell_volume", 60_000_000),
    "buy_sell_imbalance": ("f_buy_sell_imbalance", 60_000_000),
    "volume_delta": ("f_volume_delta", 60_000_000),
    "aggr_buy_intensity": ("f_aggr_buy_intensity", 60_000_000),
    "trade_size_mean": ("f_trade_size_mean", 60_000_000),
    "trade_size_std": ("f_trade_size_std", 60_000_000),
    "ret_60s": ("f_ret", 60_000_000),
    "ret_300s": ("f_ret", 300_000_000),
    "ret_900s": ("f_ret", 900_000_000),
    "realized_vol_300s": ("f_realized_vol", 300_000_000),
    "momentum_900_300s": ("f_momentum", (300_000_000, 900_000_000)),
    "acceleration_300s": ("f_acceleration", (300_000_000, 300_000_000)),
    "volume_intensity_60s": ("f_volume_intensity", (60_000_000, 360 * 60_000_000)),
    "trade_size_z_60s": ("f_trade_size_z", (60_000_000, 60 * 60_000_000)),
}

#: Public feature names list (must match FEATURE_REGISTRY keys)
FEATURE_NAMES: list = list(FEATURE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# TradeFeatureEngine
# ---------------------------------------------------------------------------

class TradeFeatureEngine:
    """Computes feature series on an explicit 60 s sample grid.

    For every sample time T the engine:
      1. derives the PIT end index from the FULL event array (asserted),
      2. computes each feature from prefix sums over [T - window, T].

    NaN means "not computable at T" (insufficient history / no trades) —
    values are never fabricated.
    """

    def __init__(self, view: PitView, grid_us: np.ndarray):
        self.view = view
        self.grid = np.asarray(grid_us, dtype=np.int64)
        if len(self.grid) == 0:
            raise ValueError("empty grid")
        if (self.grid[1:] < self.grid[:-1]).any():
            raise ValueError("grid must be sorted")
        self._idx1 = np.array([view.end_index(int(t)) for t in self.grid])
        # prefix sums (PIT-safe: prefixes are built over the whole day once,
        # windows are always [idx0, idx1] with idx1 = pit_end(T))
        p_price = np.log(np.asarray(view.arrays["price"], dtype=np.float64))
        self._P_logp = np.concatenate(([0.0], np.cumsum(p_price)))
        self._P_cnt = np.concatenate(([0.0], np.cumsum(np.ones(self.view.n))))
        self._P_qty = np.concatenate(([0.0], np.cumsum(view.arrays["qty"])))
        bm = view.arrays["buyer_maker"]
        q_buy = np.where(~bm, view.arrays["qty"], 0.0)  # aggressive buys lift ask
        q_sell = np.where(bm, view.arrays["qty"], 0.0)  # aggressive sells hit bid
        self._P_qbuy = np.concatenate(([0.0], np.cumsum(q_buy)))
        self._P_qsell = np.concatenate(([0.0], np.cumsum(q_sell)))
        self._P_qsq = np.concatenate(([0.0], np.cumsum(view.arrays["qty"] ** 2)))

    # -- window helpers -----------------------------------------------------

    def _idx0(self, window_us: int) -> np.ndarray:
        return np.searchsorted(self.view.ts, self.grid - window_us, side="left")

    def _window_sum(self, prefix: np.ndarray, window_us: int) -> np.ndarray:
        i0, i1 = self._idx0(window_us), self._idx1
        return prefix[i1] - prefix[i0]

    def _logp_at(self, j: np.ndarray) -> np.ndarray:
        """log last-trade-price at each sample T (PIT: ts <= T)."""
        valid = self._idx1[j] > 0
        out = np.full(len(j), np.nan)
        out[valid] = self._P_logp[self._idx1[j][valid] - 1]
        return out

    def _logp_at_bucket(self, ts_targets: np.ndarray) -> np.ndarray:
        i1 = np.searchsorted(self.view.ts, ts_targets, side="right")
        valid = i1 > 0
        out = np.full(len(ts_targets), np.nan)
        out[valid] = self._P_logp[i1[valid] - 1]
        return out

    # -- feature series (each returns np.ndarray of len(grid)) --------------

    def f_trade_count(self, window_us: int) -> np.ndarray:
        return self._window_sum(self._P_cnt, window_us)

    def f_trade_volume(self, window_us: int) -> np.ndarray:
        return self._window_sum(self._P_qty, window_us)

    def f_buy_volume(self, window_us: int) -> np.ndarray:
        return self._window_sum(self._P_qbuy, window_us)

    def f_sell_volume(self, window_us: int) -> np.ndarray:
        return self._window_sum(self._P_qsell, window_us)

    def f_buy_sell_imbalance(self, window_us: int) -> np.ndarray:
        b = self.f_buy_volume(window_us)
        s = self.f_sell_volume(window_us)
        denom = b + s
        out = np.full(len(b), np.nan)
        np.divide(b - s, denom, out=out, where=denom > 0)
        return out

    def f_volume_delta(self, window_us: int) -> np.ndarray:
        return self.f_buy_volume(window_us) - self.f_sell_volume(window_us)

    def f_aggr_buy_intensity(self, window_us: int) -> np.ndarray:
        b = self.f_buy_volume(window_us)
        denom = b + self.f_sell_volume(window_us)
        out = np.full(len(b), np.nan)
        np.divide(b, denom, out=out, where=denom > 0)
        return out

    def f_trade_size_mean(self, window_us: int) -> np.ndarray:
        cnt = self.f_trade_count(window_us)
        vol = self.f_trade_volume(window_us)
        out = np.full(len(cnt), np.nan)
        np.divide(vol, cnt, out=out, where=cnt > 0)
        return out

    def f_trade_size_std(self, window_us: int) -> np.ndarray:
        cnt = self.f_trade_count(window_us)
        vol = self.f_trade_volume(window_us)
        sq = self._window_sum(self._P_qsq, window_us)
        n = np.maximum(cnt, 1)
        var = np.maximum(0.0, (sq - vol**2 / n) / (n - 1))
        return np.sqrt(var)

    def f_ret(self, window_us: int) -> np.ndarray:
        """log return from last trade at T-window to last trade at T."""
        i0 = self._idx0(window_us)
        valid = (self._idx1 > 0) & (i0 > 0)
        out = np.full(len(self.grid), np.nan)
        out[valid] = (
            self._P_logp[self._idx1[valid] - 1] - self._P_logp[i0[valid] - 1]
        )
        return out

    def f_realized_vol(self, window_us: int) -> np.ndarray:
        """Annualized realized vol from the 60 s grid returns inside
        [T - window, T]. Units: annualized fraction (sqrt(86400/60) scaling
        already applied in f_grid_returns + sqrt(365.25*86400/window_sec))."""
        return _rolling_vol(self._grid_returns(), window_us // GRID_US)

    def _grid_returns(self) -> np.ndarray:
        lp = self._logp_at(np.arange(len(self.grid)))
        valid = ~np.isnan(lp)
        r = np.full(len(self.grid), np.nan)
        r[1:] = lp[1:] - lp[:-1]
        return r

    def f_momentum(self, fast_us: int, slow_us: int) -> np.ndarray:
        """slow-window return minus fast-window return (both closed at T)."""
        return self.f_ret(slow_us) - self.f_ret(fast_us)

    def f_acceleration(self, base_us: int, step_us: int) -> np.ndarray:
        """Change in base-window return between T-step and T."""
        r = self.f_ret(base_us)
        out = np.full(len(r), np.nan)
        out[1:] = r[1:] - r[:-1]
        return out

    def f_volume_intensity(self, window_us: int, ref_us: int) -> np.ndarray:
        """window volume / median(window volume) over the prior ref_us window
        (uses only samples with T' <= T - window_us; min ref samples = 60)."""
        v = self.f_trade_volume(window_us)
        k = max(1, ref_us // window_us)
        out = np.full(len(v), np.nan)
        for j in range(k, len(v)):
            hist = v[j - k:j]
            if np.isfinite(hist).sum() < max(k, 60):
                continue
            med = float(np.nanmedian(hist))
            if med > 0:
                out[j] = v[j] / med
        return out

    def f_trade_size_z(self, window_us: int, ref_us: int) -> np.ndarray:
        """z-score of mean trade size vs the prior ref_us window (liquidity
        change proxy). NaN where reference is degenerate."""
        m = self.f_trade_size_mean(window_us)
        k = max(1, ref_us // window_us)
        out = np.full(len(m), np.nan)
        for j in range(k, len(m)):
            hist = m[j - k:j]
            good = hist[np.isfinite(hist)]
            if len(good) < min(k, 30):
                continue
            std = float(np.std(good))
            if std > 0 and np.isfinite(m[j]):
                out[j] = (m[j] - float(np.mean(good))) / std
        return out


PIT_STRICT = True


def _rolling_vol(grid_returns: np.ndarray, k: int) -> np.ndarray:
    """annualized std of last k grid returns at each sample j (k >= 2)."""
    n = len(grid_returns)
    out = np.full(n, np.nan)
    if k < 2:
        return out
    from numpy.lib.stride_tricks import sliding_window_view

    for j in range(k - 1, n):
        window = grid_returns[j - k + 1 : j + 1]
        good = window[np.isfinite(window)]
        if len(good) < max(2, k // 2):
            continue
        out[j] = float(np.std(good) * np.sqrt(86400.0 / 60.0 * 365.25))
    return out


# ---------------------------------------------------------------------------
# Feature registry: maps feature names to (method_name, window_us_or_tuple)
# The method_name corresponds to methods on TradeFeatureEngine (which already
# have the 'f_' prefix). The code in build() uses getattr(fe, method_name)
# (no extra "f_" added).
# ---------------------------------------------------------------------------

FEATURE_REGISTRY: dict = {
    "trade_count": ("f_trade_count", 60_000_000),
    "trade_volume": ("f_trade_volume", 60_000_000),
    "buy_volume": ("f_buy_volume", 60_000_000),
    "sell_volume": ("f_sell_volume", 60_000_000),
    "buy_sell_imbalance": ("f_buy_sell_imbalance", 60_000_000),
    "volume_delta": ("f_volume_delta", 60_000_000),
    "aggr_buy_intensity": ("f_aggr_buy_intensity", 60_000_000),
    "trade_size_mean": ("f_trade_size_mean", 60_000_000),
    "trade_size_std": ("f_trade_size_std", 60_000_000),
    "ret_60s": ("f_ret", 60_000_000),
    "ret_300s": ("f_ret", 300_000_000),
    "ret_900s": ("f_ret", 900_000_000),
    "realized_vol_300s": ("f_realized_vol", 300_000_000),
    "momentum_900_300s": ("f_momentum", (300_000_000, 900_000_000)),
    "acceleration_300s": ("f_acceleration", (300_000_000, 300_000_000)),
    "volume_intensity_60s": ("f_volume_intensity", (60_000_000, 360 * 60_000_000)),
    "trade_size_z_60s": ("f_trade_size_z", (60_000_000, 60 * 60_000_000)),
}

#: Public feature names list (must match FEATURE_REGISTRY keys)
FEATURE_NAMES: list = list(FEATURE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# FeatureRecord
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureRecord:
    """One feature sample with full provenance metadata."""

    name: str
    symbol: str
    timestamp_us: int
    value: float
    calculator_version: str
    source: str
    window_us: int

    def as_row(self) -> dict:
        return {
            "feature": self.name,
            "symbol": self.symbol,
            "ts_us": int(self.timestamp_us),
            "value": float(self.value),
            "calculator_version": self.calculator_version,
            "source": self.source,
            "window_us": int(self.window_us),
        }


CALCULATOR_VERSION = "phase2.v1"


# ---------------------------------------------------------------------------
# TradeFeatureEngine methods (see class definition above)
# ---------------------------------------------------------------------------

# (The method definitions f_trade_count through f_trade_size_z are included
#  inline above; the registry references them by name.)