"""Feature/label dataset builder (Phase 2, STEPS 2-7).

Loads each symbol ONCE from the Phase 1 event store (all days concatenated),
builds the 60 s sample grid, computes every feature series with strict PIT
windows, and every label series with the separate label engine.

Output: a pandas DataFrame (saved as CSV per symbol) with columns
    symbol, day, ts_us,
    <feature columns>, 
    label_gross_<H>, label_net_<H>, label_valid_<H>,
    vol_regime, direction_regime

Feature naming convention: <name>_<window in s> (or _<fast>_<slow> for
momentum). Each feature's provenance is documented in PHASE2_FEATURE_CATALOG.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from market_events.research.features_trade import FEATURE_REGISTRY, TradeFeatureEngine
from .labels import HORIZONS_US, LabelEngine
from .pit import PitView
from .regimes import ALL_DAYS, DEV_DAYS, classify_day, vol_terciles_from_days

GRID_US = 60_000_000


def _day_of_grid(grid: np.ndarray, day_of: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Day label for each grid point (UTC date of the sample)."""
    idx = np.searchsorted(ts, grid, side="right") - 1
    idx = np.clip(idx, 0, len(day_of) - 1)
    return day_of[idx]


def save_dataset(df: pd.DataFrame, out_dir: str, symbol: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{symbol}.csv")
    df.to_csv(path, index=False)
    return path


@dataclass
class SymbolDataset:
    store: EventStore
    market: str
    symbol: str
    days: List[str] = None
    grid_us: int = GRID_US

    def __post_init__(self):
        if self.days is None:
            self.days = ALL_DAYS

    def load_arrays(self) -> dict:
        ts_all, price_all, qty_all, bm_all, day_of = [], [], [], [], []
        for day in self.days:
            data = self.store.load_day(self.market, self.symbol, day)
            ts_all.append(data["ts_us"])
            price_all.append(data["price"])
            qty_all.append(data["qty"])
            bm_all.append(data["buyer_maker"])
            day_of.append(np.full(len(data["ts_us"]), day, dtype=object))
        return {
            "ts_us": np.concatenate(ts_all),
            "price": np.concatenate(price_all),
            "qty": np.concatenate(qty_all),
            "buyer_maker": np.concatenate(bm_all),
            "day_of": np.concatenate(day_of),
        }

    def build(self, cost_cfg=None, horizons_us: Optional[List[int]] = None) -> pd.DataFrame:
        arrays = self.load_arrays()
        view = PitView(
            arrays["ts_us"], price=arrays["price"], qty=arrays["qty"],
            buyer_maker=arrays["buyer_maker"].astype(bool),
        )
        first = int(arrays["ts_us"][0] // self.grid_us * self.grid_us)
        last = int(arrays["ts_us"][-1] // self.grid_us * self.grid_us) + self.grid_us
        grid = np.arange(first, last, self.grid_us, dtype=np.int64)

        fe = TradeFeatureEngine(view, grid)
        le = LabelEngine(view, grid)

        df = pd.DataFrame({
            "symbol": self.symbol,
            "day": _day_of_grid(grid, arrays["day_of"], arrays["ts_us"]),
            "ts_us": grid,
        })

        # Compute all registered features
        for name in FEATURE_REGISTRY:
            method_name, window_us = FEATURE_REGISTRY[name]
            if isinstance(window_us, tuple):
                # Special features like momentum (fast, slow)
                if name == "momentum_900_300s":
                    series = fe.f_momentum(*window_us)
                elif name == "acceleration_300s":
                    series = fe.f_acceleration(*window_us)
                elif name == "volume_intensity_60s":
                    series = fe.f_volume_intensity(*window_us)
                elif name == "trade_size_z_60s":
                    series = fe.f_trade_size_z(*window_us)
                else:
                    series = np.full(len(grid), np.nan)
            else:
                series = getattr(fe, method_name)(window_us)
            df[name] = series

        # Compute labels
        cost_bps = None  # Will be set per symbol in orchestrator
        for h in horizons_us or HORIZONS_US:
            g = le.gross_labels(h, cost_bps)
            df[f"label_gross_{int(h/1_000_000)}s"] = g["label"]
            df[f"label_valid_{int(h/1_000_000)}s"] = g["valid"]
            if cost_bps is not None:
                df[f"label_net_{int(h/1_000_000)}s"] = g["label_net"]

        self._add_regimes(df)
        return df

    def _add_regimes(self, df: pd.DataFrame) -> None:
        r300 = df["ret_300s"].to_numpy()
        day_returns: Dict[str, np.ndarray] = {}
        for day in df["day"].unique():
            day_returns[day] = r300[df["day"].to_numpy() == day]
        terciles = vol_terciles_from_days(
            {d: day_returns[d] for d in day_returns if d in DEV_DAYS})
        vols = df["day"].map(
            lambda d: classify_day(d, day_returns.get(d, np.array([])), terciles))
        df["vol_regime"] = vols.map(lambda r: r["vol"])
        df["direction_regime"] = vols.map(lambda r: r["direction"])


def build_dataset(store: EventStore, market: str, symbol: str,
                  days: List[str] = None,
                  horizons_us: Optional[List[int]] = None) -> pd.DataFrame:
    """Convenience function: build a dataset for one symbol."""
    ds = SymbolDataset(store, market, symbol, days, GRID_US)
    return ds.build(cost_cfg=None, horizons_us=HORIZONS_US)


def _day_of_grid(grid: np.ndarray, day_of: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Day label for each grid point (UTC date of the sample)."""
    idx = np.searchsorted(ts, grid, side="right") - 1
    idx = np.clip(idx, 0, len(day_of) - 1)
    return day_of[idx]


def save_dataset(df: pd.DataFrame, out_dir: str, symbol: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{symbol}.csv")
    df.to_csv(path, index=False)
    return path
FEATURE_NAMES = list(FEATURE_REGISTRY.keys())
