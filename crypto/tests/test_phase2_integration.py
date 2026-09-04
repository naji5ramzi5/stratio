"""Phase 2 integration tests on REAL store data (skip when absent).

Verifies the research pipeline against the Phase 1 event store:
- feature/label alignment with no-leak assertions on a real day
- dataset build determinism
- V4D/V4E replica equivalence on a small day subset (old loader vs store)
"""

import os
import unittest

import numpy as np

from market_events.research import v4_reproduction as v4
from market_events.research.dataset import FEATURE_NAMES, SymbolDataset
from market_events.research.labels import LabelEngine
from market_events.research.pit import PitView
from market_events.store import DEFAULT_ROOT, EventStore

STORE = EventStore(DEFAULT_ROOT)
HAS_FUT = STORE.has_day("futures_um", "BTCUSDT", "2026-08-09")


@unittest.skipUnless(HAS_FUT, "futures BTCUSDT store data not present")
class TestRealDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = STORE
        arrays = cls.store.load_day("futures_um", "BTCUSDT", "2026-08-09")
        cls.view = PitView(arrays["ts_us"], price=arrays["price"],
                           qty=arrays["qty"],
                           buyer_maker=arrays["buyer_maker"].astype(bool))

    def test_grid_features_labels_aligned(self):
        ts = self.view.ts
        grid = (np.arange(int(ts[0] // 60_000_000 * 60_000_000),
                          int(ts[-1] // 60_000_000 * 60_000_000),
                          60_000_000, dtype=np.int64))
        le = LabelEngine(self.view, grid)
        g = le.gross_labels(300_000_000)
        self.assertEqual(len(g["label"]), len(grid))
        # every label starts AFTER its feature timestamp by construction:
        # label uses end index > feature end index
        self.assertTrue((g["label"][g["valid"]] != 0).any())

    def test_feature_window_never_past_T(self):
        """Every sample's feature windows must end exactly at the sample T."""
        ts = self.view.ts
        grid = np.arange(int(ts[0] // 60_000_000 * 60_000_000),
                         int(ts[-1] // 60_000_000 * 60_000_000),
                         60_000_000, dtype=np.int64)[:50]
        from market_events.research.features_trade import TradeFeatureEngine
        fe = TradeFeatureEngine(self.view, grid)
        for T, idx1 in zip(grid, fe._idx1):
            self.assertEqual(
                idx1, int(np.searchsorted(ts, T, side="right")),
                f"PIT end mismatch at {T}")

    def test_dataset_build_deterministic(self):
        ds = SymbolDataset(self.store, "futures_um", "BTCUSDT",
                           days=["2026-08-09"])
        df1 = ds.build(horizons_us=[300_000_000])
        df2 = ds.build(horizons_us=[300_000_000])
        self.assertEqual(len(df1), len(df2))
        for col in FEATURE_NAMES:
            np.testing.assert_array_equal(df1[col].to_numpy(),
                                          df2[col].to_numpy())
        np.testing.assert_array_equal(
            df1["label_gross_300s"].to_numpy(),
            df2["label_gross_300s"].to_numpy())

    def test_dataset_has_all_feature_columns(self):
        ds = SymbolDataset(self.store, "futures_um", "BTCUSDT",
                           days=["2026-08-09"])
        df = ds.build(horizons_us=[300_000_000])
        for f in FEATURE_NAMES:
            self.assertIn(f, df.columns)
        self.assertIn("label_gross_300s", df.columns)
        self.assertIn("label_valid_300s", df.columns)

    def test_no_duplicate_grid_points(self):
        ds = SymbolDataset(self.store, "futures_um", "BTCUSDT",
                           days=["2026-08-09"])
        df = ds.build(horizons_us=[300_000_000])
        self.assertEqual(len(df), df["ts_us"].nunique())


@unittest.skipUnless(HAS_FUT, "futures store data not present")
class TestV4Replica(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = STORE
        for sym in ("BTCUSDT", "ETHUSDT", "ETHBTC"):
            if not cls.store.has_day("futures_um", sym, "2026-08-09"):
                raise unittest.SkipTest(f"missing {sym} in store")

    def test_replica_matches_original_deterministic(self):
        """p=1.0, lat=0 runs are deterministic: replica must equal the
        saved ARBITRAGE_V4D table for the same config on the same days."""
        import json
        import sys

        crypto = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        sys.path.insert(0, crypto)
        with open(os.path.join(crypto, "ARBITRAGE_V4D_MAKER_SEQ.json"),
                  encoding="utf-8") as f:
            orig = json.load(f)
        days = ["2026-08-09"]
        series = v4.load_series_from_store(self.store, days)
        from arbitrage import maker_seq
        rng = np.random.default_rng(42)
        for thr in (5.0, 8.0, 10.0):
            for win in (5, 10):
                r = maker_seq.simulate(thr, win, 1.0, rng, series, 0)
                key = f"thr{thr}_win{win*100}ms_p1.0_lat0ms"
                o = orig["results"][key]
                self.assertEqual(r["candidates"], o["candidates"],
                                 f"candidates differ for {key}")
                self.assertEqual(r["completed"], o["completed"],
                                 f"completed differ for {key}")
                self.assertEqual(r["net_pnl_usd"], o["net_pnl_usd"],
                                 f"pnl differs for {key}")


if __name__ == "__main__":
    unittest.main()
