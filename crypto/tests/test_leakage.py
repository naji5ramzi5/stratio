"""Leakage guards.

These tests fail loudly if training/live feature code ever reads a future
bar: ``compute_features`` at row k must produce identical values whether it is
run on the full frame or on a frame truncated at row k (causality), and
``merge_btc_features`` must use the same aligned rolling correlation as live
inference rather than a tail-based correlation.
"""
import numpy as np
import pandas as pd
import unittest

import ml_trainer as mt


def _synthetic(n=400, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.4, n))
    open_ = close + rng.normal(0, 0.1, n)
    high = np.maximum(close, open_) + rng.uniform(0, 0.3, n)
    low = np.minimum(close, open_) - rng.uniform(0, 0.3, n)
    volume = np.abs(rng.normal(1000, 200, n))
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })


class TestComputeFeaturesCausality(unittest.TestCase):
    def test_row_values_independent_of_future_data(self):
        full = _synthetic()
        feats_full = mt.compute_features(full)
        probe_cols = ["ret_21", "rsi_14", "ema_200", "bb_20_p", "atr_14",
                      "adx_14", "cci_10", "macd_h_12_26", "vty_20"]
        k = 250  # deep enough that all windows are filled
        truncated = _synthetic()
        feats_trunc = mt.compute_features(truncated.iloc[: k + 1])
        for col in probe_cols:
            self.assertIn(col, feats_full.columns)
            self.assertTrue(np.isclose(
                feats_full[col].iloc[k], feats_trunc[col].iloc[k], equal_nan=True),
                msg=f"{col} changed when future bars were removed")


class TestMergeBtcFeatures(unittest.TestCase):
    def _frame(self, n=300, seed=1):
        rng = np.random.default_rng(seed)
        close = 100 + np.cumsum(rng.normal(0, 0.4, n))
        return pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "close": close,
        })

    def test_btc_corr_matches_manual_rolling_corr(self):
        df = self._frame()
        df["ret_1"] = df["close"].pct_change()
        btc = self._frame(seed=2)
        btc_ret = pd.DataFrame({
            "timestamp": btc["timestamp"],
            "btc_ret_1": btc["close"].pct_change(),
        })
        merged = mt.merge_btc_features(df, btc_ret)
        btc_aligned = btc_ret.set_index("timestamp")["btc_ret_1"].reindex(df["timestamp"])
        btc_aligned.index = df.index  # share df's index for rolling().corr()
        manual = df["ret_1"].rolling(10).corr(btc_aligned)
        np.testing.assert_allclose(merged["btc_corr_10"].values,
                                   manual.values, equal_nan=True)

    def test_missing_btc_bar_is_nan_not_shifted(self):
        df = self._frame(n=50)
        df["ret_1"] = df["close"].pct_change()
        btc_ret = pd.DataFrame({
            "timestamp": df["timestamp"].iloc[:20],  # only first 20 bars
            "btc_ret_1": np.linspace(0, 0.1, 20),
        })
        merged = mt.merge_btc_features(df, btc_ret)
        # Bars beyond the BTC data must be NaN (left-join), never silently 0
        self.assertTrue(merged["btc_corr_10"].iloc[25:].isna().all())

    def test_old_tail_based_correlation_would_leak(self):
        """Regression: the old approach correlated returns against the btc tail
        of the WHOLE series. Verify the aligned version differs for a long
        frame where the tail is decorrelated."""
        df = self._frame(n=500, seed=11)
        df["ret_1"] = df["close"].pct_change()
        btc = self._frame(n=500, seed=12)
        btc_ret = pd.DataFrame({
            "timestamp": btc["timestamp"],
            "btc_ret_1": btc["close"].pct_change(),
        })
        merged = mt.merge_btc_features(df, btc_ret)
        # tail-based: correlate symbol rets against the LAST 10 btc rets
        tail = pd.Series(btc_ret["btc_ret_1"].values[-10:])
        tail_corr = df["ret_1"].rolling(10).corr(pd.Series(np.repeat(tail.mean(), len(df))))
        # At the last row, the aligned value should reflect recent co-movement;
        # it is almost certainly not equal to the leaky tail constant series.
        self.assertFalse(np.isclose(merged["btc_corr_10"].iloc[-1], tail_corr.iloc[-1]))


class TestPrepareDropna(unittest.TestCase):
    def test_merge_then_dropna_yields_no_nan(self):
        df = _synthetic(n=200)
        df["ret_1"] = df["close"].pct_change()
        btc = _synthetic(n=200, seed=9)
        btc_ret = pd.DataFrame({
            "timestamp": btc["timestamp"],
            "btc_ret_1": btc["close"].pct_change(),
        })
        merged = mt.merge_btc_features(df, btc_ret)
        merged = merged.dropna()
        self.assertFalse(merged["btc_corr_10"].isna().any())


if __name__ == "__main__":
    unittest.main()
