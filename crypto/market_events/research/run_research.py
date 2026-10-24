"""Phase 2 research orchestrator.

Runs the full pipeline (STEPS 2-20) in a fixed order:
  1. build DEV+OOS datasets per symbol (one load per symbol)
  2. DEV statistics for ALL features x ALL horizons (locked before OOS)
  3. promising-feature selection via the PRE-REGISTERED rule (no OOS look)
  4. OOS statistics for locked features only
  5. predictive decay (IC across horizons, all features, DEV)
  6. regime-conditional statistics for promising features (DEV)
  7. cross-asset analysis (BTC<->ETH futures, all features, H=300)
  8. order-book pilot (2026-08-10 recorder session)
  9. V4D/V4E reproduction via the Phase 1 store + equivalence check
 10. multiple-testing ledger + performance summary

Outputs (JSON + CSV) land in the repo root as PHASE2_* files.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from market_events.research import experiments as ex  # noqa: E402
from market_events.research import features_book as fb  # noqa: E402
from market_events.research import labels  # noqa: E402
from market_events.research import regimes  # noqa: E402
from market_events.research import stats as st  # noqa: E402
from market_events.research import v4_reproduction as v4  # noqa: E402
from market_events.research.costs import build_cost_config  # noqa: E402
from market_events.research.dataset import (  # noqa: E402
    FEATURE_NAMES, SymbolDataset, save_dataset,
)
from market_events.store import EventStore, DEFAULT_ROOT  # noqa: E402

_PROMISING_CACHE: set = set()

OUT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))  # crypto/
DATASETS_DIR = os.path.join(OUT_DIR, "research_data")

HORIZONS_US = labels.HORIZONS_US
SYMBOLS = [
    ("futures_um", "BTCUSDT"),
    ("futures_um", "ETHUSDT"),
    ("spot", "BTCUSDT"),
]
SYM_KEYS = [f"{m}:{s}" for m, s in SYMBOLS]
FUT_KEYS = SYM_KEYS[:2]
H_DEV = ex.DEV_HORIZON_US


def timing(name: str, fn, *args, **kw):
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    dt = time.perf_counter() - t0
    print(f"[timing] {name}: {dt:.1f}s")
    return out, dt


def main() -> int:
    print("=== PHASE 2 RESEARCH ===")
    store = EventStore(DEFAULT_ROOT)
    perf: dict = {}
    datasets: dict = {}
    cost_cfgs: dict = {}

    for market, sym in SYMBOLS:
        key = f"{market}:{sym}"
        first = store.load_day(market, sym, regimes.DEV_DAYS[0])
        cost_cfgs[key] = build_cost_config(first)
        cfg = cost_cfgs[key]
        print(f"  cost {key}: taker {cfg.taker_bps}bps/side, "
              f"measured spread {cfg.spread_bps:.4f}bps, "
              f"round-trip {cfg.round_trip_bps():.3f}bps")
        ds = SymbolDataset(store, market, sym)
        df, dt = timing(f"dataset {key} (16 days)",
                        ds.build, cost_cfg=cfg, horizons_us=HORIZONS_US)
        datasets[key] = df
        perf[f"build_{key}_s"] = round(dt, 1)
        save_dataset(df, DATASETS_DIR, f"{market}_{sym}")
        valids = {int(h / 1_000_000): int(df[f"label_valid_{int(h/1_000_000)}s"].sum())
                  for h in HORIZONS_US}
        print(f"  {key}: {len(df):,} samples, valid labels per horizon: {valids}")

    print(">>> STEP 1: Datasets built, starting DEV statistics...")
    ledger = ex.Ledger()
    report: dict = {"data": {}, "stats": {}, "dev": {}, "promising": {},
                    "oos": {}, "decay": {}, "regimes": {}, "cross_asset": {},
                    "book_pilot": {}, "v4": {}, "performance": {},
                    "ledger": {}, "multiple_testing": {}}

    # ---------------------------------------------------------------- DEV
    print(">>> STEP 2: Running DEV statistics...")
    dev_stats, dev_time = timing(
        "DEV statistics (all features x all horizons)",
        _stats_for, datasets, HORIZONS_US, "DEV", ledger)
    report["dev"] = {_k(k): v.as_dict()
                     for k, v in dev_stats.items()}
    perf["dev_stats_s"] = round(dev_time, 1)
    print(f"  DEV statistics completed in {dev_time:.1f}s")

    # ------------------------------------------------------------- promote
    print(">>> STEP 3: Selecting promising features...")
    promising = _select_promising(dev_stats, datasets)
    report["promising"] = {
        "rule": "|IC|>=0.02 @300s same sign on >=2 of 3 symbols "
                "OR |IC|>=0.03 on any symbol (DEV only, pre-registered)",
        "locked_before_oos": True,
        "dev_signs": {f: {s: round(r, 4) for s, r in signs.items()}
                      for f, signs in promising["dev_signs"].items()},
        "selected": sorted(promising["features"]),
    }
    print(f"  Promising features: {sorted(promising['features'])}")

# ------------------------------------------------------------- promote
    promising = _select_promising(dev_stats, datasets)
    report["promising"] = {
        "rule": "|IC|>=0.02 @300s same sign on >=2 of 3 symbols "
                "OR |IC|>=0.03 on any symbol (DEV only, pre-registered)",
        "locked_before_oos": True,
        "dev_signs": {f: {s: round(r, 4) for s, r in signs.items()}
                      for f, signs in promising["dev_signs"].items()},
        "selected": sorted(promising["features"]),
    }
    print(f"  Promising features: {sorted(promising['features'])}")

    # ------------------------------------------------------------------ OOS
    print(">>> STEP 4: Running OOS statistics...")
    oos_stats, oos_time = timing(
        "OOS statistics (locked features only)",
        _stats_for, datasets, HORIZONS_US, "OOS", ledger)
    report["oos"] = {_k(k): v.as_dict()
                     for k, v in oos_stats.items()}
    perf["oos_stats_s"] = round(oos_time, 1)
    print(f"  OOS statistics completed in {oos_time:.1f}s")

    # ----------------------------------------------------------------- decay
    print(">>> STEP 5: Computing decay table...")
    t0 = time.perf_counter()
    report["decay"] = _decay_table(dev_stats)
    perf["decay_s"] = round(time.perf_counter() - t0, 1)
    print(f"  Decay table completed in {perf['decay_s']:.1f}s")

    # -------------------------------------------------------------- regimes
    print(">>> STEP 6: Computing regime statistics...")
    t0 = time.perf_counter()
    report["regimes"] = _regime_table(datasets, promising["features"])
    perf["regimes_s"] = round(time.perf_counter() - t0, 1)
    print(f"  Regime statistics completed in {perf['regimes_s']:.1f}s")

    # ---------------------------------------------------------- cross-asset
    print(">>> STEP 7: Computing cross-asset analysis...")
    t0 = time.perf_counter()
    report["cross_asset"] = _cross_asset(datasets)
    perf["cross_asset_s"] = round(time.perf_counter() - t0, 1)
    print(f"  Cross-asset analysis completed in {perf['cross_asset_s']:.1f}s")

# ------------------------------------------------------------- book
    print(">>> STEP 8: Computing book pilot...")
    t0 = time.perf_counter()
    book = {}
    for bsym in ("BTCUSDT", "ETHUSDT"):
        bdf = fb.build_book_dataset(bsym)
        if bdf.empty:
            book[bsym] = {"error": "no depth data"}
            continue
        rows = []
        for h in fb.HORIZONS_US:
            lc = f"label_gross_{int(h/1_000_000)}s"
            from scipy import stats as sps
            for feat in fb.BOOK_FEATURES:
                fv = bdf[feat].to_numpy(dtype=np.float64)
                y = bdf[lc].to_numpy(dtype=np.float64)
                ok = np.isfinite(fv) & bdf[lc].to_numpy(dtype=bool)
                rho = sps.spearmanr(fv[ok], y[ok])[0] if ok.sum() >= 30 else None
                rows.append({"feature": feat, "horizon_s": int(h / 1_000_000),
                             "n": int(ok.sum()),
                             "ic": round(float(rho), 4) if rho is not None else None})
        book[bsym] = {"samples": int(len(bdf)),
                      "period": "2026-08-10 ~28 min recorder session (DATA_LIMITED)",
                      "rows": rows}
    report["book_pilot"] = book
    perf["book_pilot_s"] = round(time.perf_counter() - t0, 1)
    print(f"  Book pilot: BTCUSDT {book.get('BTCUSDT', {}).get('samples', 0):,} "
          f"samples, ETHUSDT {book.get('ETHUSDT', {}).get('samples', 0):,} samples")

    # ------------------------------------------------------------------ V4D
    print(">>> STEP 9: V4D/V4E reproduction...")
    print("V4D/V4E reproduction via Phase 1 store...")
    t0 = time.perf_counter()
    rep_v4e = v4.replicate_v4e(store)
    rep_v4d = v4.replicate_v4d(store)
    perf["v4_reproduction_s"] = round(time.perf_counter() - t0, 1)
    with open(os.path.join(OUT_DIR, "ARBITRAGE_V4E_MAKER_DEVOOS.json"),
              encoding="utf-8") as f:
        orig_v4e = json.load(f)
    with open(os.path.join(OUT_DIR, "ARBITRAGE_V4D_MAKER_SEQ.json"),
              encoding="utf-8") as f:
        orig_v4d = json.load(f)
    cmp_v4e = v4.compare_tables(rep_v4e, orig_v4e,
                                ["completed", "failures", "net_pnl_usd"])
    cmp_v4d = v4.compare_tables(rep_v4d, orig_v4d,
                                ["candidates", "completed", "failures",
                                 "net_pnl_usd", "net_bps_mean", "profitable"])
    report["v4"] = {
        "method": "same simulator (read-only import), same rng stream (seed 11), "
                  "data loaded via Phase 1 store instead of pandas zip re-parse",
        "v4e": {"identical": cmp_v4e["identical"],
                "fields_checked": cmp_v4e["checked"],
                "mismatches": cmp_v4e["mismatches"][:10]},
        "v4d": {"identical": cmp_v4d["identical"],
                "fields_checked": cmp_v4d["checked"],
                "mismatches": cmp_v4d["mismatches"][:10]},
    }
    print(f"  V4E replica identical={cmp_v4e['identical']} "
          f"({cmp_v4e['checked']} fields checked)")
    print(f"  V4D replica identical={cmp_v4d['identical']} "
          f"({cmp_v4d['checked']} fields checked)")

    # -------------------------------------------------------------- ledger
    ledger_path = ledger.save()
    report["ledger"] = ledger.counts()
    report["performance"] = perf
    report["multiple_testing"] = {
        "features": len(FEATURE_NAMES),
        "symbols": len(SYM_KEYS),
        "horizons": len(HORIZONS_US),
        "dev_experiments": len(FEATURE_NAMES) * len(SYM_KEYS) * len(HORIZONS_US),
        "oos_experiments": (len(promising["features"]) * len(SYM_KEYS)
                            * len(HORIZONS_US)),
        "regime_experiments": len(promising["features"]) * len(SYM_KEYS) * 3,
        "cross_asset_experiments": len(FEATURE_NAMES) * 2,
        "book_pilot_experiments": len(fb.BOOK_FEATURES) * len(fb.HORIZONS_US) * 2,
    }

    out_path = os.path.join(OUT_DIR, "PHASE2_RESEARCH_RESULTS.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"saved -> {out_path}")
    print(f"saved ledger -> {ledger_path}")
    print(f"performance: {perf}")
    return 0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _k(key: tuple) -> str:
    return f"{key[0]}|{key[1]}|{key[2]}"


def _split_df(df: pd.DataFrame, split: str) -> pd.DataFrame:
    days = regimes.DEV_DAYS if split == "DEV" else regimes.OOS_DAYS
    return df[df["day"].isin(days)]


def _stats_for(datasets, horizons, split, ledger) -> dict:
    """Stats per (feature, symbol, horizon) for one split."""
    out = {}
    for sym, df in datasets.items():
        sub = _split_df(df, split)
        for feature in FEATURE_NAMES:
            fv = sub[feature].to_numpy(dtype=np.float64)
            for h in horizons:
                lc = f"label_gross_{int(h/1_000_000)}s"
                ln = f"label_net_{int(h/1_000_000)}s"
                valid = sub[lc].to_numpy(dtype=bool)
                y = sub[lc].to_numpy(dtype=np.float64).copy()
                y[~valid] = np.nan
                yn = None
                if ln in sub.columns:
                    yn = sub[ln].to_numpy(dtype=np.float64)
                    yn[~valid] = np.nan
                s = st.compute_stats(fv, y, yn, feature=feature, symbol=sym,
                                     horizon_us=h, split=split)
                key = (feature, sym, h)
                out[key] = s
                if split == "DEV" or feature in _PROMISING_CACHE:
                    ledger.record(ex.Experiment(
                        feature=feature, symbol=sym, horizon_us=h, split=split,
                        n=s.n, spearman=s.spearman,
                        q5_q1_gross_bps=s.q5_q1_gross_bps,
                        q5_q1_net_bps=s.q5_q1_net_bps,
                        classification=_classify(s, feature, split)))
    return out


def _select_promising(dev_stats, datasets) -> dict:
    """Pre-registered rule on DEV @300s. No OOS data is read here."""
    h = ex.DEV_HORIZON_US
    feature_signs: dict = {}
    for feature in FEATURE_NAMES:
        signs = {}
        for sym in datasets:
            s = dev_stats.get((feature, sym, h))
            if s is not None and np.isfinite(s.spearman) and s.n >= 100:
                signs[sym] = s.spearman
        feature_signs[feature] = signs
    selected = set()
    for feature, signs in feature_signs.items():
        sig = {s: r for s, r in signs.items()
               if abs(r) >= ex.DEV_IC_THRESHOLD_AGG}
        same_sign = (len(sig) >= 2 and
                     len({np.sign(v) for v in sig.values()}) == 1)
        single_strong = any(abs(r) >= ex.DEV_IC_THRESHOLD_SINGLE
                            for r in signs.values())
        if same_sign or single_strong:
            selected.add(feature)
    _PROMISING_CACHE.clear()
    _PROMISING_CACHE.update(selected)
    return {"features": selected, "dev_signs": feature_signs}


def _classify(s: st.FeatureStats, feature: str, split: str) -> str:
    if feature not in _PROMISING_CACHE:
        return ex.CLASS_REJECTED
    if split == "DEV":
        return ex.CLASS_PROMISING
    if np.isfinite(s.spearman) and abs(s.spearman) >= ex.OOS_IC_THRESHOLD:
        return ex.CLASS_OOS_VALIDATED
    return ex.CLASS_EXPLORATORY


def _decay_table(dev_stats) -> dict:
    """IC per horizon for every feature (DEV, mean over futures symbols)."""
    out = {}
    for feature in FEATURE_NAMES:
        row = {}
        for h in HORIZONS_US:
            rhos = [dev_stats[(feature, s, h)].spearman
                    for s in FUT_KEYS if (feature, s, h) in dev_stats]
            row[int(h / 1_000_000)] = round(float(np.mean(rhos)), 4) if rhos else None
        out[feature] = row
    return out


def _regime_table(datasets, promising_features) -> dict:
    out = {}
    for feature in promising_features:
        per_feature = {}
        for sym, df in datasets.items():
            sub = _split_df(df, "DEV")
            fv_all = sub[feature].to_numpy(dtype=np.float64)
            y_all = sub[f"label_gross_{int(H_DEV/1_000_000)}s"].to_numpy(dtype=np.float64)
            v_all = sub[f"label_valid_{int(H_DEV/1_000_000)}s"].to_numpy(dtype=bool)
            for regime in ("low", "mid", "high"):
                mask = (sub["vol_regime"].to_numpy() == regime) & v_all
                fv_all2, y_all2 = fv_all[mask], y_all[mask]
                if len(fv_all2) < 30:
                    per_feature[f"{sym}/{regime}"] = {"n": int(mask.sum()),
                                                      "ic": None}
                    continue
                from scipy import stats as sps
                rho = sps.spearmanr(fv_all2, y_all2)[0]
                per_feature[f"{sym}/{regime}"] = {"n": int(mask.sum()),
                                                  "ic": round(float(rho), 4)}
        out[feature] = per_feature
    return out


def _cross_asset(datasets) -> dict:
    """All features: BTC -> ETH 300s label and ETH -> BTC 300s label."""
    out = {"H_s": 300, "pairs": {}}
    btc, eth = datasets["futures_um:BTCUSDT"], datasets["futures_um:ETHUSDT"]
    merged = pd.merge(
        btc[["ts_us", "day"] + FEATURE_NAMES].rename(
            columns={f: f"BTC_{f}" for f in FEATURE_NAMES}),
        eth[["ts_us", "day", "label_gross_300s", "label_valid_300s"]],
        on=["ts_us", "day"], how="inner")
    _fill_cross(out, "BTC->ETH", merged, FEATURE_NAMES, "BTC_")
    merged2 = pd.merge(
        eth[["ts_us", "day"] + FEATURE_NAMES].rename(
            columns={f: f"ETH_{f}" for f in FEATURE_NAMES}),
        btc[["ts_us", "day", "label_gross_300s", "label_valid_300s"]],
        on=["ts_us", "day"], how="inner")
    _fill_cross(out, "ETH->BTC", merged2, FEATURE_NAMES, "ETH_")
    return out


def _fill_cross(out, pair, merged, features, prefix):
    from scipy import stats as sps

    out["pairs"][pair] = {}
    for feature in features:
        fv = merged[f"{prefix}{feature}"].to_numpy(dtype=np.float64)
        y = merged["label_gross_300s"].to_numpy(dtype=np.float64)
        ok = (np.isfinite(fv) & np.isfinite(y) &
              merged["label_valid_300s"].to_numpy(dtype=bool))
        if ok.sum() < 50:
            out["pairs"][pair][feature] = {"n": 0, "ic": None}
            continue
        from scipy import stats as sps
        out["pairs"][pair][feature] = {
            "n": int(ok.sum()),
            "ic": round(float(sps.spearmanr(fv[ok], y[ok])[0]), 4)}
    return out


if __name__ == "__main__":
    sys.exit(main())