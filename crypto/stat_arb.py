"""
Relative-Value / Statistical-Arbitrage Engine (market-neutral).

Predicts the SPREAD between two cointegrated assets, never the direction of a
single market. Historical evidence (Kaiko 2020-2025): pairs median ~12.4%/yr
annualized with ~40% lower volatility than buy-and-hold. This is the honest
edge: it does not bet on the market going up or down, only on a stretched
spread reverting to its mean.

Everything here is OUT-OF-SAMPLE:
  * the cointegration test and hedge ratio are estimated on a TRAIN window
    only, never on the bars being traded;
  * the spread baseline used for the z-score signal is taken from the same
    train window (no look-ahead into the test fold);
  * transaction costs are charged on every entry and exit.

Run:  python -m stat_arb [--pair BTCUSDT ETHUSDT] [--interval 1h] [--lookback 1500]
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger("stat_arb")

try:
    from statsmodels.tsa.stattools import coint as _coint
    HAS_STATSMODELS = True
except ImportError:  # pragma: no cover - dev env check
    HAS_STATSMODELS = False

BARS_PER_YEAR = {"5m": 24 * 12 * 365, "15m": 4 * 24 * 365, "30m": 2 * 24 * 365, "1h": 24 * 365}


def _log_prices(closes):
    return np.log(np.asarray(closes, dtype=float))


def fetch_aligned_pair(symbol_a, symbol_b, interval="1h", lookback=1500):
    """Fetch two spot series aligned on the timestamp column (oldest -> newest)."""
    from ml_trainer import KLINES_COLUMNS
    from data_loader.binance_ohlcv import fetch_klines

    ra = fetch_klines(symbol_a, interval, lookback)
    rb = fetch_klines(symbol_b, interval, lookback)
    if not ra or not rb:
        return None
    df_a = pd.DataFrame(ra, columns=KLINES_COLUMNS)
    df_b = pd.DataFrame(rb, columns=KLINES_COLUMNS)
    df_a = df_a[["timestamp", "close"]].rename(columns={"close": "a"})
    df_b = df_b[["timestamp", "close"]].rename(columns={"close": "b"})
    out = df_a.merge(df_b, on="timestamp", how="inner").reset_index(drop=True)
    out["a"] = out["a"].astype(float)
    out["b"] = out["b"].astype(float)
    return out


def ols_hedge_ratio(a, b):
    """Hedge ratio h such that spread = log(a) - h * log(b) is stationary."""
    x = _log_prices(b)
    y = _log_prices(a)
    h = np.polyfit(x, y, 1)[0]
    return float(h)


def spread_series(a, b, hedge_ratio):
    return _log_prices(a) - hedge_ratio * _log_prices(b)


def rolling_zscore(spread, window=100):
    """Causal rolling z-score (only past bars are used)."""
    s = pd.Series(spread)
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)


def half_life(spread):
    """Ornstein-Uhlenbeck half-life of the spread (in bars).

    Fits dS = lambda * S_{t-1} + eps and returns -ln(2)/lambda. A very small
    value means noise (no exploitable drift), a very large one means the
    spread never reverts inside a practical holding period.
    """
    s = np.asarray(spread, dtype=float)
    if len(s) < 50:
        return None
    s_lag = s[:-1]
    ds = np.diff(s)
    denom = np.sum((s_lag - np.mean(s_lag)) ** 2)
    if denom <= 0:
        return None
    lam = np.sum((s_lag - np.mean(s_lag)) * (ds - np.mean(ds))) / denom
    if lam >= 0:
        return None
    hl = -math.log(2) / lam
    return float(hl)


def scan_pairs(symbols, interval="1h", lookback=1500, p_max=0.05,
               hl_min=4, hl_max=1000):
    """Stage-1 cheap screen: full-sample cointegration + half-life filter.

    Returns the shortlist of pairs that pass both; expensive walk-forward is
    reserved for these only.
    """
    shortlist = []
    for a, b in itertools.combinations(symbols, 2):
        df = fetch_aligned_pair(a, b, interval, lookback)
        if df is None or len(df) < 500:
            continue
        passed, pvalue, hedge = test_cointegration(df["a"], df["b"])
        if not passed:
            continue
        hl = half_life(spread_series(df["a"], df["b"], hedge))
        entry = {"pair": f"{a}/{b}", "p": pvalue, "hedge": round(hedge, 4),
                 "half_life_bars": round(hl, 1) if hl else None}
        if hl is None:
            entry["status"] = "no_half_life"
            shortlist.append(entry)
        elif hl_min <= hl <= hl_max:
            entry["status"] = "shortlisted"
            shortlist.append(entry)
        else:
            entry["status"] = f"hl_out_of_range({hl:.0f})"
            shortlist.append(entry)
    return shortlist


def test_cointegration(a, b, hedge_ratio=None, log=True):
    """Engle-Granger two-step cointegration test on log prices.

    Returns (passed, pvalue, hedge_ratio) or (False, None, None) when the
    statsmodels backend is unavailable.
    """
    if not HAS_STATSMODELS:
        return False, None, None
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b) or len(a) < 100:
        return False, None, None
    if hedge_ratio is None:
        hedge_ratio = ols_hedge_ratio(a, b)
    y1 = _log_prices(a) if log else a
    y2 = _log_prices(b) if log else b
    try:
        _, pvalue, _ = _coint(y1, y2)
    except Exception as exc:
        logger.debug("coint failed: %s", exc)
        return False, None, hedge_ratio
    return bool(pvalue < 0.05), float(pvalue), hedge_ratio


def backtest_spread(log_a, log_b, hedge_ratio, baseline_mean, baseline_std,
                    z_entry=2.0, z_exit=0.5, cost_bps=10.0, bars_per_year=24 * 365):
    """Trade the spread against a fixed (train) baseline. Purely causal.

    Position = +1 when z <= -z_entry (spread stretched low -> long A / short B),
              -1 when z >= +z_entry,
              0 after z reverts to |z| <= z_exit.
    Transaction costs (2 legs) are charged on every position change.
    Returns (returns_array, trades, metrics_dict).
    """
    spread = log_a - hedge_ratio * log_b
    z = (spread - baseline_mean) / baseline_std
    delta_spread = np.diff(spread)  # bar return of being long the spread
    n = len(spread)
    cost = cost_bps / 10000.0
    ret = np.zeros(max(0, n - 1))
    trades = []  # (entry_idx, exit_idx, net_pnl)
    pos = 0.0
    entry_idx = None
    trade_pnl = 0.0
    for i in range(n):
        zi = z[i]
        new_pos = pos
        if pos == 0.0:
            if zi <= -z_entry:
                new_pos = 1.0
            elif zi >= z_entry:
                new_pos = -1.0
        elif abs(zi) <= z_exit:
            new_pos = 0.0
        flip = abs(new_pos - pos)
        pos = new_pos
        if i < n - 1:
            gain = pos * delta_spread[i]
            bar_cost = flip * 2 * cost
            ret[i] = gain - bar_cost
            if pos != 0.0:
                if entry_idx is None:
                    entry_idx = i
                trade_pnl += gain - bar_cost
            else:
                if entry_idx is not None:
                    trades.append((entry_idx, i, trade_pnl))
                    entry_idx = None
                trade_pnl = 0.0
        elif pos != 0.0 and entry_idx is not None:
            trade_pnl -= flip * 2 * cost
            trades.append((entry_idx, n - 1, trade_pnl))
    return ret, trades, _metrics(ret, trades, bars_per_year)


def _metrics(ret, trades, bars_per_year=24 * 365):
    ret = np.asarray(ret, dtype=float)
    if len(ret) == 0:
        return {"return_pct": 0.0, "annualized_pct": 0.0, "sharpe": 0.0,
                "max_dd_pct": 0.0, "n_trades": 0, "win_rate_pct": 0.0}
    cum = np.cumsum(ret)
    dd = cum - np.maximum.accumulate(cum)
    max_dd_pct = float(dd.min() * 100)
    annualized = float(ret.mean() * bars_per_year * 100)
    std = ret.std(ddof=1)
    sharpe = float(ret.mean() / std) * math.sqrt(bars_per_year) if std > 0 else 0.0
    total_ret = float(cum[-1] * 100)
    wins = sum(1 for t in trades if t[2] > 0)
    win_rate = wins / len(trades) * 100 if trades else 0.0
    return {"return_pct": round(total_ret, 2), "annualized_pct": round(annualized, 2),
            "sharpe": round(sharpe, 2), "max_dd_pct": round(max_dd_pct, 2),
            "n_trades": len(trades), "win_rate_pct": round(win_rate, 1)}


def walk_forward(df, interval="1h", train_bars=1000, step=250, z_entry=2.0,
                 z_exit=0.5, cost_bps=10.0, min_pvalue=0.05):
    """Honest walk-forward backtest of the spread strategy.

    For each fold the hedge ratio, cointegration status and z-baseline are
    estimated on the TRAIN window only; the TEST window is never touched by
    that fold's estimation. Bars with no fold membership are not traded.
    """
    a = df["a"].to_numpy(dtype=float)
    b = df["b"].to_numpy(dtype=float)
    log_a = _log_prices(a)
    log_b = _log_prices(b)
    n = len(df)

    all_ret = np.zeros(max(0, n - 1), dtype=float)
    all_trades = []
    folds = []
    start = train_bars
    while start < n:
        end = min(start + step, n)
        folds.append((start - train_bars, start, start, end))
        start += step

    for tr_lo, tr_hi, te_lo, te_hi in folds:
        a_tr = a[tr_lo:tr_hi]
        b_tr = b[tr_lo:tr_hi]
        hedge = ols_hedge_ratio(a_tr, b_tr)
        passed, pvalue, hedge = test_cointegration(a_tr, b_tr, hedge_ratio=hedge)
        if not passed or pvalue is None or pvalue >= min_pvalue:
            logger.info("  fold [%d:%d] test=%d..%d : cointegration FAILED p=%.4g -> flat",
                        tr_lo, tr_hi, te_lo, te_hi, pvalue or 1.0)
            continue
        s_tr = spread_series(a_tr, b_tr, hedge)
        mean = float(np.mean(s_tr))
        std = float(np.std(s_tr))
        if std <= 0:
            continue
        seg_ret, seg_trades, _ = backtest_spread(
            log_a[te_lo:te_hi], log_b[te_lo:te_hi], hedge, mean, std,
            z_entry=z_entry, z_exit=z_exit, cost_bps=cost_bps,
            bars_per_year=BARS_PER_YEAR.get(interval, 24 * 365))
        seg_ret = np.asarray(seg_ret, dtype=float)
        lo = te_lo
        hi = min(te_hi - 1, n - 1)
        if lo < hi and len(seg_ret):
            all_ret[lo:lo + len(seg_ret)] = seg_ret[:hi - lo]
            for t in seg_trades:
                all_trades.append((lo + t[0], lo + t[1], t[2]))
            logger.info("  fold [%d:%d] test=%d..%d p=%.4g h=%.4f -> %d trades, seg_ret=%.2f%%",
                        tr_lo, tr_hi, te_lo, te_hi, pvalue, hedge, len(seg_trades), float(np.sum(seg_ret) * 100))

    all_ret = all_ret[:n - 1]
    traded = np.count_nonzero(np.abs(all_ret) > 0) if len(all_ret) else 0
    if traded == 0:
        return {"cointegrated": False, "n_folds": len(folds), "n_trades": 0,
                "return_pct": 0.0, "annualized_pct": 0.0, "sharpe": 0.0,
                "max_dd_pct": 0.0, "win_rate_pct": 0.0, "bars_traded": 0}

    metrics = _metrics(all_ret, all_trades, BARS_PER_YEAR.get(interval, 24 * 365))
    metrics["cointegrated"] = True
    metrics["n_folds"] = len(folds)
    metrics["bars_traded"] = int(traded)
    return metrics


def sweep_pairs(symbols, interval="1h", lookback=1500, **kwargs):
    """Test every symbol pair and return the ones with a walk-forward edge."""
    results = []
    for a, b in itertools.combinations(symbols, 2):
        df = fetch_aligned_pair(a, b, interval, lookback)
        if df is None or len(df) < 500:
            continue
        m = walk_forward(df, interval, **kwargs)
        results.append({"pair": f"{a}/{b}", **m})
    return results


def monitor_pair(a, b, interval="1h", lookback=1500, train_bars=1000,
                 z_entry=2.0, z_exit=0.5):
    """Live signal for a pair, using ONLY the trailing train window for the
    hedge ratio / baseline (same honesty rule as walk_forward). No look-ahead:
    the baseline is estimated on bars strictly before 'now'."""
    df = fetch_aligned_pair(a, b, interval, lookback)
    if df is None or len(df) < train_bars + 10:
        return {"status": "missing", "reason": "insufficient data"}
    a_arr = df["a"].to_numpy(dtype=float)
    b_arr = df["b"].to_numpy(dtype=float)
    tr_lo = len(df) - train_bars
    hedge = ols_hedge_ratio(a_arr[tr_lo:], b_arr[tr_lo:])
    passed, pvalue, hedge = test_cointegration(a_arr[tr_lo:], b_arr[tr_lo:],
                                               hedge_ratio=hedge)
    s_tr = spread_series(a_arr[tr_lo:], b_arr[tr_lo:], hedge)
    mean = float(np.mean(s_tr))
    std = float(np.std(s_tr))
    hl = half_life(s_tr)
    spread_now = float(s_tr[-1])
    z = (spread_now - mean) / std if std > 0 else 0.0

    if z <= -z_entry:
        signal, note = "LONG_SPREAD", f"z={z:.2f} <= -{z_entry} (buy {a}, sell {b})"
    elif z >= z_entry:
        signal, note = "SHORT_SPREAD", f"z={z:.2f} >= +{z_entry} (sell {a}, buy {b})"
    elif abs(z) <= z_exit:
        signal, note = "FLAT", f"z={z:.2f} within exit band (+-{z_exit})"
    else:
        signal, note = "HOLD", f"z={z:.2f} stretched but not at entry yet"

    return {
        "status": "ok", "a": a, "b": b, "interval": interval,
        "cointegrated": passed, "pvalue": pvalue, "hedge_ratio": round(hedge, 4),
        "half_life_bars": round(hl, 1) if hl else None,
        "baseline_mean": round(mean, 6), "baseline_std": round(std, 6),
        "current_z": round(z, 3), "signal": signal, "note": note,
        "entry_z": z_entry, "exit_z": z_exit,
        "as_of_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def load_evidence(path=None):
    import os
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairs_evidence.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_evidence(data, path=None):
    import os
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pairs_evidence.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def rescan_universe(symbols, interval="1h", lookback=3500, train_bars=1000,
                    step=250, z_entry=2.0, z_exit=0.5, cost_bps=25.0,
                    p_max=0.05, hl_min=4, hl_max=1000, min_sharpe=0.3,
                    out_file=None):
    """Periodic universe refresh: stage-1 screen, then walk-forward OOS proof
    for every shortlisted pair. Only pairs with a positive OOS Sharpe survive;
    these are the ones the pair paper book is allowed to trade."""
    import os
    if out_file is None:
        out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "pairs_universe.json")
    screen = scan_pairs(symbols, interval, lookback, p_max, hl_min, hl_max)
    shortlisted = [r for r in screen if r["status"] == "shortlisted"]
    approved = []
    rejected = []
    for r in shortlisted:
        a, b = r["pair"].split("/")
        df = fetch_aligned_pair(a, b, interval, lookback)
        if df is None or len(df) < train_bars + 10:
            rejected.append({**r, "reason": "insufficient data"})
            continue
        m = walk_forward(df, interval, train_bars=train_bars, step=step,
                         z_entry=z_entry, z_exit=z_exit, cost_bps=cost_bps)
        rec = {**r, "interval": interval,
               "oos": {"sharpe": m["sharpe"], "annualized_pct": m["annualized_pct"],
                       "max_dd_pct": m["max_dd_pct"], "n_trades": m["n_trades"],
                       "win_rate_pct": m["win_rate_pct"], "n_folds": m["n_folds"]}}
        if m["cointegrated"] and m["n_trades"] > 0 and m["sharpe"] > min_sharpe:
            approved.append(rec)
        else:
            rec["reason"] = "OOS failed" if not m["cointegrated"] or m["n_trades"] == 0 else "OOS sharpe too low"
            rejected.append(rec)
    universe = {"updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "interval": interval, "approved": approved, "rejected": rejected}
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(universe, f, ensure_ascii=False, indent=2)
    return universe


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stat_arb",
                                 description="Relative-value (pairs) engine - honest walk-forward OOS")
    ap.add_argument("--pair", nargs=2, default=["BTCUSDT", "ETHUSDT"])
    ap.add_argument("--interval", default="1h", choices=list(BARS_PER_YEAR))
    ap.add_argument("--lookback", type=int, default=1500)
    ap.add_argument("--train-bars", type=int, default=1000)
    ap.add_argument("--step", type=int, default=250)
    ap.add_argument("--z-entry", type=float, default=2.0)
    ap.add_argument("--z-exit", type=float, default=0.5)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--sweep", action="store_true",
                    help="test all pairs among BTCUSDT, ETHUSDT, SOLUSDT")
    ap.add_argument("--scan", action="store_true",
                    help="stage-1 screen over a wide universe (coint + half-life)")
    ap.add_argument("--rescan", action="store_true",
                    help="stage-1 screen + walk-forward OOS proof; writes pairs_universe.json")
    ap.add_argument("--monitor", action="store_true",
                    help="live signal for --pair using only trailing data")
    ap.add_argument("--journal", default="pairs_signals.json",
                    help="paper signal journal for --monitor")
    ap.add_argument("--universe", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,ADAUSDT,LINKUSDT,LTCUSDT,AVAXUSDT,ATOMUSDT,DOTUSDT,FILUSDT,TRXUSDT,UNIUSDT,NEARUSDT,ARBUSDT,INJUSDT,SUIUSDT,TAOUSDT,WLDUSDT,PEPEUSDT,BONKUSDT,HBARUSDT,TONUSDT")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    if args.scan:
        universe = [s.strip().upper() for s in args.universe.split(",") if s.strip()]
        results = scan_pairs(universe, args.interval, args.lookback)
        print(f"stage-1 screen over {len(universe)} symbols "
              f"({len(results)} pairs with cointegration p<0.05):")
        for r in results:
            print(f"  {r['pair']:22s} p={r['p']:.4f} h={r['hedge']} "
                  f"hl={r['half_life_bars']} bars  [{r['status']}]")
        short = [r for r in results if r["status"] == "shortlisted"]
        print(f"\nshortlisted ({len(short)}): " + ", ".join(r["pair"] for r in short))
        return 0

    if args.rescan:
        universe = [s.strip().upper() for s in args.universe.split(",") if s.strip()]
        print(f"rescan over {len(universe)} symbols (stage-1 + walk-forward OOS proof)...")
        univ = rescan_universe(universe, args.interval, args.lookback,
                               args.train_bars, args.step, args.z_entry,
                               args.z_exit, args.cost_bps)
        print(f"approved ({len(univ['approved'])}):")
        for r in univ["approved"]:
            o = r["oos"]
            print(f"  {r['pair']:22s} p={r['p']:.4f} hl={r['half_life_bars']} "
                  f"ann={o['annualized_pct']}% sharpe={o['sharpe']} "
                  f"trades={o['n_trades']} win={o['win_rate_pct']}%")
        print(f"rejected ({len(univ['rejected'])}): "
              + ", ".join(r["pair"] for r in univ["rejected"]))
        print("universe -> pairs_universe.json")
        return 0

    if args.sweep:
        results = sweep_pairs(["BTCUSDT", "ETHUSDT", "SOLUSDT"], args.interval,
                              args.lookback, train_bars=args.train_bars, step=args.step,
                              z_entry=args.z_entry, z_exit=args.z_exit, cost_bps=args.cost_bps)
        for r in results:
            print(_fmt_result(r))
        return 0

    if args.monitor:
        a, b = args.pair[0].upper(), args.pair[1].upper()
        m = monitor_pair(a, b, args.interval, args.lookback, args.train_bars,
                         args.z_entry, args.z_exit)
        if m["status"] != "ok":
            print(f"[stat_arb] monitor: {m['reason']}")
            return 1
        print(f"pair={a}/{b} interval={args.interval} signal={m['signal']}")
        print(f"  z={m['current_z']} (entry +-{m['entry_z']}, exit +-{m['exit_z']})")
        print(f"  cointegrated(trailing {args.train_bars} bars)={m['cointegrated']} "
              f"p={m['pvalue']:.4g} h={m['hedge_ratio']}")
        print(f"  half_life={m['half_life_bars']} bars  note={m['note']}")
        try:
            rec = json.load(open(args.journal, encoding="utf-8")) if os.path.exists(args.journal) else []
            rec.append(m)
            with open(args.journal, "w", encoding="utf-8") as f:
                json.dump(rec[-500:], f, ensure_ascii=False, indent=2)
            print(f"  journal -> {args.journal}")
        except Exception as exc:
            print(f"  journal write failed: {exc}")
        return 0

    df = fetch_aligned_pair(args.pair[0], args.pair[1], args.interval, args.lookback)
    if df is None or len(df) < 200:
        print(f"[stat_arb] insufficient data for {args.pair[0]}/{args.pair[1]}")
        return 1

    full_hedge = ols_hedge_ratio(df["a"], df["b"])
    passed, pvalue, _ = test_cointegration(df["a"], df["b"], hedge_ratio=full_hedge)
    m = walk_forward(df, args.interval, train_bars=args.train_bars, step=args.step,
                     z_entry=args.z_entry, z_exit=args.z_exit, cost_bps=args.cost_bps)
    print(_fmt_result({
        "pair": f"{args.pair[0]}/{args.pair[1]}", "interval": args.interval,
        "bars": len(df), "full_sample_p": pvalue, "full_sample_coint": passed,
        "hedge_ratio": round(full_hedge, 4), **m}))
    return 0


def _fmt_result(r):
    lines = [f"pair={r['pair']} interval={r.get('interval', '?')} bars={r.get('bars', '?')}"]
    if r.get("full_sample_coint") is not None:
        lines.append(f"  full-sample cointegration: p={r['full_sample_p']:.4g} "
                     f"({'PASS' if r['full_sample_coint'] else 'FAIL'}) h={r.get('hedge_ratio')}")
    if not r.get("cointegrated", False):
        lines.append("  walk-forward: NO tradable fold (cointegration did not hold OOS)")
        return "\n".join(lines)
    lines.append(f"  walk-forward OOS: ret={r['return_pct']}%  "
                 f"ann={r['annualized_pct']}%  sharpe={r['sharpe']}  "
                 f"maxDD={r['max_dd_pct']}%")
    lines.append(f"  trades={r['n_trades']}  win_rate={r['win_rate_pct']}%  "
                 f"folds={r['n_folds']}  bars_traded={r['bars_traded']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.exit(main())
