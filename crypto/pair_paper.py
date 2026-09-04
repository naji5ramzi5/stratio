"""Pair paper book — dedicated two-leg paper trading for the relative-value
edge. Kept separate from the single-leg PaperTradingEngine on purpose: a
spread position has no market price feed, so price-based SL/TP does not apply.

Strengthened vs v1:
  * Multi-TF confirmation: entry only when 1h AND 4h agree on direction.
  * Spread stop-loss: exit if |z| exceeds z_stop (cointegration broke).
  * Position sizing: scales with z-extremity (bigger edge = bigger size, capped).
  * Portfolio exposure cap: gross notional across all open pairs <= max_gross.
  * PnL mirrors crypto/stat_arb.backtest_spread exactly.
"""
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np

import stat_arb

logger = logging.getLogger(__name__)

PAIR_PAPER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "paper_pairs.json")
DEFAULT_COST_BPS = 25.0
DEFAULT_MAX_HOLD_BARS = 168
DEFAULT_MAX_GROSS_EXPOSURE = 0.5
MAX_POSITION_SIZE = 0.15
MIN_POSITION_SIZE = 0.03


class PairPosition:
    def __init__(self, a, b, direction, hedge_ratio, entry_spread, entry_z,
                 entry_time, cost_bps, size_pct):
        self.pair = f"{a}/{b}"
        self.a = a
        self.b = b
        self.direction = direction
        self.hedge_ratio = hedge_ratio
        self.entry_spread = entry_spread
        self.entry_z = entry_z
        self.entry_time = entry_time
        self.bars_held = 0
        self.exit_time = None
        self.exit_z = None
        self.exit_reason = None
        self.pnl_pct = None
        self.cost_bps = cost_bps
        self.size_pct = size_pct
        self.id = f"{self.pair}_{direction:+.0f}_{datetime.now(timezone.utc).timestamp():.0f}"

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(d):
        p = PairPosition(d["a"], d["b"], d["direction"], d["hedge_ratio"],
                         d["entry_spread"], d["entry_z"], d["entry_time"],
                         d["cost_bps"], d.get("size_pct", 0.05))
        p.__dict__.update(d)
        return p


def _size_for_z(z_abs):
    """Bigger z-extremity = clearer edge = bigger size, capped."""
    if z_abs <= 2.0:
        return MIN_POSITION_SIZE
    size = MIN_POSITION_SIZE + (z_abs - 2.0) * ((MAX_POSITION_SIZE - MIN_POSITION_SIZE) / 2.0)
    return min(size, MAX_POSITION_SIZE)


class PairPaperBook:
    """Persistent pairs book with multi-TF confirmation, spread SL, and
    portfolio exposure cap."""

    def __init__(self, file=PAIR_PAPER_FILE):
        self.file = file
        self.positions = []
        self.closed = []
        self.load()

    @property
    def open_positions(self):
        return [p for p in self.positions if p.exit_reason is None]

    @property
    def gross_exposure(self):
        return sum(p.size_pct for p in self.open_positions)

    def _position_index(self, pair):
        for i, p in enumerate(self.positions):
            if p.pair == pair and p.exit_reason is None:
                return i
        return -1

    def _baseline(self, df, train_bars):
        """Compute causal baseline on PAST bars and evaluate the current (last)
        bar. Returns (direction, z, hedge, pvalue) or None if not usable.
        direction: +1 long spread, -1 short spread, 0 flat/no-coint.

        CRITICAL: the current bar is EXCLUDED from the baseline. Including it
        would inflate std and shrink a real stretch (z), weakening the signal."""
        if df is None or len(df) < train_bars + 10:
            return None
        a_arr = df["a"].to_numpy(dtype=float)
        b_arr = df["b"].to_numpy(dtype=float)
        n = len(a_arr)
        tr_lo = n - train_bars
        a_tr, b_tr = a_arr[tr_lo:n - 1], b_arr[tr_lo:n - 1]
        if len(a_tr) < 50:
            return None
        hedge = stat_arb.ols_hedge_ratio(a_tr, b_tr)
        passed, pvalue, hedge = stat_arb.test_cointegration(a_tr, b_tr, hedge_ratio=hedge)
        if not passed or pvalue is None:
            return (0, 0.0, hedge, pvalue or 1.0)
        s_tr = stat_arb.spread_series(a_tr, b_tr, hedge)
        mean, std = float(np.mean(s_tr)), float(np.std(s_tr))
        if std <= 0:
            return (0, 0.0, hedge, pvalue)
        spread_now = stat_arb.spread_series(a_arr[n - 1:n], b_arr[n - 1:n], hedge)[0]
        z = (spread_now - mean) / std
        direction = 0
        if z <= -2.0:
            direction = 1
        elif z >= 2.0:
            direction = -1
        return (direction, z, hedge, pvalue)

    def _direction_df(self, df, train_bars):
        out = self._baseline(df, train_bars)
        if out is None:
            return 0, 0.0, 1.0, 1.0
        return out

    def _direction_on_interval(self, a, b, interval, lookback, train_bars):
        try:
            df = stat_arb.fetch_aligned_pair(a, b, interval, lookback)
        except Exception:
            return 0, 0.0, 1.0, 1.0
        return self._direction_df(df, train_bars)

    def step(self, a, b, interval="1h", lookback=1500, train_bars=1000,
             z_entry=2.0, z_exit=0.5, z_stop=3.5,
             cost_bps=DEFAULT_COST_BPS, max_hold_bars=DEFAULT_MAX_HOLD_BARS,
             max_gross=DEFAULT_MAX_GROSS_EXPOSURE, df=None, df_4h=None):
        events = []
        if df is None:
            try:
                df = stat_arb.fetch_aligned_pair(a, b, "1h", lookback)
            except Exception:
                df = None
        if df is None or len(df) < train_bars + 10:
            events.append({"type": "error", "pair": f"{a}/{b}", "note": "insufficient data"})
            return events

        base = self._baseline(df, train_bars)
        if base is None:
            return events
        dir_1h, z_now, hedge, pvalue = base
        a_arr = df["a"].to_numpy(dtype=float)
        b_arr = df["b"].to_numpy(dtype=float)
        spread_now = stat_arb.spread_series(a_arr[-1:], b_arr[-1:], hedge)[0]
        key = f"{a}/{b}"
        cost = cost_bps / 10000.0

        if df_4h is not None:
            dir_4h, z_4h, _, _ = self._direction_df(df_4h, 400)
        else:
            dir_4h, z_4h, _, _ = self._direction_on_interval(a, b, "4h", 800, 400)

        idx = self._position_index(key)

        if idx >= 0:
            pos = self.positions[idx]
            pos.bars_held += 1
            closed = False
            reason = None
            if abs(z_now) <= z_exit:
                reason, closed = "closed_revert", True
            elif abs(z_now) >= z_stop:
                reason, closed = "closed_stoploss", True
            elif pos.bars_held >= max_hold_bars:
                reason, closed = "closed_expired", True
            if closed:
                pnl = pos.direction * (spread_now - pos.entry_spread) - 2 * cost
                pos.pnl_pct = round(pnl * 100, 4)
                pos.exit_z = round(float(z_now), 3)
                pos.exit_time = datetime.now(timezone.utc).isoformat(timespec="seconds")
                pos.exit_reason = reason
                self.closed.append(pos.to_dict())
                events.append({"type": "close", "pair": key, "reason": reason,
                               "pnl_pct": pos.pnl_pct, "bars_held": pos.bars_held,
                               "z": pos.exit_z, "size_pct": pos.size_pct})
                self.save()
            return events

        if dir_1h == 0 or abs(z_now) < z_entry:
            return events
        direction = float(dir_1h)
        if dir_4h == 0 or (dir_4h > 0) != (direction > 0):
            events.append({"type": "skip_mtf", "pair": key,
                           "note": "1h dir=%+.0f but 4h dir=%+.0f" % (direction, dir_4h)})
            return events
        if self.gross_exposure >= max_gross:
            events.append({"type": "skip_exposure", "pair": key,
                           "note": "gross %.0f%% >= %.0f%%" % (self.gross_exposure * 100, max_gross * 100)})
            return events
        size = _size_for_z(abs(z_now))
        pos = PairPosition(a, b, direction, hedge, spread_now, float(z_now),
                           datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           cost_bps, round(size, 4))
        self.positions.append(pos)
        self.save()
        events.append({"type": "open", "pair": key,
                       "direction": "LONG_SPREAD" if direction > 0 else "SHORT_SPREAD",
                       "z": round(float(z_now), 3), "z_4h": round(float(z_4h), 3),
                       "hedge": round(hedge, 4), "size_pct": pos.size_pct,
                       "spread": round(float(spread_now), 6)})
        return events

    def stats(self):
        closed = [p for p in self.closed if p.get("pnl_pct") is not None]
        if not closed:
            return {"status": "no_closed", "open_positions": len(self.open_positions),
                    "gross_exposure": self.gross_exposure, "total_closed": len(self.closed)}
        wins = sum(1 for p in closed if p["pnl_pct"] > 0)
        pnls = [p["pnl_pct"] * p.get("size_pct", 0.05) for p in closed]
        return {
            "status": "active" if self.open_positions else "idle",
            "open_positions": len(self.open_positions),
            "gross_exposure": self.gross_exposure,
            "total_closed": len(closed),
            "wins": wins, "losses": len(closed) - wins,
            "win_rate_pct": round(wins / len(closed) * 100, 1),
            "avg_pnl_contrib_pct": round(float(np.mean(pnls)), 4),
            "total_contrib_pct": round(float(np.sum(pnls)), 3),
        }

    def save(self):
        try:
            data = {"positions": [p.to_dict() for p in self.positions],
                    "closed": self.closed[-500:]}
            with open(self.file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[PAIR-PAPER] save failed: {e}")

    def load(self):
        if not os.path.exists(self.file):
            return
        try:
            with open(self.file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.positions = [PairPosition.from_dict(p) for p in data.get("positions", [])]
            self.closed = data.get("closed", [])
        except Exception as e:
            logger.warning(f"[PAIR-PAPER] load failed: {e}")


_book = PairPaperBook()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="pair_paper", description="two-leg paper book for the pairs edge")
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"))
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--lookback", type=int, default=1500)
    ap.add_argument("--train-bars", type=int, default=1000)
    ap.add_argument("--z-entry", type=float, default=2.0)
    ap.add_argument("--z-exit", type=float, default=0.5)
    ap.add_argument("--z-stop", type=float, default=3.5)
    ap.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    ap.add_argument("--max-hold-bars", type=int, default=DEFAULT_MAX_HOLD_BARS)
    ap.add_argument("--max-gross", type=float, default=DEFAULT_MAX_GROSS_EXPOSURE)
    ap.add_argument("--book", default=PAIR_PAPER_FILE)
    ap.add_argument("--stats", action="store_true", help="print book stats and exit")
    args = ap.parse_args(argv)

    book = PairPaperBook(args.book)
    if args.stats:
        print(json.dumps(book.stats(), ensure_ascii=False, indent=2))
        return 0
    if not args.pair:
        ap.error("--pair A B is required unless --stats is given")

    events = book.step(args.pair[0], args.pair[1], args.interval, args.lookback,
                       args.train_bars, args.z_entry, args.z_exit, args.z_stop,
                       args.cost_bps, args.max_hold_bars, args.max_gross)
    for ev in events:
        print(json.dumps(ev, ensure_ascii=False))
    if not events:
        print('{"type":"hold"}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
