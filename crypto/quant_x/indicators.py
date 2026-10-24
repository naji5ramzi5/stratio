"""Technical heuristics for QUANT-X steps 1-5.

All functions operate on a pandas DataFrame built from klines:
``timestamp, open, high, low, close, volume, taker_buy_vol, ...``.

Methods are clearly-labelled heuristics (fractals, FVG, order blocks,
Wyckoff phase detection, volume profile) — they are structured observations,
not guarantees. Every result carries its input evidence so the report can
explain *why*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# STEP 1 — Market Structure
# ---------------------------------------------------------------------------

def detect_swings(df, k=3):
    """Fractal swing highs/lows (lookback k bars on each side)."""
    highs, lows = [], []
    n = len(df)
    for i in range(k, n - k):
        h = df["high"].iloc[i]
        l = df["low"].iloc[i]
        if h > df["high"].iloc[i - k:i].max() and h > df["high"].iloc[i + 1:i + k + 1].max():
            highs.append({"idx": i, "price": float(h)})
        if l < df["low"].iloc[i - k:i].min() and l < df["low"].iloc[i + 1:i + k + 1].min():
            lows.append({"idx": i, "price": float(l)})
    return highs, lows


def market_structure(df, k=3):
    """Trend, HH/HL/LH/LL, last BOS and CHOCH from the last ~10 swings."""
    highs, lows = detect_swings(df, k=k)
    swings = sorted(highs + lows, key=lambda s: s["idx"])
    swings = swings[-10:] if len(swings) > 10 else swings
    if len(swings) < 3:
        return {"status": "insufficient_swings", "n_swings": len(swings)}

    last_high = highs[-1]["price"] if highs else None
    prev_high = highs[-2]["price"] if len(highs) >= 2 else None
    last_low = lows[-1]["price"] if lows else None
    prev_low = lows[-2]["price"] if len(lows) >= 2 else None

    hh = last_high is not None and prev_high is not None and last_high > prev_high
    hl = last_low is not None and prev_low is not None and last_low > prev_low
    lh = last_high is not None and prev_high is not None and last_high < prev_high
    ll = last_low is not None and prev_low is not None and last_low < prev_low

    if hh and hl:
        trend = "UPTREND"
    elif lh and ll:
        trend = "DOWNTREND"
    else:
        trend = "RANGING"

    close = float(df["close"].iloc[-1])
    bos = None
    choch = None
    if trend == "UPTREND" and last_high is not None:
        if close > last_high:
            bos = {"type": "BOS_UP", "level": last_high, "close": close}
    if trend == "DOWNTREND" and last_low is not None:
        if close < last_low:
            bos = {"type": "BOS_DOWN", "level": last_low, "close": close}
    # CHOCH: a shift against the trend against the most recent swing
    if trend == "UPTREND" and last_low is not None and close < last_low:
        choch = {"type": "CHOCH_DOWN", "level": last_low, "close": close}
    if trend == "DOWNTREND" and last_high is not None and close > last_high:
        choch = {"type": "CHOCH_UP", "level": last_high, "close": close}

    return {
        "status": "ok",
        "trend": trend,
        "hh": bool(hh), "hl": bool(hl), "lh": bool(lh), "ll": bool(ll),
        "last_swing_high": last_high, "last_swing_low": last_low,
        "bos": bos, "choch": choch,
        "close": close,
        "n_swings": len(swings),
    }


# ---------------------------------------------------------------------------
# STEP 2 — Liquidity
# ---------------------------------------------------------------------------

def _clusters(values, tol_pct=0.25):
    """Group swing levels that are within ``tol_pct`` of each other."""
    clusters = []
    for v in sorted(values):
        if clusters and abs(clusters[-1]["avg"] - v) / clusters[-1]["avg"] * 100 <= tol_pct:
            c = clusters[-1]
            c["levels"].append(v)
            c["avg"] = sum(c["levels"]) / len(c["levels"])
        else:
            clusters.append({"avg": v, "levels": [v]})
    return [c for c in clusters if len(c["levels"]) >= 2]


def liquidity_levels(df, k=3):
    """Equal highs/lows (liquidity pools) and the nearest resting liquidity."""
    highs, lows = detect_swings(df, k=k)
    close = float(df["close"].iloc[-1])
    eq_highs = _clusters([h["price"] for h in highs])
    eq_lows = _clusters([l["price"] for l in lows])

    pools = [{"type": "sell_side", "price": round(c["avg"], 4), "taps": len(c["levels"])}
             for c in eq_highs if c["avg"] < close]  # sell-side resting below
    pools += [{"type": "buy_side", "price": round(c["avg"], 4), "taps": len(c["levels"])}
              for c in eq_highs if c["avg"] > close]
    pools += [{"type": "sell_side", "price": round(c["avg"], 4), "taps": len(c["levels"])}
              for c in eq_lows if c["avg"] < close]
    pools += [{"type": "buy_side", "price": round(c["avg"], 4), "taps": len(c["levels"])}
              for c in eq_lows if c["avg"] > close]

    above = [p for p in pools if p["price"] > close]
    below = [p for p in pools if p["price"] < close]
    nearest = {
        "above": min(above, key=lambda p: p["price"] - close) if above else None,
        "below": max(below, key=lambda p: close - p["price"]) if below else None,
    }
    return {
        "status": "ok",
        "equal_highs": eq_highs, "equal_lows": eq_lows,
        "pools": pools, "nearest_liquidity": nearest,
        "close": close,
    }


# ---------------------------------------------------------------------------
# STEP 3 — ICT
# ---------------------------------------------------------------------------

def fair_value_gaps(df, max_gaps=6):
    """3-candle FVGs, most recent first. A gap is marked mitigated once price
    has traded back through it since formation."""
    fvgs = []
    n = len(df)
    for i in range(2, n):
        hi2, lo2 = float(df["high"].iloc[i - 2]), float(df["low"].iloc[i - 2])
        hi, lo = float(df["high"].iloc[i]), float(df["low"].iloc[i])
        if lo > hi2:  # bullish gap
            fvgs.append({"type": "bullish", "idx": i, "top": lo, "bottom": hi2,
                         "t": str(df["timestamp"].iloc[i])})
        elif hi < lo2:  # bearish gap
            fvgs.append({"type": "bearish", "idx": i, "top": lo2, "bottom": hi,
                         "t": str(df["timestamp"].iloc[i])})
    for g in fvgs:
        since = df.iloc[g["idx"] + 1:]
        g["mitigated"] = bool(
            (since["low"].min() <= g["bottom"] if g["type"] == "bullish" else
             since["high"].max() >= g["top"]).__bool__())
    return fvgs[-max_gaps:][::-1]


def order_blocks(df, move_bars=5, move_pct=1.0, max_blocks=4):
    """Last opposite-color body candle before an impulsive move."""
    blocks = []
    n = len(df)
    for i in range(move_bars, n):
        r = (float(df["close"].iloc[i]) - float(df["close"].iloc[i - move_bars])) \
            / float(df["close"].iloc[i - move_bars]) * 100
        if abs(r) >= move_pct:
            for j in range(i - 1, i - move_bars - 1, -1):
                o, c = float(df["open"].iloc[j]), float(df["close"].iloc[j])
                if (r > 0 and c < o) or (r < 0 and c > o):  # opposite-color body
                    blocks.append({
                        "type": "bullish" if r > 0 else "bearish",
                        "idx": j, "top": max(o, c), "bottom": min(o, c),
                        "t": str(df["timestamp"].iloc[j]),
                        "move_pct": round(r, 2),
                    })
                    break
    return blocks[-max_blocks:][::-1]


def premium_discount(df, window=100):
    """Where price sits inside the recent range (EQ = 50%)."""
    seg = df.iloc[-window:]
    hi, lo = float(seg["high"].max()), float(seg["low"].min())
    close = float(df["close"].iloc[-1])
    if hi - lo <= 0:
        return {"status": "flat"}
    pos = (close - lo) / (hi - lo) * 100
    return {
        "status": "ok",
        "range_high": hi, "range_low": lo, "eq": (hi + lo) / 2,
        "position_pct": round(pos, 1),
        "zone": "PREMIUM" if pos > 50 else "DISCOUNT",
    }


def kill_zone(now_hour_utc):
    """ICT kill zones by UTC hour (informational — not a signal)."""
    london = 7 <= now_hour_utc <= 12
    new_york = 12 <= now_hour_utc <= 16
    if london:
        return "LONDON_KZ (07-12 UTC)"
    if new_york:
        return "NEW_YORK_KZ (12-16 UTC)"
    return "OFF_KILL_ZONE"


def ict_analysis(df, now_hour_utc):
    fvgs = fair_value_gaps(df)
    obs = order_blocks(df)
    pd_ = premium_discount(df)
    return {
        "status": "ok",
        "fair_value_gaps": fvgs,
        "order_blocks": obs,
        "premium_discount": pd_,
        "kill_zone": kill_zone(now_hour_utc),
    }


# ---------------------------------------------------------------------------
# STEP 4 — Wyckoff (heuristic phase detection)
# ---------------------------------------------------------------------------

def wyckoff_analysis(df, lookback=100):
    seg = df.iloc[-lookback:]
    n = len(seg)
    if n < 30:
        return {"status": "insufficient_data"}

    lo = float(seg["low"].min())
    hi = float(seg["high"].max())
    close = float(seg["close"].iloc[-1])
    rng = hi - lo

    # prior trend: slope of closes over the window
    x = np.arange(n)
    slope = np.polyfit(x, seg["close"].values, 1)[0]
    prior_trend = "up" if slope > 0 else "down"

    # volume behaviour: bottom vs top third of the range
    vol = seg["volume"].values.astype(float)
    lows_mask = seg["low"].values <= lo + rng / 3
    highs_mask = seg["low"].values >= hi - rng / 3
    v_bottom = float(vol[lows_mask].mean()) if lows_mask.any() else 0.0
    v_top = float(vol[highs_mask].mean()) if highs_mask.any() else 0.0
    avg_vol = float(vol.mean())

    # range bound?
    rng_pct = rng / lo * 100
    ranged = rng_pct < 25

    # spring: low swept below prior range low then recovered above EQ
    eq = (hi + lo) / 2
    spring = close > eq and lows_mask[-5:].any() and prior_trend == "down"
    utad = close < eq and highs_mask[-5:].any() and prior_trend == "up"

    phase = "UNKNOWN"
    phase_desc = ""
    if not ranged:
        if prior_trend == "up":
            phase, phase_desc = "MARKUP (D-E)", "Impulse above a recent range — SOS behavior."
        else:
            phase, phase_desc = "MARKDOWN (D-E)", "Impulse below a recent range — SOW behavior."
    else:
        if prior_trend == "down":
            if spring:
                phase, phase_desc = "ACCUMULATION (A-C, spring)", \
                    "Downtrend into range; lows swept and price recovered above EQ."
            else:
                phase, phase_desc = "ACCUMULATION (A-B)", \
                    "Downtrend into range; building. Watch a spring at lows."
        else:
            if utad:
                phase, phase_desc = "DISTRIBUTION (A-C, UTAD)", \
                    "Uptrend into range; highs swept and price rejected below EQ."
            else:
                phase, phase_desc = "DISTRIBUTION (A-B)", \
                    "Uptrend into range; selling absorbed. Watch a UTAD at highs."

    # SOS / SOW by breakout on expanding volume
    breakout = None
    last5 = seg.iloc[-5:]
    if last5["close"].max() > hi - 0.05 * rng and vol[-5:].mean() > 1.2 * avg_vol:
        breakout = {"type": "SOS", "level": hi, "vol_expansion": True}
    if last5["close"].min() < lo + 0.05 * rng and vol[-5:].mean() > 1.2 * avg_vol:
        breakout = {"type": "SOW", "level": lo, "vol_expansion": True}

    return {
        "status": "ok",
        "phase": phase,
        "phase_desc": phase_desc,
        "range_pct": round(rng_pct, 1),
        "prior_trend": prior_trend,
        "v_bottom_third": round(v_bottom, 1),
        "v_top_third": round(v_top, 1),
        "avg_volume": round(avg_vol, 1),
        "spring": bool(spring),
        "utad": bool(utad),
        "breakout": breakout,
        "range_high": hi, "range_low": lo,
    }


# ---------------------------------------------------------------------------
# STEP 5 — Volume / Order Flow
# ---------------------------------------------------------------------------

def volume_analysis(df, lookback=100, bins=40):
    seg = df.iloc[-lookback:]
    close = float(df["close"].iloc[-1])

    # Delta from klines (real buyer/seller aggressor volume)
    taker = seg["taker_buy_vol"].astype(float)
    vol = seg["volume"].astype(float)
    seg_delta = (2 * taker - vol)
    cum_delta = float(seg_delta.sum())

    vol_mean = vol.rolling(20).mean()
    spikes = seg[vol > 1.5 * vol_mean].tail(5)
    spike_list = [{
        "idx": int(i), "t": str(row["timestamp"]),
        "vol": round(float(row["volume"]), 0),
        "delta": round(float(seg_delta[i]), 0),
        "dir": "UP" if row["close"] > row["open"] else "DOWN",
    } for i, row in spikes.iterrows()]

    # Volume profile
    hi, lo = float(seg["high"].max()), float(seg["low"].min())
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(seg["low"].values, edges) - 1, 0, bins - 1)
    prof = np.zeros(bins)
    for i in range(len(idx)):
        prof[idx[i]] += vol.iloc[i]
    edges = np.array(edges)
    poc_bin = int(np.argmax(prof))
    poc_price = float((edges[poc_bin] + edges[poc_bin + 1]) / 2)
    mean_bin = float(prof.mean())
    hvn = [(round(float((edges[b] + edges[b + 1]) / 2), 4),
            round(float(prof[b]), 0)) for b in range(bins) if prof[b] > 1.1 * mean_bin]
    lvn = [(round(float((edges[b] + edges[b + 1]) / 2), 4),
            round(float(prof[b]), 0)) for b in range(bins) if 0 < prof[b] < 0.7 * mean_bin]

    # VWAP (session-anchored, last session in window)
    seg2 = df.iloc[-min(lookback, len(df)):]
    tp = (seg2["high"] + seg2["low"] + seg2["close"]) / 3
    vwap_series = (tp * vol).cumsum() / vol.cumsum()
    vwap = float(vwap_series.iloc[-1])
    anchored_vwap = float((tp * vol).sum() / vol.sum())

    return {
        "status": "ok",
        "cum_delta": round(cum_delta, 0),
        "delta_bar_pct": round(float(seg_delta.iloc[-1]), 0),
        "volume_spikes": spike_list,
        "poc": round(poc_price, 4),
        "hvn": hvn, "lvn": lvn,
        "vwap": round(vwap, 4),
        "anchored_vwap": round(anchored_vwap, 4),
        "price_vs_vwap": "ABOVE" if close > vwap else "BELOW",
        "price_vs_poc": "ABOVE" if close > poc_price else "BELOW",
        "close": close,
    }
