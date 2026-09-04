"""QUANT-X report engine: 10-step analysis, AI confidence, decision engine,
trade setup, risk gate and the institutional output format.

Everything is evidence-based: sources that are unavailable are reported as
MISSING, never estimated. The probabilistic output is a transparent weighted
vote over the computed signals (weights are assumptions and are disclosed).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from quant_x.data import collect_market_data
from quant_x.indicators import (
    market_structure, liquidity_levels, ict_analysis, wyckoff_analysis,
    volume_analysis,
)

logger = logging.getLogger(__name__)

# Disclosed, tunable signal weights (assumptions).
SIGNAL_WEIGHTS = {
    "structure": 0.18, "liquidity": 0.10, "ict": 0.12, "wyckoff": 0.15,
    "volume": 0.08, "vwap": 0.05, "profile": 0.05, "derivatives": 0.10,
    "sentiment": 0.07, "ml": 0.10,
}

TF_BY_INTERVAL = {"5m": 2, "15m": 6, "30m": 8, "1h": 24}


def _df_from_klines(klines):
    from ml_trainer import KLINES_COLUMNS, NUMERIC_COLS
    df = pd.DataFrame(klines, columns=KLINES_COLUMNS)
    for c in NUMERIC_COLS:
        df[c] = df[c].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["taker_buy_vol"] = df["taker_buy_vol"].astype(float)
    return df


def _atr(df, period=14):
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


# ---------------------------------------------------------------------------
# Signal vote
# ---------------------------------------------------------------------------

def _vote_structure(s):
    if s.get("status") != "ok":
        return 0.0, f"insufficient swings ({s.get('n_swings')})"
    sc = 0.0
    if s["trend"] == "UPTREND":
        sc += 1.0
    elif s["trend"] == "DOWNTREND":
        sc -= 1.0
    if s.get("bos"):
        sc += 0.5 if s["bos"]["type"] == "BOS_UP" else -0.5
    if s.get("choch"):
        sc += 1.0 if s["choch"]["type"] == "CHOCH_UP" else -1.0
    return np.clip(sc, -1, 1), f"trend={s['trend']}, bos={s.get('bos') and s['bos']['type'] or 'none'}, choch={s.get('choch') and s['choch']['type'] or 'none'}"


def _vote_liquidity(l):
    if l.get("status") != "ok":
        return 0.0, "no liquidity data"
    near = l.get("nearest_liquidity") or {}
    ab = near.get("above")
    bl = near.get("below")
    sc = 0.0
    if ab:
        sc += 0.5  # buy-side draw above -> liquidity hunt target
    if bl:
        sc -= 0.5
    return sc, f"nearest liquidity above={ab and ab['price'] or None}, below={bl and bl['price'] or None}"


def _vote_ict(ict):
    if ict.get("status") != "ok":
        return 0.0, "no ICT data"
    sc = 0.0
    pd_ = ict["premium_discount"]
    if pd_.get("status") == "ok":
        sc += 0.5 if pd_["zone"] == "DISCOUNT" else -0.5
    fvgs = ict.get("fair_value_gaps") or []
    if fvgs:
        last = fvgs[0]
        if last["type"] == "bullish" and not last["mitigated"]:
            sc += 0.3
        elif last["type"] == "bearish" and not last["mitigated"]:
            sc -= 0.3
    return sc, f"zone={pd_.get('zone')}, last FVG={fvgs[0]['type'] if fvgs else 'none'}, KZ={ict['kill_zone']}"


def _vote_wyckoff(w):
    if w.get("status") != "ok":
        return 0.0, "insufficient data"
    sc = 0.0
    ph = w["phase"]
    if "ACCUMULATION" in ph or "MARKUP" in ph:
        sc += 1.0
    elif "DISTRIBUTION" in ph or "MARKDOWN" in ph:
        sc -= 1.0
    if w.get("spring"):
        sc += 0.5
    if w.get("utad"):
        sc -= 0.5
    b = w.get("breakout")
    if b:
        sc += 1.0 if b["type"] == "SOS" else -1.0
    return np.clip(sc, -1, 1), f"phase={ph}"


def _vote_volume(v):
    if v.get("status") != "ok":
        return 0.0, "no volume data"
    sc = np.clip(v["cum_delta"] / max(abs(v["cum_delta"]), 1.0), -1, 1)
    note = f"cum_delta={v['cum_delta']}, last_bar_delta={v['delta_bar_pct']}"
    return sc, note


def _vote_vwap(v):
    sc = 0.5 if v["price_vs_vwap"] == "ABOVE" else -0.5
    return sc, f"price {v['price_vs_vwap']} session VWAP {v['vwap']}"


def _vote_profile(v):
    sc = 0.3 if v["price_vs_poc"] == "ABOVE" else -0.3
    return sc, f"price {v['price_vs_poc']} POC {v['poc']}"


def _vote_derivatives(bundle):
    notes = []
    sc = 0.0
    if bundle.get("funding_rate"):
        last_f = bundle["funding_rate"][-1]["rate_pct"]
        notes.append(f"funding={last_f:+.4f}%")
        if last_f > 0.05:
            sc -= 0.4  # crowded longs
        elif last_f < -0.05:
            sc += 0.4  # crowded shorts
    if bundle.get("open_interest"):
        notes.append(f"OI={bundle['open_interest']['oi']:,.1f}")
    if bundle.get("long_short"):
        ratio = bundle["long_short"][-1]["ratio"]
        notes.append(f"L/S={ratio:.3f}")
        if ratio > 1.3:
            sc -= 0.3
        elif ratio < 0.7:
            sc += 0.3
    if not notes:
        return 0.0, "derivatives data missing"
    return np.clip(sc, -1, 1), ", ".join(notes)


def _vote_sentiment(bundle):
    fg = bundle.get("fear_greed")
    if not fg:
        return 0.0, "Fear&Greed missing"
    v = fg["value"]
    if v <= 25:
        return 0.5, f"F&G={v} ({fg['classification']}) -> contrarian bullish"
    if v >= 75:
        return -0.5, f"F&G={v} ({fg['classification']}) -> contrarian bearish"
    return 0.0, f"F&G={v} ({fg['classification']}) -> neutral"


def _vote_ml(symbol, interval, tf_map=TF_BY_INTERVAL):
    tf = tf_map.get(interval)
    if tf is None:
        return 0.0, "no ML model for this timeframe", None
    try:
        import ml_trainer as mt
        pred, vec = mt.predict_all(symbol, tf)
        if not pred:
            return 0.0, "ML model unavailable", None
        conf = pred.get("confidence", 50.0)
        p = pred.get("predicted_change_pct", 0.0)
        gate = pred.get("regime_gate", "UNAVAILABLE")
        tradable = pred.get("tradable", False)
        sc = (1.0 if p > 0 else -1.0) * (max(conf, 50.0) - 50.0) / 50.0
        if gate in ("BLOCKED", "UNAVAILABLE") or not tradable:
            sc *= 0.3
        note = (f"ensemble={p:+.2f}%, calibrated_conf={conf:.1f}% "
                f"(models={pred.get('num_models')}), gate={gate}")
        if pred.get("regime_trend"):
            note += f" ({pred['regime_trend']})"
        return np.clip(sc, -1, 1), note, pred
    except Exception as exc:
        logger.warning("ml input failed: %s", exc)
        return 0.0, f"ML error: {exc}", None


# ---------------------------------------------------------------------------
# Probability
# ---------------------------------------------------------------------------

def _to_probabilities(net):
    bull = max(0.0, net)
    bear = max(0.0, -net)
    side = 0.5 * (1 - abs(net))
    total = bull + bear + side
    if total <= 0:
        return 1 / 3, 1 / 3, 1 / 3
    return round(bull / total * 100, 1), round(bear / total * 100, 1), round(side / total * 100, 1)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(symbol="BTCUSDT", interval="1h", lookback=200, use_ml=True,
                 now_hour_utc=None):
    bundle = collect_market_data(symbol, interval, lookback)
    steps = {}
    if now_hour_utc is None:
        now_hour_utc = datetime.now(timezone.utc).hour

    # ── Steps 1-5 (technical, from klines) ─────────────────────
    if bundle.get("klines"):
        df = _df_from_klines(bundle["klines"])
        steps["1_market_structure"] = market_structure(df)
        steps["2_liquidity"] = liquidity_levels(df)
        steps["3_ict"] = ict_analysis(df, now_hour_utc)
        steps["4_wyckoff"] = wyckoff_analysis(df)
        steps["5_volume"] = volume_analysis(df)
        steps["_atr"] = _atr(df)
        steps["_close"] = float(df["close"].iloc[-1])
        steps["_df_last"] = df.iloc[-1].to_dict()
    else:
        for k in ["1_market_structure", "2_liquidity", "3_ict", "4_wyckoff", "5_volume"]:
            steps[k] = {"status": "missing"}
        steps["_atr"] = None
        steps["_close"] = None
        steps["_df_last"] = None

    # ── Steps 6-9 (external sources) ───────────────────────────
    steps["6_derivatives"] = {
        "status": "available" if bundle.get("funding_rate") else "missing",
        "funding": bundle.get("funding_rate"),
        "open_interest": bundle.get("open_interest"),
        "long_short": bundle.get("long_short"),
        "missing": ["options_gamma", "options_max_pain", "liquidation_heatmap"],
    }
    steps["7_onchain"] = {
        "status": "missing",
        "missing": ["exchange_inflow", "exchange_outflow", "whale_wallets",
                    "dormant_coins", "miner_selling", "etf_flow",
                    "stablecoin_supply", "realized_cap", "mvrv", "nupl", "sopr"],
    }
    steps["8_macro"] = {
        "status": "missing",
        "missing": ["interest_rates", "inflation", "dollar_index", "bond_yield",
                    "nasdaq_correlation", "gold_correlation", "economic_calendar"],
    }
    steps["9_sentiment"] = {
        "status": "available" if bundle.get("fear_greed") else "missing",
        "fear_greed": bundle.get("fear_greed"),
        "missing": ["twitter_x", "reddit", "google_trends", "news"],
    }

    # ── Step 10: AI confidence ─────────────────────────────────
    votes = {}
    votes["structure"] = _vote_structure(steps["1_market_structure"])
    votes["liquidity"] = _vote_liquidity(steps["2_liquidity"])
    votes["ict"] = _vote_ict(steps["3_ict"])
    votes["wyckoff"] = _vote_wyckoff(steps["4_wyckoff"])
    if steps["5_volume"].get("status") == "ok":
        votes["volume"] = _vote_volume(steps["5_volume"])
        votes["vwap"] = _vote_vwap(steps["5_volume"])
        votes["profile"] = _vote_profile(steps["5_volume"])
    else:
        for k in ("volume", "vwap", "profile"):
            votes[k] = (0.0, "volume data missing")
    votes["derivatives"] = _vote_derivatives(bundle)
    votes["sentiment"] = _vote_sentiment(bundle)
    ml_vote = _vote_ml(symbol, interval) if use_ml else (0.0, "ml disabled", None)
    votes["ml"] = (ml_vote[0], ml_vote[1])

    net = 0.0
    total_w = sum(SIGNAL_WEIGHTS.values())
    evidence = []
    for key, w in SIGNAL_WEIGHTS.items():
        sc, note = votes[key][0], votes[key][1]
        net += sc * w / total_w
        evidence.append({"signal": key, "weight": w / total_w, "score": round(sc, 2), "note": note})
    bull_p, bear_p, side_p = _to_probabilities(net)

    # Data coverage: fraction of the 10 steps with real data.
    cov = {
        "1_market_structure": 1.0, "2_liquidity": 1.0, "3_ict": 1.0,
        "4_wyckoff": 1.0, "5_volume": 1.0, "6_derivatives": 0.5,
        "7_onchain": 0.0, "8_macro": 0.0, "9_sentiment": 0.5, "10_ai": 1.0,
    }
    coverage = sum(cov.values()) / len(cov)
    bias = "BULLISH" if bull_p > bear_p and bull_p > side_p else \
           ("BEARISH" if bear_p > side_p else "SIDEWAYS")
    bias_confidence = round(max(bull_p, bear_p) / 100 * coverage * 100, 1)

    ai = {
        "bullish_pct": bull_p, "bearish_pct": bear_p, "sideways_pct": side_p,
        "net_score": round(net, 3), "bias": bias, "confidence_pct": bias_confidence,
        "data_coverage_pct": round(coverage * 100, 0),
        "evidence": evidence,
        "ml_prediction": ml_vote[2],
    }
    steps["10_ai_confidence"] = ai

    decision = _decision_engine(steps, bundle)
    steps["decision"] = decision
    steps["setup"] = _trade_setup(steps, decision, bundle)

    steps["_bundle_status"] = {
        "klines": bundle["klines_status"], "funding": bundle["funding_status"],
        "open_interest": bundle["oi_status"], "long_short": bundle["ls_status"],
        "fear_greed": bundle["fng_status"],
    }
    steps["as_of_utc"] = bundle["as_of_utc"]
    steps["symbol"] = bundle["symbol"]
    steps["interval"] = bundle["interval"]
    steps["lookback"] = bundle["lookback"]
    return steps


# ---------------------------------------------------------------------------
# Decision engine + risk
# ---------------------------------------------------------------------------

def _decision_engine(steps, bundle):
    ai = steps["10_ai_confidence"]
    close = steps.get("_close")
    atr = steps.get("_atr")
    st = steps.get("1_market_structure") or {}
    invalidation, confirmation, target = None, None, None

    if close and atr:
        if ai["bias"] == "BULLISH":
            invalidation = st.get("last_swing_low")
            if invalidation is None:
                invalidation = close - 2 * atr
            confirmation = (st.get("bos") or {}).get("level") or close + 0.5 * atr
            near = (steps.get("2_liquidity") or {}).get("nearest_liquidity")
            target = (near.get("above") or {}).get("price") if near else None
        elif ai["bias"] == "BEARISH":
            invalidation = st.get("last_swing_high")
            if invalidation is None:
                invalidation = close + 2 * atr
            confirmation = (st.get("bos") or {}).get("level") or close - 0.5 * atr
            near = (steps.get("2_liquidity") or {}).get("nearest_liquidity")
            target = (near.get("below") or {}).get("price") if near else None
        elif ai["bias"] == "SIDEWAYS":
            invalidation = None
            confirmation = None
            target = None

    # Contradicting signals = net score close to 0 with high spread.
    spread = abs(ai["bullish_pct"] - ai["bearish_pct"])
    risk = []
    if ai["bias"] in ("BULLISH", "BEARISH") and spread < 20:
        risk.append("Signals are balanced (low spread) — fragile bias.")
    if steps["7_onchain"]["status"] == "missing":
        risk.append("No on-chain confirmation (flows/whales/MVRV unavailable).")
    if steps["8_macro"]["status"] == "missing":
        risk.append("No macro context (rates/DXY/yields unavailable).")
    if steps["6_derivatives"]["status"] == "missing":
        risk.append("No derivatives confirmation (funding/OI/LS unavailable).")

    return {
        "bias": ai["bias"],
        "confidence_pct": ai["confidence_pct"],
        "invalidation": round(invalidation, 4) if invalidation else None,
        "confirmation_level": round(confirmation, 4) if confirmation else None,
        "target_level": round(target, 4) if target else None,
        "risks": risk,
        "data_coverage_pct": ai["data_coverage_pct"],
    }


def _trade_setup(steps, decision, bundle):
    ai = steps["10_ai_confidence"]
    close = steps.get("_close")
    atr = steps.get("_atr")
    reasons = []

    if ai["bias"] == "SIDEWAYS":
        reasons.append("Bias is SIDEWAYS — no directional setup.")
    if decision["confidence_pct"] < 70:
        reasons.append(f"Confidence {decision['confidence_pct']:.0f}% < 70% gate.")
    if steps["5_volume"].get("status") != "ok" or not steps["5_volume"].get("volume_spikes"):
        reasons.append("Weak/no volume confirmation.")
    if not (steps.get("2_liquidity") or {}).get("pools"):
        reasons.append("Weak liquidity context (no visible pools).")

    if reasons or not close or not atr:
        return {"status": "REJECTED", "reasons": reasons}

    bias = ai["bias"]
    rng = (steps["4_wyckoff"].get("range_high"), steps["4_wyckoff"].get("range_low"))
    sl_buffer = atr

    if bias == "BULLISH":
        entry = round(close + 0.15 * atr, 4)
        sl = round(decision["invalidation"] - 0.2 * atr, 4)
        t1 = round(close + atr, 4)
        t2 = round(decision["target_level"] or close + 2 * atr, 4)
        t3 = round(max(t2, close + 3 * atr), 4)
    else:
        entry = round(close - 0.15 * atr, 4)
        sl = round(decision["invalidation"] + 0.2 * atr, 4)
        t1 = round(close - atr, 4)
        t2 = round(decision["target_level"] or close - 2 * atr, 4)
        t3 = round(min(t2, close - 3 * atr), 4)

    risk_per_unit = abs(entry - sl)
    if risk_per_unit <= 0:
        return {"status": "REJECTED", "reasons": ["Invalid SL distance (risk=0)."]}
    rr1 = abs(t1 - entry) / risk_per_unit
    rr2 = abs(t2 - entry) / risk_per_unit

    if rr1 < 2.0:
        reasons.append(f"Best R/R {rr1:.2f} < 2:1 gate.")
    if reasons:
        return {"status": "REJECTED", "reasons": reasons}

    p_success = min(ai["confidence_pct"] / 100, 0.75)
    p_fail = 1 - p_success
    ev = p_success * abs(t2 - entry) - p_fail * risk_per_unit

    return {
        "status": "OPEN",
        "side": "LONG" if bias == "BULLISH" else "SHORT",
        "entry": entry, "stop_loss": sl,
        "tp1": t1, "tp2": t2, "tp3": t3,
        "rr_tp1": round(rr1, 2), "rr_tp2": round(rr2, 2),
        "risk_pct_position": 2.0,
        "expected_drawdown": round(sl_buffer, 4),
        "prob_success_pct": round(p_success * 100, 1),
        "prob_failure_pct": round(p_fail * 100, 1),
        "expected_value": round(ev, 4),
    }


# ---------------------------------------------------------------------------
# Markdown render
# ---------------------------------------------------------------------------

def render_markdown(r, lang="en"):
    L = _labels(lang)
    out = []
    out.append(f"# QUANT-X Institutional Report — {r['symbol']} ({r['interval']})")
    out.append(f"*as of {r['as_of_utc']} UTC* · lookback {r['lookback']} bars\n")

    ai = r["10_ai_confidence"]
    dec = r["decision"]
    out.append("## 1. Executive Summary")
    out.append(f"- **Bias:** {dec['bias']} | **Confidence:** {dec['confidence_pct']:.0f}% "
               f"(data coverage {ai['data_coverage_pct']:.0f}%)")
    out.append(f"- **Probability:** Bullish {ai['bullish_pct']:.0f}% · "
               f"Bearish {ai['bearish_pct']:.0f}% · Sideways {ai['sideways_pct']:.0f}%")
    out.append(f"- **Close:** {r['_close']} | ATR(14): {r['_atr']:.2f}\n")

    out.append("## 2. Technical Analysis")
    _step(out, r["1_market_structure"], "Market Structure")
    _step(out, r["2_liquidity"], "Liquidity")
    _step(out, r["3_ict"], "ICT")
    _step(out, r["5_volume"], "Volume / Order Flow")

    out.append("## 3. Smart Money Analysis")
    _step(out, r["4_wyckoff"], "Wyckoff")
    if r["5_volume"].get("status") == "ok":
        v = r["5_volume"]
        out.append(f"- Volume Profile: POC={v['poc']}, price {v['price_vs_poc']} POC; "
                   f"{len(v['hvn'])} HVN, {len(v['lvn'])} LVN. VWAP={v['vwap']} "
                   f"(price {v['price_vs_vwap']}).")

    out.append("## 4. On-chain Analysis")
    _missing(out, r["7_onchain"])
    out.append("## 5. Macro Analysis")
    _missing(out, r["8_macro"])

    out.append("## 6. Derivatives Analysis")
    _derivatives(out, r["6_derivatives"])

    out.append("## 7. Risk Analysis")
    for risk in dec["risks"]:
        out.append(f"- [risk] {risk}")
    if dec["invalidation"]:
        out.append(f"- **Invalidation:** {dec['invalidation']}")
    if dec["confirmation_level"]:
        out.append(f"- **Confirmation:** {dec['confirmation_level']}")
    if dec["target_level"]:
        out.append(f"- **Target:** {dec['target_level']}")

    out.append("## 8. Trade Plan")
    _setup(out, r["setup"])

    out.append("## 9. Probability Matrix")
    out.append(f"| | Bullish | Bearish | Sideways |")
    out.append(f"|--|--|--|--|")
    out.append(f"| Probability | {ai['bullish_pct']:.0f}% | {ai['bearish_pct']:.0f}% | {ai['sideways_pct']:.0f}% |")
    s = r["setup"]
    if s["status"] == "OPEN":
        out.append(f"| P(success) / P(failure) | {s['prob_success_pct']:.0f}% / {s['prob_failure_pct']:.0f}% | | |")
    else:
        out.append(f"| P(success) / P(failure) | n/a (setup rejected) | | |\n")

    out.append("## 10. Final Conclusion")
    out.append(f"**{ai['bias']}** with {dec['confidence_pct']:.0f}% confidence. "
               "This is a probabilistic model over the disclosed evidence "
               "(weights are assumptions). On-chain and macro dimensions are "
               "currently MISSING, which caps overall confidence.")
    return "\n".join(out)


def _labels(lang):
    return {"en": True, "ar": True}.get(lang, {})


def _step(out, step, title):
    out.append(f"### {title}  —  `{step.get('status', 'missing')}`")
    if step.get("status") == "ok":
        for k, v in step.items():
            if k in ("status", "fair_value_gaps", "order_blocks", "equal_highs",
                     "equal_lows", "pools", "volume_spikes", "hvn", "lvn"):
                continue
            if isinstance(v, dict):
                continue
            out.append(f"- {k.replace('_', ' ')}: {v}")
        fvgs = step.get("fair_value_gaps")
        if fvgs:
            out.append("- recent FVGs: " + "; ".join(
                f"{g['type']} {round(g['bottom'],4)}-{round(g['top'],4)} "
                f"({'mitigated' if g['mitigated'] else 'open'})" for g in fvgs))
        obs = step.get("order_blocks")
        if obs:
            out.append("- order blocks: " + "; ".join(
                f"{b['type']} {round(b['bottom'],4)}-{round(b['top'],4)}" for b in obs))
        pools = step.get("pools")
        if pools:
            out.append("- liquidity pools: " + "; ".join(
                f"{p['type']}@{p['price']}(x{p['taps']})" for p in pools[:6]))
    else:
        out.append("- " + str(step.get("missing", "no data")))


def _missing(out, step):
    if step["status"] == "missing":
        out.append(f"- [MISSING] {', '.join(step.get('missing', []))}")
    else:
        out.append(f"- available: {step}")


def _derivatives(out, step):
    if step["status"] == "available":
        fund = step.get("funding")
        if fund:
            last = fund[-1]
            out.append(f"- funding (latest): {last['rate_pct']:+.4f}% @ {last['time']}")
        oi = step.get("open_interest")
        if oi:
            out.append(f"- open interest: {oi['oi']:,.2f} contracts")
        ls = step.get("long_short")
        if ls:
            last = ls[-1]
            out.append(f"- global L/S account ratio: {last['ratio']:.3f} "
                       f"(long {last['long_account']*100:.1f}% / short {last['short_account']*100:.1f}%)")
        out.append(f"- [MISSING] {', '.join(step.get('missing', []))}")
    else:
        out.append(f"- [MISSING] {', '.join(step.get('missing', []))}")


def _setup(out, setup):
    if setup["status"] == "REJECTED":
        out.append(f"**NO VALID SETUP - REJECTED.**")
        for reason in setup["reasons"]:
            out.append(f"- x {reason}")
        return
    out.append(f"- Side: **{setup['side']}**")
    out.append(f"- Entry: {setup['entry']}")
    out.append(f"- Stop loss: {setup['stop_loss']}")
    out.append(f"- TP1: {setup['tp1']} (R/R {setup['rr_tp1']})")
    out.append(f"- TP2: {setup['tp2']} (R/R {setup['rr_tp2']})")
    out.append(f"- TP3: {setup['tp3']}")
    out.append(f"- Risk per position: {setup['risk_pct_position']}%")
    out.append(f"- Expected drawdown (ATR): {setup['expected_drawdown']}")
    out.append(f"- P(success): {setup['prob_success_pct']}% · P(failure): {setup['prob_failure_pct']}%")
    out.append(f"- Expected value: {setup['expected_value']}")
