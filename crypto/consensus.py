import numpy as np
import logging

logger = logging.getLogger(__name__)


def multi_timeframe_consensus(symbol, predictor_func, timeframes=None):
    if timeframes is None:
        timeframes = [2, 6, 8, 24]

    results = {}
    for tf in timeframes:
        try:
            result = predictor_func(symbol, tf)
            if result and "error" not in result:
                results[tf] = result
        except Exception as e:
            logger.debug(f"Consensus TF {tf}h failed: {e}")

    if not results:
        return None

    changes = [r["predicted_change_pct"] for r in results.values()]
    confs = [r.get("confidence", 50) for r in results.values()]
    weighted_change = np.average(changes, weights=confs) if sum(confs) > 0 else np.mean(changes)
    avg_conf = np.mean(confs)

    directions = [c > 0 for c in changes]
    agreement = sum(directions) / len(directions)
    consensus_strength = max(agreement, 1 - agreement)

    short_term = [r for tf, r in results.items() if tf <= 6]
    long_term = [r for tf, r in results.items() if tf > 6]
    short_change = np.mean([r["predicted_change_pct"] for r in short_term]) if short_term else 0
    long_change = np.mean([r["predicted_change_pct"] for r in long_term]) if long_term else 0
    trend_aligned = (short_change > 0 and long_change > 0) or (short_change < 0 and long_change < 0)

    if consensus_strength >= 0.75 and avg_conf >= 65:
        consensus_grade = "STRONG"
    elif consensus_strength >= 0.6 and avg_conf >= 50:
        consensus_grade = "MODERATE"
    elif consensus_strength >= 0.5:
        consensus_grade = "WEAK"
    else:
        consensus_grade = "CONFLICTING"

    if not trend_aligned:
        consensus_grade = "DIVERGING"

    return {
        "symbol": symbol,
        "consensus_change_pct": round(weighted_change, 2),
        "consensus_confidence": round(avg_conf, 1),
        "consensus_grade": consensus_grade,
        "agreement_pct": round(consensus_strength * 100, 1),
        "trend_aligned": trend_aligned,
        "timeframes_analyzed": len(results),
        "short_term_change_pct": round(short_change, 2),
        "long_term_change_pct": round(long_change, 2),
        "details": results,
    }


def filter_high_confidence(results, min_confidence=65, min_tfs=2):
    filtered = {tf: r for tf, r in results.items()
                if r.get("confidence", 0) >= min_confidence and "error" not in r}
    if len(filtered) < min_tfs:
        return {}
    return filtered