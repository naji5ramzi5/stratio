"""Conservative master-decision gate for research and paper-trading signals.

This module does not execute orders. It turns the independent bot outputs into
an explicit, auditable decision and makes uncertainty a first-class outcome.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Decision:
    decision: str  # TRADE_READY | ALERT_ONLY | NO_TRADE
    direction: str  # LONG | SHORT | NEUTRAL
    score: int
    reasons: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_signal(prediction: dict, liquidity: dict, flow: dict,
                    ai_score: float, *, ai_threshold: float = 50.0,
                    min_probability: float = 0.55,
                    min_probability_gap: float = 0.10,
                    max_expected_vol_pct: float = 12.0) -> Decision:
    """Return an explainable decision without inventing certainty.

    An uncalibrated forecast can be useful as an alert, but can never be marked
    ``TRADE_READY``. Conflicting order-flow/liquidity evidence is a no-trade.
    """
    up = float(prediction.get("prob_up", 0.0) or 0.0)
    down = float(prediction.get("prob_down", 0.0) or 0.0)
    neutral = float(prediction.get("prob_neutral", 0.0) or 0.0)
    expected_vol = float(prediction.get("expected_vol_pct", 0.0) or 0.0)
    imbalance = float(liquidity.get("imbalance_pct", 0.0) or 0.0)
    buy_pct = float(flow.get("buy_pct", 50.0) or 50.0)

    direction = "LONG" if up > down else "SHORT" if down > up else "NEUTRAL"
    dominant_probability = max(up, down)
    probability_gap = abs(up - down)
    reasons: list[str] = []
    conflicts: list[str] = []

    if direction == "NEUTRAL" or neutral >= dominant_probability:
        reasons.append("neutral_probability_dominant")
    if dominant_probability < min_probability:
        reasons.append(f"probability_below_threshold={dominant_probability:.2f}")
    if probability_gap < min_probability_gap:
        reasons.append(f"directional_probability_gap_too_small={probability_gap:.2f}")
    if ai_score < ai_threshold:
        reasons.append(f"ai_score_below_threshold={ai_score:.1f}")
    if expected_vol > max_expected_vol_pct:
        reasons.append(f"expected_volatility_too_high={expected_vol:.2f}%")
    if liquidity.get("spoofing_detected"):
        reasons.append("spoofing_detected")

    liquidity_direction = "LONG" if imbalance > 15 else "SHORT" if imbalance < -15 else "NEUTRAL"
    flow_direction = "LONG" if buy_pct > 52 else "SHORT" if buy_pct < 48 else "NEUTRAL"
    for label, evidence_direction in (("liquidity", liquidity_direction), ("order_flow", flow_direction)):
        if evidence_direction != "NEUTRAL" and evidence_direction != direction:
            conflicts.append(f"{label}_opposes_{direction.lower()}")

    # This score is a decision-audit score, not a probability or confidence.
    score = round(100 * dominant_probability)
    score -= min(30, len(reasons) * 12)
    score -= min(30, len(conflicts) * 15)
    score = max(0, min(100, score))

    if reasons or conflicts:
        return Decision("NO_TRADE", direction, score, reasons, conflicts)
    if not prediction.get("confidence_available", False):
        return Decision("ALERT_ONLY", direction, score,
                        ["prediction_probability_not_oos_calibrated"], conflicts)
    return Decision("TRADE_READY", direction, score, reasons, conflicts)
