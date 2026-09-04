"""Multiple-testing ledger (Phase 2, STEP 17).

Every experiment (feature × symbol × horizon × split) is recorded once,
with its classification. The ledger makes research-selection bias visible:
a "validated" claim must survive the accounting of how many experiments
were run to find it.

Classifications (pre-registered in PHASE2_RESEARCH_AUDIT.md):
  REJECTED       - does not meet the DEV promising rule
  EXPLORATORY    - meets the DEV rule but not further validated
  PROMISING      - meets DEV rule; locked before OOS evaluation
  OOS_VALIDATED  - PROMISING and confirmed on OOS (same sign, |rho| >= 0.02)
  ROBUST         - OOS_VALIDATED + net edge positive on DEV and OOS +
                   regime coverage >= 2 vol regimes

DEV promising rule (locked): |spearman IC| >= 0.02 at H=300s with the same
sign on >=2 of the 3 trade symbols, OR |IC| >= 0.03 on any single symbol.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DEV_IC_THRESHOLD_AGG = 0.02
DEV_IC_THRESHOLD_SINGLE = 0.03
DEV_HORIZON_US = 300_000_000
OOS_IC_THRESHOLD = 0.02

CLASS_REJECTED = "REJECTED"
CLASS_EXPLORATORY = "EXPLORATORY"
CLASS_PROMISING = "PROMISING"
CLASS_OOS_VALIDATED = "OOS_VALIDATED"
CLASS_ROBUST = "ROBUST"


@dataclass
class Experiment:
    feature: str
    symbol: str
    horizon_us: int
    split: str
    n: int
    spearman: float
    q5_q1_gross_bps: float
    q5_q1_net_bps: Optional[float]
    classification: str = ""

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class Ledger:
    def __init__(self, path: Optional[str] = None):
        self.path = path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "PHASE2_EXPERIMENTS.json")
        self.experiments: List[Experiment] = []

    def record(self, e: Experiment) -> None:
        self.experiments.append(e)

    def counts(self) -> dict:
        from collections import Counter

        c = Counter(x.classification for x in self.experiments)
        return {
            "total": len(self.experiments),
            "features": len({x.feature for x in self.experiments}),
            "symbols": sorted({x.symbol for x in self.experiments}),
            "horizons": sorted({x.horizon_us for x in self.experiments}),
            "splits": sorted({x.split for x in self.experiments}),
            "by_classification": dict(c),
        }

    def save(self) -> str:
        payload = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pre_registered": {
                "dev_ic_agg": DEV_IC_THRESHOLD_AGG,
                "dev_ic_single": DEV_IC_THRESHOLD_SINGLE,
                "dev_horizon_us": DEV_HORIZON_US,
                "oos_ic": OOS_IC_THRESHOLD,
                "promising_rule": "|IC|>=0.02 @300s same sign on >=2 symbols "
                                  "OR |IC|>=0.03 on any symbol (DEV only)",
                "quantiles": "fixed quintiles, never selected post-hoc",
            },
            "summary": self.counts(),
            "experiments": [e.as_dict() for e in self.experiments],
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return self.path


def classify_dev(dev_spearman_map: Dict[str, float]) -> Dict[str, str]:
    """DEV-only classification per feature (locked before any OOS look).

    dev_spearman_map: {symbol: spearman IC @ H=300s}.
    Returns {feature_placeholder: classification} — caller assigns to the
    actual feature name.
    """
    out = {}
    sig = {s: r for s, r in dev_spearman_map.items() if abs(r) >= DEV_IC_THRESHOLD_AGG}
    same_sign = len({np_sign(v) for v in sig.values()} - {0}) == 1 and len(sig) >= 2
    single_strong = any(abs(r) >= DEV_IC_THRESHOLD_SINGLE for r in dev_spearman_map.values())
    return {"promising": same_sign or single_strong}


def np_sign(x: float) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)
