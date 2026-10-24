"""Statistical research engine (Phase 2, STEPS 8-10).

Per (feature, symbol, horizon, split) computes:
- sample count, missing count
- mean, std, quantiles (fixed 5,10,25,50,75,90,95)
- Pearson correlation, Spearman correlation (IC) with t-stat
- directional accuracy (sign agreement on nonzero pairs)
- conditional forward returns by feature quintile (Q1..Q5) + Q5-Q1 spread
- monotonicity: Spearman between quintile means and quintile ranks

Correlation is evidence of association, not tradability: every result is
reported together with its cost-adjusted net edge (see report).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import stats as sps


@dataclass
class FeatureStats:
    feature: str
    symbol: str
    horizon_us: int
    split: str
    n: int = 0
    n_missing: int = 0
    mean: float = float("nan")
    std: float = float("nan")
    quantiles: Dict[str, float] = field(default_factory=dict)
    pearson: float = float("nan")
    spearman: float = float("nan")
    spearman_t: float = float("nan")
    directional_accuracy: float = float("nan")
    quintile_means: List[float] = field(default_factory=list)
    quintile_nets: List[float] = field(default_factory=list)
    q5_q1_gross_bps: float = float("nan")
    q5_q1_net_bps: float = float("nan")
    monotonicity: float = float("nan")

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def compute_stats(
    feature_values: np.ndarray,
    labels: np.ndarray,
    net_labels: Optional[np.ndarray] = None,
    feature: str = "",
    symbol: str = "",
    horizon_us: int = 0,
    split: str = "",
) -> FeatureStats:
    s = FeatureStats(feature=feature, symbol=symbol, horizon_us=horizon_us, split=split)
    f = np.asarray(feature_values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    if len(f) != len(y):
        raise ValueError("feature/label length mismatch")
    good = np.isfinite(f) & np.isfinite(y)
    s.n_missing = int((~np.isfinite(f)).sum())
    s.n = int(good.sum())
    if s.n < 10:
        return s
    fg, yg = f[good], y[good]

    s.mean = float(np.nanmean(fg))
    s.std = float(np.nanstd(fg))
    s.quantiles = {
        f"q{int(p*100)}": float(np.quantile(fg, p)) for p in QUANTILES
    }

    if s.n >= 20:
        s.pearson = float(sps.pearsonr(fg, yg)[0])
        rho, pval = sps.spearmanr(fg, yg)
        s.spearman = float(rho)
        if rho < 1.0 and s.n > 2:
            s.spearman_t = float(rho * np.sqrt(s.n - 2) / np.sqrt(1 - rho**2))
        # directional accuracy
        nz = fg != 0
        if nz.any():
            s.directional_accuracy = float(
                np.mean(np.sign(fg[nz]) == np.sign(yg[nz]))
            )

    # fixed quintiles by feature value (boundaries NEVER selected post-hoc)
    q_bounds = np.quantile(fg, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(fg, q_bounds)
    q_means, q_nets = [], []
    for qi in range(5):
        mask = bins == qi
        if mask.any():
            q_means.append(float(np.mean(yg[mask])))
        else:
            q_means.append(float("nan"))
        if net_labels is not None:
            ng = np.asarray(net_labels)[good][mask]
            q_nets.append(float(np.mean(ng)))
    s.quintile_means = q_means
    s.quintile_nets = q_nets
    q1, q5 = q_means[0], q_means[-1]
    if np.isfinite(q1) and np.isfinite(q5):
        s.q5_q1_gross_bps = float((q5 - q1) * 1e4)
    if len(q_nets) == 5 and np.isfinite(q_nets[0]) and np.isfinite(q_nets[-1]):
        s.q5_q1_net_bps = float((q_nets[-1] - q_nets[0]) * 1e4)
    ranks = np.arange(1, 6)
    good_q = np.isfinite(q_means)
    if good_q.sum() >= 3:
        s.monotonicity = float(sps.spearmanr(
            ranks[good_q], np.asarray(q_means)[good_q]
        )[0])
    return s


def table_rows(stats_list: List[FeatureStats]) -> List[dict]:
    return [s.as_dict() for s in stats_list]
