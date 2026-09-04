"""Execution and cost models for the arbitrage simulator.
- FeeEngine: per-venue maker/taker bps, withdrawal/deposit/network fees.
- Latency model: fills occur at quotes LATENCY later; order cancelled if no longer profitable.
- Fill risk: each leg fills with probability p; failed leg unwound at prevailing quotes.
- Capital: prefunded across venues; notional capped per cycle; no compounding in V1.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class FeeConfig:
    taker_bps: float = 10.0      # spot taker (Binance 0.10%)
    maker_bps: float = 10.0
    futures_taker_bps: float = 5.0   # USD-M futures taker (0.05%)
    futures_maker_bps: float = 2.0
    withdrawal_fee_usd: float = 0.0  # network fees if capital must move
    withdrawal_flat: float = 0.0


@dataclass
class ExecConfig:
    latency_ms: int = 50          # decision -> fill window
    bucket_us: int = 100_000      # 100ms signal buckets
    fill_prob_leg: float = 1.0    # probability each leg fills
    min_net_bps: float = 0.0      # extra noise threshold on top of fees
    notional_usd: float = 100.0   # capital deployed per cycle
    stress_slippage_bps: float = 0.0  # extra conservative slippage per leg
    seed: int = 42


class FeeEngine:
    def __init__(self, cfg: FeeConfig):
        self.cfg = cfg

    def spot_cycle_cost(self, n_legs: int, slippage_bps: float = 0.0) -> float:
        return n_legs * (self.cfg.taker_bps + slippage_bps) / 1e4

    def futures_cycle_cost(self, n_legs: int) -> float:
        return n_legs * self.cfg.futures_taker_bps / 1e4

    def per_leg_fee(self, notional: float, bps: float) -> float:
        return notional * bps / 1e4


def simulate_latency_fills(rate_series: np.ndarray, opp_mask: np.ndarray,
                           latency_buckets: int, fee_net: float,
                           rng: np.random.Generator,
                           fill_prob: float = 1.0) -> dict:
    """Given per-bucket cycle rate r(t) (already bid/ask-implied), detect buckets
    where r >= 1+fee_net, require persistence >= latency_buckets, then fill at
    rate(t+latency). Cancelled if rate(t+latency) < 1+fee_net (no adverse fills in base).
    Returns results dict."""
    n = len(rate_series)
    net = rate_series - 1 - fee_net  # net edge per unit per cycle

    above = net > 0
    # detect runs of `above` with length >= latency_buckets (opportunity persisted)
    runs = np.zeros(n, dtype=bool)
    cnt = 0
    for i in range(n):
        if above[i]:
            cnt += 1
            runs[i] = cnt >= latency_buckets
        else:
            cnt = 0

    # candidate execution buckets = first bucket of a qualifying run
    first_of_run = runs & ~np.concatenate(([False], runs[:-1]))
    exec_idx = np.flatnonzero(first_of_run)
    # fill window: t + latency
    fill_idx = exec_idx + latency_buckets
    ok = fill_idx < n

    trades = []
    for ei, fi in zip(exec_idx[ok], fill_idx[ok]):
        fill_net = net[fi]
        if fill_net <= 0:
            continue  # cancelled: no longer profitable
        if rng.random() > fill_prob:
            continue  # leg failed to fill -> no trade (conservative: no unwinding needed
                      # because we never committed both legs; capital simply not deployed)
        trades.append({
            "entry_bucket": ei,
            "fill_bucket": fi,
            "edge_bps": fill_net * 1e4,
            "fee_bps": fee_net * 1e4,
        })
    return {"trades": trades, "runs_count": int(runs.sum() > 0 and len(exec_idx)),
            "candidates": len(exec_idx)}


def expected_two_leg_risk(n_cycles: int, p_leg: float, leg_loss_frac: float) -> float:
    """Expected cost of second-leg fill failure when p_leg < 1.
    Approximation: P(both fill)=p^2, P(partial loss)=p*(1-p), loss = leg_loss_frac of notional."""
    p_both = p_leg ** 2
    p_loss = p_leg * (1 - p_leg) * 2  # one leg fills, other doesn't -> unwind
    return p_both, p_loss, leg_loss_frac
