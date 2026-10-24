# STRATO CRYPTO — PHASE 1: FULL SYSTEM AUDIT

> Methodology: honest, evidence-based. Every claim references a file:line.
> No performance fabricated. Where edge is unproven, it is flagged as such.
> Date of audit: 2026-08-22. Codebase: StratoCrypto (single monorepo).

---

## 0. ARCHITECTURE REALITY CHECK

The system is a **single Python monorepo**. The "5 bots" are NOT 5 separate Telegram
bots — they are functions inside `crypto/advanced_bot.py` plus a separate
`crypto/news_bot.py` (Bot 1). The spec's numbering maps to code as:

| Spec bot | Code reality | Primary files |
|---|---|---|
| 1 News / Market Intel | Rule-based NLP news aggregator | `news_provider.py`, `news_agent.py`, `news_bot.py` |
| 2 Prediction / ML | Multi-model direction ensemble | `predictor.py`, `ml_trainer.py`, `prediction_tracker.py`, `meta_labeling.py`, `foundation_models.py` |
| 3 Signal | Hand-weighted AI score | `advanced_bot.py:compute_ai_score` (L536), `compute_trade_setup` (L610) |
| 4 Liquidity / Microstructure | Order-book/derivatives analytics | `advanced_bot.py:analyze_liquidity` (L418), `analyze_derivatives` (L393), `analyze_order_flow` (L363) |
| 5 Master / Decision | 3-bot AND-gate convergence | `advanced_bot.py:run_master_scan` (L1313) |

**Critical structural gaps (must fix before claiming "production"):**

1. **News is siloed.** `news_bot.py` runs as its own process and broadcasts alerts,
   but its output is **never an input** to `run_master_scan`. The Master bot consumes
   AI score + liquidity + ML prediction only. Spec requires NEWS as a Master input.
2. **Risk engine is bypassed in live trading.** `risk_engine.py` (a well-designed
   module: position limits, drawdown halts, Kelly, correlation, circuit breaker) is
   imported ONLY by `live_paper.py` and `run_full_system.py` — NOT by
   `advanced_bot.py` (the actual live bot). The live bot uses a separate, weaker
   inline `compute_risk_level`. So live trades do not pass through the real risk gate.
3. **Two competing risk systems.** Inline `compute_risk_level` (advanced_bot) vs
   standalone `risk_engine.py`. Fragmentation → inconsistent enforcement.
4. **Duplication everywhere.** 3+ backtesters (`backtester.py`, `fast_backtest.py`,
   `backtest/*`), 10+ ML model trainers (`ml_trainer.py`, `train.py`, `models/*`,
   `foundation_models.py`), multiple RL/arbitrage frameworks. High maintenance risk,
   unclear which is "the" system.

---

## 1. BACKTESTING FRAMEWORK (cross-cutting)

**`crypto/backtester.py`** (stat-arb pair mean-reversion):
- Has fee (10 bps) + slippage (5 bps) config and per-trade Sharpe. Good intent.
- **BUG — cost math is broken.** `costs = total_cost_fraction * notional / 100`,
  then `net_pnl = raw_pnl - costs / notional`. Units are mixed: `raw_pnl` is a
  dimensionless log-spread delta, while `costs/notional` ≈ 0.00006 (≈0.006%),
  i.e. costs are effectively **zero** relative to signal. Real round-trip cost should
  be ~0.6% (4 fills × (0.1% fee + 0.05% slip)). This understates costs by ~100× and
  makes any "profitable" result meaningless.
- **Fake stress test.** `stress_test()` (L298) does NOT re-run with stressed params;
  it multiplies Sharpe by 0.6 / 0.3 fudge factors. Not a real stress test.
- **No walk-forward.** Demo trains on first 300 bars, tests last 200 (single split),
  despite docstring claiming walk-forward. No rolling window, no OOS period.
- **No funding / latency / partial fills** modeled.

**`crypto/backtest/multi_asset_backtester.py`** (L33, L109): uses
`position.shift(1) * actual_return` — **correct point-in-time pattern** (no
look-ahead from position). Better foundation than `backtester.py`. But fee inclusion
must be verified (only slippage shown at L35).

**Verdict on backtesting:** Not trustworthy yet. Costs understated in primary
backtester; no real WF/OOS/Monte Carlo harness exists as a single source of truth.

---

## 2. NEWS INTELLIGENCE (Bot 1)

Strengths:
- Multi-source aggregation: Google News RSS (keyless), CoinDesk/Cointelegraph/Decrypt
  RSS, official RSS (Fed/SEC/Treasury/WhiteHouse/CFTC/ECB), NewsAPI.org.
- Rule-based classification: category, asset detection, sentiment lexicon, impact &
  confidence scoring, event clustering, price coupling.

Weaknesses (per spec requirements):
- **Official RSS feeds return 404/403** (WhiteHouse/Treasury endpoints dead) →
  "official government sources" requirement is NOT actually met.
- **No verification tier** (VERIFIED / SINGLE SOURCE / UNVERIFIED) as specified.
- **No deduplication** of repeated stories across 10 sites.
- **No market-reaction engine** (BTC ±1m/5m/15m/1h post-event). Correlation vs
  causation not separated.
- **Siloed from Master** (see §0.1) — news never influences trading decisions.
- No BREAKING alert format with IMPACT score / CONFIDENCE / sources layout.

Verdict: 🟡 NEEDS MORE VALIDATION. Functionally useful as an alert feed, but does not
meet the spec's verification/dedup/market-reaction/White-House-linkage requirements.

---

## 3. ML PREDICTION (Bot 2)

Strengths:
- Large feature set (price action, vol, volume, RSI/MACD/ATR, funding, OI,
  liquidations, order-book imbalance, dominance, news sentiment).
- Ensemble: MLP+RF+GBR+XGB+LR; calibration + meta-labeling present; online learning.

Risks (need verification in Phase 2):
- **OOS edge unproven.** No walk-forward/OOS report exists. Online learning on a
  small verified sample can overfit.
- **Possible look-ahead** in label construction (`ml_trainer.py`) — must be audited
  with a point-in-time leakage test (a `tests/` leakage test exists; must be run).
- Calibration (Brier/calibration error) claimed but not yet validated with numbers.
- Foundation models (Chronos-2/TimesFM) are zero-shot and uncalibrated for crypto.

Verdict: 🟠 EXPERIMENTAL. Promising scaffolding, but edge is unproven until OOS +
calibration + leakage tests are run and reported.

---

## 4. SIGNAL ENGINE (Bot 3)

- `compute_ai_score` (L536) = hand-weighted sum: Liquidity 25 + Technical 25 +
  Order Flow 25 + Derivatives 25 + MTF bonus 5. Weights are **arbitrary**, not
  derived from data. No backtest shows these weights produce edge.
- `confidence = score * 0.95` is a **fake confidence formula** (not calibrated).
- `compute_trade_setup` (L610): fixed ATR multiples (SL 1.5×, TP 1.5/3/5×). No
  volatility-regime adaptation beyond ATR.
- No explicit signal-quality thresholds (0–49 NO TRADE … 85–100 VERY STRONG) frozen
  from training data.

Verdict: 🟠 EXPERIMENTAL. Rule-of-thumb scoring, not validated.

---

## 5. LIQUIDITY / MICROSTRUCTURE (Bot 4)

Strengths:
- Rich features: order-book imbalance, spoofing detection, bid/ask walls, CVD,
  funding rate, long/short ratio, open interest, whale detection.

Weaknesses:
- These are **analytics, not a backtested signal**. No P&L attributed to
  "buy pressure / short squeeze / liquidity vacuum / absorption" states.
- `smart_money_score` used in AI score — its derivation not audited.
- No liquidity-pressure score (-100→+100) or microstructure confidence (0→100) as spec.

Verdict: 🟡 NEEDS VALIDATION. Best-engineered bot, but unproven as a trade signal.

---

## 6. MASTER / DECISION + RISK (Bot 5)

- `run_master_scan` (L1313) is a **hard AND-gate**: ai_score ≥ threshold AND
  liquidity present AND predicted_change ≥ 3%. NOT a weighted ensemble of
  NEWS+ML+SIGNAL+LIQUIDITY+REGIME+RISK.
- **No conflict detection.** If ML says down but liquidity says up, it simply fails
  the AND-gate (no-trade by accident, not by design).
- **No explicit no-trade regime logic** (extreme vol / low liquidity / major-event
  uncertainty).
- **Risk gate bypassed** (see §0.2): uses inline `compute_risk_level`, not the real
  `risk_engine.py`. No position sizing from equity/vol/DD; no cooldown after losses.
- News is not an input at all.

Verdict: 🟠 EXPERIMENTAL / ARCHITECTURALLY INCOMPLETE. The most important layer per
spec is the weakest vs. requirements.

---

## 7. INFRASTRUCTURE

- Telegram: `advanced_bot.py` uses `telebot`; `news_bot.py` uses pyTelegramBotAPI.
  Separate tokens/processes. News process is not even launched by
  `start_bots.bat` / `launch_bots.py` (references a non-existent `kimi_bot.py`).
- Data: Binance REST (spot + futures) via `get_binance`. No websocket; polling only.
- Scheduling: `while True` loops with `time.sleep` inside `advanced_bot.py`;
  `dashboard_server.py` uses APScheduler. No unified scheduler.
- Storage: JSON files (`prediction_history.json`, `pairs_db.json`), pickle models.
  No DB. Data-quality guards exist in `market_events/` (point-in-time replay) but are
  not wired into live bots.
- Tests: `crypto/tests/` has 30+ pytest files (calibration, leakage, risk,
  integration). Good — but not yet run as a gate.

---

## 8. BOT SCORECARD (initial, pre-fix)

Scores are /100. These are AUDIT judgements, not measured performance.

| Bot | Robustness | Profitability* | Reliability | Data Quality | Prod Ready | Verdict |
|---|---|---|---|---|---|---|
| 1 News | 50 | N/A (alert only) | 60 | 55 | 45 | 🟡 NEEDS VALIDATION |
| 2 ML | 40 | UNPROVEN | 50 | 60 | 35 | 🟠 EXPERIMENTAL |
| 3 Signal | 35 | UNPROVEN | 45 | 55 | 30 | 🟠 EXPERIMENTAL |
| 4 Liquidity | 55 | UNPROVEN | 55 | 60 | 45 | 🟡 NEEDS VALIDATION |
| 5 Master/Risk | 30 | UNPROVEN | 40 | 50 | 25 | 🟠 INCOMPLETE |

*Profitability = UNPROVEN because no trustworthy OOS backtest exists yet.

---

## 9. TOP PRIORITIES (ordered by impact)

P1. Fix `backtester.py` cost math + real stress test + single WF/OOS harness.
P2. Wire `risk_engine.py` + `execution_engine.py` into `advanced_bot` live path.
P3. Make News an input to Master (verification + dedup + market-reaction + White
    House/official linkage with BREAKING alerts).
P4. Replace arbitrary AI-score weights with data-derived, frozen thresholds (OOS).
P5. Run leakage/calibration/OOS tests as a gate; produce honest reports.
P6. Consolidate duplicated backtesters/models into one source of truth.

> No strategy here is claimed profitable. Several are EXPERIMENTAL until OOS-validated.
> The system is a strong RESEARCH/ALERT scaffold, not yet a production trading system.
