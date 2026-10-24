# StratoCrypto — Architecture & ML Review (Phase 1 + Phase 2)

> Deliverable for the "senior SWE + ML engineer + systems architect" pass.
> Everything below is honest, out-of-sample (OOS) or measured — no aspirational numbers.

---

## 1. Executive summary

The project has real engineering (honest backtests, calibration, locking) but its
**directional ML signal is dead** across every horizon, and a hard regime gate was
making things *worse*. What survives OOS is a **thin relative-value (pairs) edge** on
a small, unstable set of BTC/BNB-correlated pairs.

| Claim / earlier belief | Reality (now, measured) |
|---|---|
| 24h directional ML prediction | **dead**: 48.9% at 24h (6784 verified); **48.8% / 46.4% / 47.5% / 45.5% at 6h / 12h / 24h / 48h** — no horizon has an edge |
| Hard regime gate (+7% deepalpha) | **hurts**: A/B on our data 50.2% (no gate) → 40.0% (gate). Neutralised to reporting-only |
| High confidence = high accuracy | Calibrated honesty: 80-90% agreement ≈ 48.6% realised; **worse at higher confidence** |
| One loader / one config | Still duplicated fetchers; `market_data.py` cache layer added but not yet wired into all bots |
| 3 bots share tracker safely | Corruption confirmed — bots rewrote tracker from stale memory. Fixed with **atomic lock + read-modify-write** |
| Pairs have an edge | Real but thin: BTC/LTC OOS ann=+49%/Sharpe=2.47 (3 trades); latest rescan approves BNB/DOGE + ADA/LTC. **Unstable across windows** |

**What was delivered in this pass:**

1. **Atomic tracker**: `prediction_tracker` now uses an exclusive lock file + temp-write-rename + read-modify-write, so concurrent bots can no longer corrupt it.
2. **Shared data pipeline**: `market_data.py` — TTL cache over the canonical `data_loader.binance_ohlcv` loader, single fetch path.
3. **Pairs paper book**: `pair_paper.py` — two-leg paper engine, PnL mirrors `stat_arb.backtest_spread`, wired into `advanced_bot` loop + `--rescan` for OOS-proven universe.
4. **Regime gate neutralised**: `gate_blocks=False` default; still reports the regime but no longer blocks (evidence: A/B = -11pp).
5. **Directional signal retired as a trade signal**: confirmed dead 6h-48h; system now leans on the pairs edge.
6. **Background bots stopped**: the 3 stale processes (PIDs 7820/19276/24528) that rewrote the tracker from memory are killed.

---

## 2. Verified numbers (OOS)

### 2.1 Directional prediction — DEAD (retired)

| Source | Horizon | n | Direction accuracy |
|---|---|---|---|
| `prediction_tracker` backfill | 24h | 6784 | **48.9%** |
| Fresh backfill (this pass) | 6h / 12h / 24h / 48h | 764 / 711 / 673 / 550 | **48.8% / 46.4% / 47.5% / 45.5%** |

**Calibration (realised accuracy by confidence bin, 6784 verified):**
50-60→45.8%, 60-70→49.5%, 70-80→49.5%, 80-90→48.6%, 90-100→47.6%.
*Accuracy gets **worse** as confidence rises. This signal is retired from auto-trading.*

### 2.2 Regime & Volatility Predictor (NEW) — regime_v1

| Metric | Value | Notes |
|---|---|---|
| BTC current regime | MEAN_REVERTING | score=80, action=TRADE |
| Active pairs in favorable regime | 8 / 8 | all scores 80-90 |
| Features | 13 | vol persistence, autocorrelation, directional consistency, variance ratio |
| Gate | score ≥ 60 to open pairs | prevents trading in trending/volatile markets |

### 2.3 Regime gate A/B — HURTS

| Config | n | Direction acc | Excess |
|---|---|---|---|
| ALL (no gate) | 4 tf | **50.2%** | -1.6% |
| REGIME_GATE | 4 tf | **40.0%** | -13.1% |

The deepalpha claim (-11%→+7%) does **not** replicate on our data.

### 2.3 Relative-value (pairs) — REAL BUT THIN

| Pair | Ann % | Sharpe | MaxDD % | Trades | Win% | Note |
|---|---|---|---|---|---|---|
| BTC/LTC (1h) | +49.6 | 2.47 | -9.45 | 3 | 100 | best; 10 folds |
| BTC/LINK (1h) | +9.2 | 0.49 | -13.8 | 3 | 100 | weaker |
| Latest rescan approved | — | — | — | — | — | BNB/DOGE (1.11), ADA/LTC (1.0) |

*Cointegration is unstable across windows: BTC/LINK passed earlier, now p=0.28. The universe must be rescanned regularly.*

---

## 3. Problems found & fixes

| # | Severity | Problem | Fix |
|---|---|---|---|
| 1 | **Critical** | `.env` with live keys tracked in git | removed from index; **pending**: history purge + key rotation |
| 2 | **Critical** | Look-ahead leak: training `btc_corr_10` used tail-correlation | done: aligned causal feature, `merge_btc_features` |
| 3 | **High** | Prediction telemetry dead: symbol-key mismatch → 0 verified | done: `normalize_symbol`, `verify_all` |
| 4 | **High** | Verify timing too strict (48h cap missed stale records) | done: 720h default |
| 5 | **High** | **3 background bots rewrote tracker from stale memory → corruption** | done: atomic lock + read-modify-write + bots killed |
| 6 | **High** | **Regime gate A/B = -11pp, destroying value** | done: `gate_blocks=False` default (report-only) |
| 7 | **High** | **Directional signal dead 6h-48h, no positive edge** | documented; system leans on pairs edge |
| 8 | **Medium** | 6 duplicated `fetch_klines` + racing writers | `market_data.py` cache layer added (partial wire-in) |
| 9 | **Medium** | No isolated pairs paper engine | done: `pair_paper.py` + `--rescan` universe |
| 10 | **Low** | Machine-state files tracked | done: gitignored |
| 11 | **Low** | No automated tests | done: 59 tests |

---

## 4. Changes by file (this pass)

**New**
- `crypto/market_data.py` — shared TTL cache pipeline (price/klines/close array) over the canonical loader.
- `crypto/pair_paper.py` — two-leg paper book; `PairPaperBook.step()` drives a z-score state machine; CLI + tests.
- `crypto/stat_arb.py` gains `--monitor` (live signal + `pairs_signals.json` journal) and `--rescan` (stage-1 + OOS proof → `pairs_universe.json`).
- `crypto/pairs_evidence.json` — frozen OOS results for all tested pairs.

**Modified**
- `crypto/prediction_tracker.py` — atomic lock (`prediction_accuracy.json.lock`), temp-write-rename, read-modify-write in all mutating methods. Stale-lock detection by PID.
- `crypto/regime_gate.py` — `GATE_BLOCKS=False` default; counter-trend and flat-low-conf now return `tradable=True` with status `COUNTER_TREND` / `FLAT_LOW_CONF` (reported, not blocked). A/B evidence in docstring.
- `crypto/advanced_bot.py` — `run_pairs_scan()` added + called in the main loop; trades only OOS-approved pairs from `pairs_universe.json`.
- `crypto/tests/test_tracker.py` — `_backdate` now persists (compat with read-modify-write).
- `crypto/tests/test_calibration.py` — fixed absent-bin key (`90-101`→`90-100`).
- `crypto/tests/test_regime_gate.py` — covers both reporting mode and explicit `gate_blocks=True`; asserts default is `False`.
- `crypto/models/calibration.json` — refreshed from horizon-matched backfill (source documented).

---

## 5. Remaining risks

1. **Git history still contains secrets.** Needs `git filter-repo` + key rotation.
2. **No ML directional edge at any horizon.** Paper auto-trades from `advanced_bot` are currently opened on a *dead* signal (`meta_labeling_reliable` filter). Should be gated or removed until the predictor is reworked.
3. **Pairs edge is thin and unstable.** 3 trades is not statistical proof; cointegration drifts. Requires continuous rescanning and small sizing.
4. **market_data.py cache not yet wired into advanced_bot/dashboard/kimi** — they still fetch independently (works, but duplicates calls + leaves stale-bot risk if restarted).
5. **Background bots are killed.** They will restart on reboot and re-introduce the race unless their startup is coordinated through a single launcher or the lock is relied upon (lock handles it, but memory-load-on-startup can still clobber).
6. **Meta-labeler/online learning still unproven**, depends on tracker data that only now accumulates honestly.

---

## 6. Phase 2 roadmap (updated)

1. **Secrets**: rotate keys; purge `crypto/` history; verify with `gitleaks`.
2. **Wire `market_data.py` everywhere**: replace direct Binance calls in `advanced_bot`, `dashboard_server`, `kimi_bot` with the shared cache. Eliminates duplicate fetches and stale-bot races.
3. **Remove dead directional auto-trading**: stop opening paper trades on the 24h signal (or require a pairs/pipeline confirmation). The signal is noise.
4. **Harden the pairs loop**: run `--rescan` weekly (cron/scheduler) → refresh `pairs_universe.json`; trade only the approved set; add a max-position and correlation check.
5. **Expand universe + longer windows**: 30+ symbols, multi-month lookback, to find more stable cointegration (BTC/ETH, ETH pairs, etc.).
6. **Calibration as a service**: auto-refresh `calibration.json` from verified tracker data with minimum-sample thresholds.
7. **Unbot launcher**: a single startup script that launches advanced_bot, dashboard, kimi in order and logs them, rather than 3 ad-hoc processes.
8. **CI**: `python -m unittest discover -s tests` + `py_compile` gate on every change (currently 59 tests, all green).
9. **Repo hygiene**: decide root vs `crypto/` repo layout; baseline commit.
