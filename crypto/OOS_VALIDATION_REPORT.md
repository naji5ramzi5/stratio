# STRATOCRYPTO — OOS / WALK-FORWARD VALIDATION REPORT

*Generated:* 2026-08-22T22:33:16.979433+00:00  |  *Pipeline:* 2.1.0

## 1. Setup

- Symbol: **BTCUSDT** | Interval: **1h** | Bars loaded: **1300**
- Walk-forward windows: **2** (Train=700, Val=150, Test=150, Step=150, Embargo=24)
- Horizon: **24 bars** | Neutral band: ±0.3%
- Round-trip cost: **30.0 bps** (taker 10.0bps + slip 5.0bps, 2 fills)
- Features: **112** (reused EXACTLY from live ml_trainer pipeline)

## 2. Data Download Log

- source: `Binance API (data_loader.binance_ohlcv)`
- symbol: `BTCUSDT`
- interval: `1h`
- bars_requested: `1300`
- rows_after_cleanup: `1227`
- n_fwd_bars: `24`
- feature_count: `112`
- start_utc: `2026-06-02T16:00:00+00:00`
- end_utc: `2026-07-23T18:00:00+00:00`
- downloaded_at_utc: `2026-08-22T22:33:01.715894+00:00`

## 3. Leakage Self-Check

- **Passed:** True
- No future-looking feature detected. Label is a forward return (allowed); features are causal.

## 4. Pooled OOS Metrics (all folds combined)

- Accuracy: **0.4067**
- Balanced Accuracy: **0.3374**
- Brier Score: **0.6803** (lower=better; 0=perfect)
- Log Loss: **1.1136**
- Neutral%: **0.003**
- Precision [DOWN,NEU,UP]: [0.3979, 0.0, 0.4259]
- Recall    [DOWN,NEU,UP]: [0.6441, 0.0, 0.368]
- Confusion [DOWN,NEU,UP]:
```
[[76, 0, 42], [37, 0, 20], [78, 1, 46]]
```

### Calibration (reliability)

- **DOWN**: predicted=[0.316, 0.505, 0.708, 0.836] observed=[0.402, 0.226, 0.6, 0.471]
- **NEUTRAL**: predicted=[0.123, 0.25] observed=[0.187, 0.204]
- **UP**: predicted=[0.13, 0.307, 0.497, 0.645] observed=[0.286, 0.452, 0.485, 0.1]

## 5. Trade Simulation (after costs)

- Model net return: **-9.667%** | trades=11 | PF=0.169 | win%=18.18 | maxDD=9.697%

### Baselines (same OOS window, same costs)

- buy_and_hold: net=-0.951% PF=0.863 win%=46.15 trades=13
- majority_class: net=-0.951% PF=0.863 win%=46.15 trades=13
- random: net=-6.522% PF=0.187 win%=25.0 trades=8
- simple_momentum: net=6.967% PF=3.985 win%=69.23 trades=13

## 6. Per-Regime

- high_vol: n=140 acc=0.3571 bal_acc=0.3583 neutral%=0.0
- low_vol: n=141 acc=0.4468 bal_acc=0.334 neutral%=0.0

## 7. Edge Verdict

**PROFITABILITY UNPROVEN**

- Edge requires at least 3 folds, 30 OOS trades, positive net return, profit factor > 1, balanced accuracy > 0.55, and outperformance versus Buy & Hold.
- Note: prediction QUALITY (metrics above) is separate from tradeability (Section 5). A model can be statistically decent yet unprofitable after costs.

## 8. Assumptions & Limits

- Features reused from `ml_trainer.compute_features` to match live inference.
- Trade sim assumes one position per `n_fwd` bars, round-trip cost, no funding, no partial fills.
- No OOS data used for tuning. Tuning restricted to Train/Validation, with an embargo equal to the forecast horizon between windows.
- If data download failed or windows were too few, results are NOT conclusive.

---
*Paper/Research mode only. No live orders were placed.*