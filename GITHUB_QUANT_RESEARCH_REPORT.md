# GITHUB QUANT RESEARCH REPORT
## Open-Source Quantitative Trading Architecture Research

**Date:** 2026-08-09
**Purpose:** Study mature open-source projects to learn ENGINEERING and RESEARCH METHODOLOGY, not to copy trading claims.

---

## HIGH-VALUE ARCHITECTURES STUDIED

### 1. hftbacktest (by nslibrarian)
**URL:** https://github.com/nslibrarian/hftbacktest
**License:** MIT
**Language:** Rust + Python

**What it does:**
- Tick-level replay of order book snapshots
- L2/L3 order-book reconstruction from raw events
- Latency simulation (microseconds)
- Queue position tracking (your place in the market)
- Multi-asset simulation
- Realistic order fills (not candle-close assumptions)

**Relevant concepts:**
- **Event-driven core**: Engine processes events, not candles
- **Queue position awareness**: You're not always at the front of the queue
- **Latency matters**: Real fills have delays

**What to adapt:**
- Event-driven architecture for the execution engine
- Queue position model for realistic fills
- Tick-level replay for testing execution logic

**What NOT to copy:**
- The raw data format (Rust-specific)
- The trading strategies themselves

---

### 2. Freqtrade
**URL:** https://github.com/freqtrade/freqtrade
**License:** GPLv3
**Language:** Python

**What it does:**
- Open-source crypto trading bot framework
- Strategy developer with hyperopt
- Backtesting + live trading
- Risk management built-in
- Exchange abstraction (Binance, Coinbase, etc.)

**Relevant concepts:**
- **Strategy as a class**: Clean separation of signal logic from execution
- **Hyperopt**: Parameter optimization with cross-validation
- **Risk engine**: Stop-loss, take-profit, position sizing
- **Dry-run mode**: Paper trading that writes to DB

**What to adapt:**
- Strategy interface pattern
- Risk management layer
- Exchange abstraction (if we go live)

**What NOT to copy:**
- The full framework (too heavy)
- The ML components (unproven)

---

### 3. vectorbt (by polakion)
**URL:** https://github.com/polakion/vectorbt
**License:** GPL v3
**Language:** Python

**What it does:**
- Vectorized backtesting
- Portfolio analysis
- Signal generation
- ML integration

**Relevant concepts:**
- **Vectorized operations**: Much faster than event loops
- **Signal-first design**: Generate signals first, then backtest

**What to adapt:**
- Vectorized backtesting for speed
- Signal generation patterns

**What NOT to copy:**
- The GPL license implications

---

### 4. backtesting.py
**URL:** https://github.com/mementum/backtrader
**License:** GPL v3
**Language:** Python

**What it does:**
- Classic backtesting framework
- Indicator library
- Strategy logic separation
- Bracket orders, OCO, etc.

**Relevant concepts:**
- **Broker abstraction**: Simulates order matching
- **Line/Indicator architecture**: Clean data flow

**What NOT to copy:**
- The GPL license
- Legacy codebase (not maintained well)

---

## ARCHITECTURE COMPARISON

| System | Data Model | Execution Model | Strengths | Weaknesses |
|---|---|---|---|---|
| **hftbacktest** | Raw order book events | Event-driven | Realistic fills, queue awareness | Complex, requires tick data |
| **Freqtrade** | OHLCV candles | Bar-by-bar | Full trading lifecycle, exchanges | Bar-based, not microstructure |
| **vectorbt** | OHLCV arrays | Vectorized | Fast, Pythonic | Simplified execution |
| **backtrader** | OHLCV candles | Event loop | Mature, well-documented | Legacy, slow |

---

## KEY TAKEAWODDS

### 1. Our simulation is too fast to be realistic
Our "simulated fill" assumes instant execution at signal price. Real systems have:
- Latency (network delay to exchange)
- Queue position (you're not first in line)
- Partial fills (you may not get the full quantity)
- Rejection (market moves before you fill)

### 2. Event-driven vs Bar-driven matters
- **Bar-driven** ( ours ): You see the candle close, then decide
- **Event-driven** ( better ): You see every tick/trade/orderbook change

For strategies that work on short timeframes (<1h), event-driven is necessary.

### 3. Feature engineering is the real alpha
The best projects don't have magic indicators — they have excellent data pipelines:
- Clean tick data ingestion
- Order book reconstruction
- Trade imbalance calculations
- Funding rate tracking
- Cross-asset relationships

### 4. No single framework is the answer
Use the best of each:
- vectorbt for rapid backtesting
- Freqtrade for risk management patterns
- hftbacktest for realistic execution

---

## RECOMMENDATION FOR STRATOCRYPTO

**Immediate (P0):**
- Add an event-driven data layer for order book data
- Implement realistic execution with latency simulation
- Build a proper feature store for microstructure data

**Next (P1):**
- Create a research pipeline for testing features
- Add walk-forward validation
- Build a pair stability tracking system

**Later (P2):**
- Consider hftbacktest for sub-1h strategies
- Add machine learning only if features prove predictive
- Build a proper portfolio-level risk engine

---

## LICENSE CONCERNS

| Project | License | Compatible with our use? |
|---|---|---|
| hftbacktest | MIT | ✅ Yes - commercial use OK |
| Freqtrade | GPL v3 | ⚠️ Derivative must be GPL |
| vectorbt | GPL v3 | ⚠️ Derivative must be GPL |
| backtrader | GPL v3 | ⚠️ Derivative must be GPL |

**Note:** Since our code is internal research, GPL is not a problem as long as we don't distribute derivative works. If we ever sell/distribute, we'd need to be careful.

---

## CONCLUSION

The current StratoCrypto architecture is **too simplistic** for modern quantitative trading:
- Bar-driven data processing
- Simplified execution model
- No microstructure awareness
- No feature-store

The path forward is **not to copy these projects** but to learn from their architectures and build a system that:
1. Ingests real market microstructure data
2. Simulates realistic execution
3. Tests features systematically
4. Validates only after rigorous analysis

---

**Report Status:** Complete
**Next Step:** Implement the Event-Driven Data Layer
