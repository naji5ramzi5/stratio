"""
Paper Trading Engine v1.0
محاكاة تداول حقيقية بأموال وهمية — عشان تعرف النظام يربح ولا يخسر
"""
import json
import os
import logging
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

PAPER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_trading.json")

# Execution costs (percent). Binance spot charges 0.1% per side; a small
# slippage per fill makes paper results comparable with the backtesters,
# which already charge commission + slippage.
DEFAULT_COMMISSION_PCT = 0.10
DEFAULT_SLIPPAGE_PCT = 0.05


def _env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


class PaperTrade:
    def __init__(self, symbol, side, entry_price, size_pct, stop_loss_pct, take_profit_pct,
                 tf_hours, confidence, reason=""):
        self.symbol = symbol
        self.side = side  # "LONG" or "SHORT"
        self.entry_price = entry_price
        self.size_pct = size_pct  # % of capital used
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.tf_hours = tf_hours
        self.confidence = confidence
        self.reason = reason
        self.entry_time = datetime.now().isoformat()
        self.exit_time = None
        self.exit_price = None
        self.pnl_pct = None
        self.pnl_usd = None
        self.status = "open"  # open, closed_sl, closed_tp, closed_expired
        self.id = f"{symbol}_{side}_{datetime.now().timestamp():.0f}"

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(d):
        t = PaperTrade(d["symbol"], d["side"], d["entry_price"], d["size_pct"],
                       d["stop_loss_pct"], d["take_profit_pct"], d["tf_hours"],
                       d["confidence"], d.get("reason", ""))
        t.__dict__.update(d)
        return t


class PaperTradingEngine:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.trades = []
        self.capital = initial_capital
        self.peak_capital = initial_capital
        self.equity_curve = []
        # Execution costs: PAPER_COMMISSION_PCT / PAPER_SLIPPAGE_PCT
        self.commission_pct = _env_float("PAPER_COMMISSION_PCT", DEFAULT_COMMISSION_PCT)
        self.slippage_pct = _env_float("PAPER_SLIPPAGE_PCT", DEFAULT_SLIPPAGE_PCT)
        self.load()

    @property
    def active_trades(self):
        return [t for t in self.trades if t.status == "open"]

    @property
    def closed_trades(self):
        return [t for t in self.trades if t.status != "open"]

    def open_trade(self, symbol, side, entry_price, size_pct=None, stop_loss_pct=None,
                   take_profit_pct=None, tf_hours=24, confidence=50, reason=""):
        if size_pct is None:
            size_pct = min(confidence / 200, 0.25)
        if stop_loss_pct is None:
            stop_loss_pct = max(0.5, 5 - confidence / 20)
        if take_profit_pct is None:
            take_profit_pct = stop_loss_pct * 2

        trade = PaperTrade(symbol.upper(), side.upper(), entry_price, size_pct,
                           stop_loss_pct, take_profit_pct, tf_hours, confidence, reason)
        self.trades.append(trade)
        self.save()
        logger.info(f"[PAPER] OPEN {side} {symbol} @ {entry_price} | SL:{stop_loss_pct}% TP:{take_profit_pct}% | {size_pct*100:.1f}%")
        return trade

    def close_trade(self, trade, exit_price, reason="manual"):
        if trade.status != "open":
            return
        trade.exit_price = exit_price
        trade.exit_time = datetime.now().isoformat()
        # Apply execution costs: slippage moves the fill against us on both
        # sides, commission is charged on entry and exit notional.
        slip = self.slippage_pct / 100
        comm = self.commission_pct / 100
        if trade.side == "LONG":
            effective_entry = trade.entry_price * (1 + slip)
            effective_exit = exit_price * (1 - slip)
            trade.pnl_pct = (effective_exit - effective_entry) / trade.entry_price * 100
        else:
            effective_entry = trade.entry_price * (1 - slip)
            effective_exit = exit_price * (1 + slip)
            trade.pnl_pct = (effective_entry - effective_exit) / trade.entry_price * 100
        trade.pnl_pct -= comm * 200  # commission on entry + exit (pct of notional)

        pnl_capital = self.capital * trade.size_pct * (trade.pnl_pct / 100)
        trade.pnl_usd = round(pnl_capital, 2)
        self.capital += pnl_capital
        self.peak_capital = max(self.peak_capital, self.capital)
        trade.status = reason
        self.equity_curve.append({
            "time": datetime.now().isoformat(),
            "capital": round(self.capital, 2),
            "pnl": round(trade.pnl_usd, 2),
        })
        self.save()
        logger.info(f"[PAPER] CLOSE {trade.side} {trade.symbol} | PnL: {trade.pnl_pct:+.2f}% (${trade.pnl_usd:+.2f}) | Capital: ${self.capital:.2f}")

    def check_prices(self, price_dict):
        for trade in self.active_trades:
            price = price_dict.get(trade.symbol)
            if price is None:
                continue
            if trade.side == "LONG":
                stop = trade.entry_price * (1 - trade.stop_loss_pct / 100)
                take = trade.entry_price * (1 + trade.take_profit_pct / 100)
                if price <= stop:
                    self.close_trade(trade, price, "closed_sl")
                elif price >= take:
                    self.close_trade(trade, price, "closed_tp")
            else:
                stop = trade.entry_price * (1 + trade.stop_loss_pct / 100)
                take = trade.entry_price * (1 - trade.take_profit_pct / 100)
                if price >= stop:
                    self.close_trade(trade, price, "closed_sl")
                elif price <= take:
                    self.close_trade(trade, price, "closed_tp")

    def check_expired(self):
        now = datetime.now()
        for trade in self.active_trades:
            try:
                entry = datetime.fromisoformat(trade.entry_time)
                if (now - entry).total_seconds() > trade.tf_hours * 3600 * 1.5:
                    self.close_trade(trade, trade.entry_price, "closed_expired")
            except Exception:
                pass

    def get_stats(self):
        closed = self.closed_trades
        total = len(closed)
        if total == 0:
            return {
                "status": "no_trades",
                "capital": round(self.capital, 2),
                "total_return_pct": round((self.capital / self.initial_capital - 1) * 100, 2),
                "initial_capital": self.initial_capital,
            }
        wins = sum(1 for t in closed if t.pnl_usd and t.pnl_usd > 0)
        losses = sum(1 for t in closed if t.pnl_usd and t.pnl_usd < 0)
        pnls = [t.pnl_usd for t in closed if t.pnl_usd is not None]

        return {
            "status": "active" if self.active_trades else "idle",
            "initial_capital": self.initial_capital,
            "current_capital": round(self.capital, 2),
            "total_return_pct": round((self.capital / self.initial_capital - 1) * 100, 2),
            "total_return_usd": round(self.capital - self.initial_capital, 2),
            "active_trades": len(self.active_trades),
            "total_closed": total,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(wins / total * 100, 1) if total > 0 else 0,
            "avg_win_pct": round(np.mean([t.pnl_pct for t in closed if t.pnl_usd and t.pnl_usd > 0] or [0]), 2),
            "avg_loss_pct": round(np.mean([t.pnl_pct for t in closed if t.pnl_usd and t.pnl_usd < 0] or [0]), 2),
            "profit_factor": round(sum(p for p in pnls if p > 0) / max(abs(sum(p for p in pnls if p < 0)), 1), 2),
            "largest_win": round(max([t.pnl_usd for t in closed if t.pnl_usd] or [0]), 2),
            "largest_loss": round(min([t.pnl_usd for t in closed if t.pnl_usd] or [0]), 2),
            "peak_capital": round(self.peak_capital, 2),
            "drawdown_pct": round((self.peak_capital - self.capital) / self.peak_capital * 100, 2) if self.peak_capital > 0 else 0,
        }

    def get_recent_trades(self, limit=20):
        return [t.to_dict() for t in self.trades[-limit:]]

    def save(self):
        try:
            data = {
                "initial_capital": self.initial_capital,
                "capital": self.capital,
                "peak_capital": self.peak_capital,
                "equity_curve": self.equity_curve[-1000:],
                "trades": [t.to_dict() for t in self.trades[-500:]],
            }
            with open(PAPER_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[PAPER] Save failed: {e}")

    def load(self):
        if not os.path.exists(PAPER_FILE):
            return
        try:
            with open(PAPER_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.initial_capital = data.get("initial_capital", 10000)
            self.capital = data.get("capital", self.initial_capital)
            self.peak_capital = data.get("peak_capital", self.capital)
            self.equity_curve = data.get("equity_curve", [])
            self.trades = [PaperTrade.from_dict(t) for t in data.get("trades", [])]
            logger.info(f"[PAPER] Loaded: ${self.capital:.2f} | {len(self.trades)} trades ({len(self.active_trades)} active)")
        except Exception as e:
            logger.warning(f"[PAPER] Load failed: {e}")

    def reset(self, capital=None):
        self.trades = []
        self.capital = capital or self.initial_capital
        self.peak_capital = self.capital
        self.equity_curve = []
        self.save()


paper_engine = PaperTradingEngine()