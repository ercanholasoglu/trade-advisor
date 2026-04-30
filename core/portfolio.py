"""
💼 Portfolio Simulator — Stateful portfolio tracking + simulation
===================================================================
Features:
  - Paper portfolio with persistent positions (in-memory, survives during session)
  - Open/close positions based on signals
  - Track PnL, drawdown, win rate in real-time
  - Full trade history with timestamps
  - Portfolio simulation: run signals on multiple tickers over time
"""

import datetime
import json
import numpy as np
import logging

logger = logging.getLogger("trade_bot.portfolio")


class PortfolioSimulator:
    """Stateful paper portfolio that tracks all trades and metrics."""

    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, dict] = {}
        self.trade_history: list[dict] = []
        self.equity_snapshots: list[dict] = []
        self._snapshot_equity()

    def execute_signal(self, signal_data: dict) -> dict:
        """
        Execute a trade based on signal engine output.
        BUY/STRONG_BUY → open long position
        SELL/STRONG_SELL → close if held, else skip (no shorting in sim)
        HOLD → no action
        """
        ticker = signal_data.get("ticker", "").upper()
        sig = signal_data.get("signal", "HOLD")
        trade = signal_data.get("trade", {})
        price = trade.get("entry_price", 0)
        shares = trade.get("shares", 0)

        if sig in ("BUY", "STRONG_BUY") and shares > 0 and price > 0:
            return self._open_position(ticker, shares, price, signal_data)
        elif sig in ("SELL", "STRONG_SELL") and ticker in self.positions:
            return self._close_position(ticker, price, f"Signal: {sig}")
        else:
            return {
                "action": "NONE",
                "ticker": ticker,
                "signal": sig,
                "reason": "HOLD — no action" if sig == "HOLD" else f"{sig} — no position to close" if sig in ("SELL","STRONG_SELL") else "Insufficient data",
            }

    def _open_position(self, ticker, shares, price, signal_data):
        cost = shares * price
        if cost > self.cash:
            shares = int(self.cash * 0.95 / price)
            cost = shares * price
        if shares <= 0:
            return {"action": "REJECTED", "ticker": ticker, "reason": "Insufficient cash"}
        if ticker in self.positions:
            return {"action": "REJECTED", "ticker": ticker, "reason": f"Already holding {ticker}"}

        self.cash -= cost
        self.positions[ticker] = {
            "shares": shares,
            "entry_price": price,
            "cost": round(cost, 2),
            "entry_date": datetime.datetime.now().isoformat(),
            "stop_loss": signal_data.get("trade", {}).get("stop_loss"),
            "take_profit": signal_data.get("trade", {}).get("take_profit"),
            "signal": signal_data.get("signal"),
            "confidence": signal_data.get("confidence"),
        }
        self._snapshot_equity()
        return {
            "action": "OPENED",
            "ticker": ticker,
            "shares": shares,
            "price": price,
            "cost": round(cost, 2),
            "remaining_cash": round(self.cash, 2),
        }

    def _close_position(self, ticker, price, reason="manual"):
        if ticker not in self.positions:
            return {"action": "ERROR", "reason": f"No position in {ticker}"}
        pos = self.positions.pop(ticker)
        proceeds = pos["shares"] * price
        self.cash += proceeds
        pnl = proceeds - pos["cost"]
        pnl_pct = (price - pos["entry_price"]) / pos["entry_price"] * 100

        trade_record = {
            "ticker": ticker,
            "side": "LONG",
            "shares": pos["shares"],
            "entry_price": pos["entry_price"],
            "exit_price": round(price, 2),
            "entry_date": pos["entry_date"],
            "exit_date": datetime.datetime.now().isoformat(),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "reason": reason,
            "signal_confidence": pos.get("confidence"),
        }
        self.trade_history.append(trade_record)
        self._snapshot_equity()
        return {"action": "CLOSED", **trade_record, "remaining_cash": round(self.cash, 2)}

    def close_all(self, current_prices: dict[str, float]) -> list[dict]:
        """Close all open positions at given prices."""
        results = []
        for ticker in list(self.positions.keys()):
            price = current_prices.get(ticker, self.positions[ticker]["entry_price"])
            results.append(self._close_position(ticker, price, "close_all"))
        return results

    def check_sl_tp(self, current_prices: dict[str, float]) -> list[dict]:
        """Check stop-loss / take-profit for all positions."""
        triggered = []
        for ticker in list(self.positions.keys()):
            pos = self.positions[ticker]
            price = current_prices.get(ticker)
            if price is None:
                continue
            if pos.get("stop_loss") and price <= pos["stop_loss"]:
                triggered.append(self._close_position(ticker, price, "STOP_LOSS"))
            elif pos.get("take_profit") and price >= pos["take_profit"]:
                triggered.append(self._close_position(ticker, price, "TAKE_PROFIT"))
        return triggered

    def _snapshot_equity(self):
        equity = self.get_equity()
        self.equity_snapshots.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "equity": round(equity, 2),
            "cash": round(self.cash, 2),
            "positions": len(self.positions),
        })

    def get_equity(self, current_prices: dict[str, float] | None = None) -> float:
        prices = current_prices or {}
        market_value = sum(
            pos["shares"] * prices.get(t, pos["entry_price"])
            for t, pos in self.positions.items()
        )
        return self.cash + market_value

    def get_metrics(self) -> dict:
        """Calculate portfolio performance metrics."""
        equity = self.get_equity()
        total_pnl = equity - self.initial_cash
        total_pnl_pct = (total_pnl / self.initial_cash) * 100

        # Trade stats
        closed = self.trade_history
        wins = [t for t in closed if t["pnl"] > 0]
        losses = [t for t in closed if t["pnl"] <= 0]
        win_rate = len(wins) / len(closed) * 100 if closed else 0

        # Drawdown from equity snapshots
        equities = [s["equity"] for s in self.equity_snapshots]
        if len(equities) >= 2:
            eq_arr = np.array(equities)
            peak = np.maximum.accumulate(eq_arr)
            dd = (eq_arr - peak) / peak * 100
            max_drawdown = float(np.min(dd))
        else:
            max_drawdown = 0

        avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
        avg_loss = np.mean([t["pnl_pct"] for t in losses]) if losses else 0

        total_won = sum(t["pnl"] for t in wins)
        total_lost = abs(sum(t["pnl"] for t in losses))
        profit_factor = total_won / total_lost if total_lost > 0 else float("inf")

        return {
            "initial_cash": self.initial_cash,
            "cash": round(self.cash, 2),
            "equity": round(equity, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "open_positions": len(self.positions),
            "total_trades": len(closed),
            "win_rate": round(win_rate, 1),
            "avg_win_pct": round(float(avg_win), 2),
            "avg_loss_pct": round(float(avg_loss), 2),
            "profit_factor": round(float(profit_factor), 2) if profit_factor != float("inf") else "∞",
            "max_drawdown_pct": round(max_drawdown, 2),
        }

    def get_positions_detail(self) -> list[dict]:
        result = []
        for ticker, pos in self.positions.items():
            result.append({
                "ticker": ticker,
                "shares": pos["shares"],
                "entry_price": pos["entry_price"],
                "cost": pos["cost"],
                "entry_date": pos["entry_date"],
                "stop_loss": pos.get("stop_loss"),
                "take_profit": pos.get("take_profit"),
                "signal": pos.get("signal"),
            })
        return result

    def get_trade_history(self, last_n: int = 50) -> list[dict]:
        return self.trade_history[-last_n:]

    def reset(self, initial_cash: float = 100000.0):
        self.__init__(initial_cash)


# ── Global instance ──
_portfolio: PortfolioSimulator | None = None


def get_portfolio(initial_cash: float = 100000.0) -> PortfolioSimulator:
    global _portfolio
    if _portfolio is None:
        _portfolio = PortfolioSimulator(initial_cash)
    return _portfolio


def reset_portfolio(initial_cash: float = 100000.0) -> PortfolioSimulator:
    global _portfolio
    _portfolio = PortfolioSimulator(initial_cash)
    return _portfolio


def run_multi_ticker_simulation(tickers: list[str], portfolio_value: float = 100000,
                                 risk_tolerance: str = "moderate") -> dict:
    """
    Run signal engine on multiple tickers, execute trades, return full results.
    This is the 'AI Trade Decision' pipeline:
      signal → explanation → backtest metrics → portfolio simulation
    """
    from core.signal_engine import generate_signal
    from core.backtester import run_backtest

    portfolio = get_portfolio(portfolio_value)
    decisions = []

    for ticker in tickers:
        # 1. Generate signal
        sig = generate_signal(ticker, portfolio_value, risk_tolerance)

        # 2. Run mini backtest for this ticker
        bt = run_backtest(ticker, period_years=1, holding_days=10)
        bt_accuracy = bt.get("summary", {}).get("accuracy_pct", 0) if "error" not in bt else 0
        bt_sharpe = bt.get("returns", {}).get("sharpe_ratio", 0) if "error" not in bt else 0

        # 3. Execute in portfolio
        exec_result = portfolio.execute_signal(sig)

        decisions.append({
            "ticker": ticker,
            "signal": sig["signal"],
            "confidence": sig["confidence"],
            "risk": sig["risk"],
            "price": sig["price"],
            "reason": sig.get("reason", ""),
            "backtest_accuracy": bt_accuracy,
            "backtest_sharpe": bt_sharpe,
            "execution": exec_result,
            "indicators": {
                "rsi": sig["indicators"]["rsi"]["value"],
                "sma": sig["indicators"]["sma_crossover"]["signal"],
                "macd": sig["indicators"]["macd"]["signal"],
                "volatility": sig["indicators"]["volatility"]["regime"],
            },
        })

    return {
        "decisions": decisions,
        "portfolio_metrics": portfolio.get_metrics(),
        "open_positions": portfolio.get_positions_detail(),
        "trade_history": portfolio.get_trade_history(),
    }
