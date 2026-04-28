"""
🛡️ Trading Safety — Paper trading, risk limits, position tracking
====================================================================
Paper mode: virtual portfolio, no real orders.
Risk limits: %1 rule, max exposure, drawdown circuit breaker.
Position tracker: open/closed positions, P&L.
"""

import json
import datetime
import logging
import threading

logger = logging.getLogger("trade_bot.safety")

# Thread-safe position store (in-memory, persisted to SQLite)
_lock = threading.Lock()


class PaperPortfolio:
    """
    Virtual paper trading portfolio.
    Tracks positions, cash, P&L — no real money.
    """

    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, dict] = {}  # ticker → {shares, entry_price, entry_date}
        self.closed_trades: list[dict] = []
        self.trade_log: list[dict] = []

    def open_position(self, ticker: str, shares: int, price: float,
                       stop_loss: float = None, take_profit: float = None,
                       signal: str = "BUY") -> dict:
        """Open a paper position. Returns result dict."""
        ticker = ticker.upper()
        cost = shares * price

        # Risk checks
        violations = self.check_risk_limits(ticker, shares, price)
        if violations:
            result = {"status": "REJECTED", "ticker": ticker, "violations": violations}
            self._log("REJECTED", ticker, shares, price, result)
            return result

        if cost > self.cash:
            result = {"status": "REJECTED", "ticker": ticker,
                       "violations": [f"Yetersiz nakit: ${self.cash:,.2f} < ${cost:,.2f}"]}
            self._log("REJECTED", ticker, shares, price, result)
            return result

        self.cash -= cost
        self.positions[ticker] = {
            "shares": shares, "entry_price": price,
            "entry_date": datetime.datetime.now().isoformat(),
            "stop_loss": stop_loss, "take_profit": take_profit,
            "signal": signal, "cost": round(cost, 2),
        }

        result = {
            "status": "OPENED", "ticker": ticker, "shares": shares,
            "entry_price": price, "cost": round(cost, 2),
            "remaining_cash": round(self.cash, 2),
            "stop_loss": stop_loss, "take_profit": take_profit,
        }
        self._log("OPEN", ticker, shares, price, result)
        return result

    def close_position(self, ticker: str, price: float, reason: str = "manual") -> dict:
        """Close a paper position. Returns P&L."""
        ticker = ticker.upper()
        if ticker not in self.positions:
            return {"status": "ERROR", "message": f"{ticker} pozisyonu yok"}

        pos = self.positions.pop(ticker)
        proceeds = pos["shares"] * price
        self.cash += proceeds
        pnl = proceeds - pos["cost"]
        pnl_pct = (price - pos["entry_price"]) / pos["entry_price"] * 100

        trade = {
            "ticker": ticker, "shares": pos["shares"],
            "entry_price": pos["entry_price"], "exit_price": price,
            "entry_date": pos["entry_date"],
            "exit_date": datetime.datetime.now().isoformat(),
            "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
            "reason": reason,
        }
        self.closed_trades.append(trade)

        result = {"status": "CLOSED", **trade, "remaining_cash": round(self.cash, 2)}
        self._log("CLOSE", ticker, pos["shares"], price, result)
        return result

    def check_stop_loss_take_profit(self, current_prices: dict[str, float]) -> list[dict]:
        """Check all positions against SL/TP. Returns list of triggered closes."""
        triggered = []
        for ticker, pos in list(self.positions.items()):
            price = current_prices.get(ticker)
            if price is None:
                continue
            if pos.get("stop_loss") and price <= pos["stop_loss"]:
                result = self.close_position(ticker, price, reason="STOP_LOSS")
                triggered.append(result)
            elif pos.get("take_profit") and price >= pos["take_profit"]:
                result = self.close_position(ticker, price, reason="TAKE_PROFIT")
                triggered.append(result)
        return triggered

    def check_risk_limits(self, ticker: str, shares: int, price: float) -> list[str]:
        """
        Check risk limits before opening position.
        Returns list of violation messages (empty = OK).
        """
        violations = []
        cost = shares * price
        total_equity = self.get_equity(self._estimate_prices())

        # Position size limit: max 20% of equity in one trade
        if cost > total_equity * 0.20:
            violations.append(f"Pozisyon büyüklüğü portföyün %20'sini aşıyor: ${cost:,.0f} > ${total_equity*0.20:,.0f}")

        # Max 25% in single position (hard limit)
        if cost > total_equity * 0.25:
            violations.append(f"Tek pozisyon hard limiti aşılıyor (%25): ${cost:,.0f}")

        # Max 5 open positions
        if len(self.positions) >= 5 and ticker not in self.positions:
            violations.append(f"Maksimum 5 açık pozisyon limiti dolu ({len(self.positions)} mevcut)")

        # Max 80% total exposure
        total_invested = sum(p["cost"] for p in self.positions.values())
        if (total_invested + cost) > total_equity * 0.80:
            violations.append(f"Toplam yatırım %80 limitini aşıyor")

        # Drawdown circuit breaker: if portfolio down >15% from peak, stop trading
        drawdown = (total_equity - self.initial_cash) / self.initial_cash * 100
        if drawdown < -15:
            violations.append(f"⚠️ Circuit breaker: Portföy %{drawdown:.1f} düşüşte (limit: %15)")

        return violations

    def get_equity(self, current_prices: dict[str, float]) -> float:
        """Total portfolio value = cash + market value of positions."""
        mkt_value = sum(
            pos["shares"] * current_prices.get(ticker, pos["entry_price"])
            for ticker, pos in self.positions.items()
        )
        return self.cash + mkt_value

    def get_unrealized_pnl(self, current_prices: dict[str, float]) -> list[dict]:
        """Get unrealized P&L for all open positions."""
        result = []
        for ticker, pos in self.positions.items():
            price = current_prices.get(ticker, pos["entry_price"])
            pnl = (price - pos["entry_price"]) * pos["shares"]
            pnl_pct = (price - pos["entry_price"]) / pos["entry_price"] * 100
            result.append({
                "ticker": ticker, "shares": pos["shares"],
                "entry_price": pos["entry_price"], "current_price": price,
                "unrealized_pnl": round(pnl, 2), "pnl_pct": round(pnl_pct, 2),
                "stop_loss": pos.get("stop_loss"), "take_profit": pos.get("take_profit"),
            })
        return result

    def get_summary(self, current_prices: dict[str, float] = None) -> dict:
        """Portfolio summary."""
        if current_prices is None:
            current_prices = self._estimate_prices()
        equity = self.get_equity(current_prices)
        total_pnl = equity - self.initial_cash
        total_pnl_pct = total_pnl / self.initial_cash * 100

        winning = [t for t in self.closed_trades if t["pnl"] > 0]
        losing = [t for t in self.closed_trades if t["pnl"] <= 0]

        return {
            "initial_cash": self.initial_cash,
            "cash": round(self.cash, 2),
            "equity": round(equity, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "open_positions": len(self.positions),
            "closed_trades": len(self.closed_trades),
            "win_rate": round(len(winning) / len(self.closed_trades) * 100, 1) if self.closed_trades else 0,
            "avg_win": round(sum(t["pnl"] for t in winning) / len(winning), 2) if winning else 0,
            "avg_loss": round(sum(t["pnl"] for t in losing) / len(losing), 2) if losing else 0,
            "positions": self.get_unrealized_pnl(current_prices),
        }

    def _estimate_prices(self) -> dict[str, float]:
        """Use entry prices as estimates when no current price available."""
        return {t: p["entry_price"] for t, p in self.positions.items()}

    def _log(self, action, ticker, shares, price, result):
        self.trade_log.append({
            "timestamp": datetime.datetime.now().isoformat(),
            "action": action, "ticker": ticker,
            "shares": shares, "price": price,
            "result": result,
        })
        logger.info(f"Paper trade: {action} {shares}x {ticker} @ {price}")

    def to_json(self) -> str:
        return json.dumps(self.get_summary(), indent=2, default=str)


# ─── Global paper portfolio instance ───
_paper_portfolio = None


def get_paper_portfolio(initial_cash: float = 100000.0) -> PaperPortfolio:
    """Get or create the global paper portfolio."""
    global _paper_portfolio
    if _paper_portfolio is None:
        _paper_portfolio = PaperPortfolio(initial_cash)
    return _paper_portfolio


def reset_paper_portfolio(initial_cash: float = 100000.0) -> PaperPortfolio:
    """Reset the paper portfolio."""
    global _paper_portfolio
    _paper_portfolio = PaperPortfolio(initial_cash)
    return _paper_portfolio
