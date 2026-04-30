"""
📊 Backtesting Engine — Test signals against historical data
================================================================
This is the GAME CHANGER. Tests the exact same signal logic
against real historical data to prove it works (or doesn't).

Metrics: accuracy, total return, Sharpe ratio, max drawdown,
profit factor, win rate, comparison vs buy & hold.
"""

import numpy as np
import datetime
import logging

logger = logging.getLogger("trade_bot.backtest")


def _calc_rsi_series(closes, period=14):
    """Calculate RSI for all data points."""
    rsi_values = [50.0] * len(closes)
    if len(closes) < period + 1:
        return rsi_values
    
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi_values[i + 1] = 100.0
        else:
            rsi_values[i + 1] = 100 - (100 / (1 + avg_gain / avg_loss))
    
    return rsi_values


def _calc_sma(closes, period):
    sma = [None] * len(closes)
    for i in range(period - 1, len(closes)):
        sma[i] = float(np.mean(closes[i - period + 1:i + 1]))
    return sma


def _calc_volatility(closes, window=20):
    """Rolling annualized volatility."""
    vols = [None] * len(closes)
    for i in range(window, len(closes)):
        rets = np.diff(closes[i-window:i+1]) / closes[i-window:i]
        vols[i] = float(np.std(rets) * np.sqrt(252))
    return vols


def run_backtest(ticker: str, period_years: float = 1, holding_days: int = 10,
                  rsi_buy: float = 35, rsi_sell: float = 65,
                  use_sma_filter: bool = True,
                  use_volatility_filter: bool = True,
                  vol_threshold: float = 0.50,
                  initial_capital: float = 100000) -> dict:
    """
    Run a full backtest with RSI + SMA crossover + volatility filter.
    
    Strategy:
      BUY when: RSI < rsi_buy AND (SMA20 > SMA50 if sma_filter) AND (vol < threshold if vol_filter)
      SELL when: RSI > rsi_sell AND (SMA20 < SMA50 if sma_filter)
      Hold for holding_days then exit.
    """
    from core.data_ingest import fetch_ohlcv
    
    period_map = {0.5: "6mo", 1: "1y", 2: "2y", 3: "3y", 5: "5y"}
    yf_period = period_map.get(period_years, "1y")
    
    result = fetch_ohlcv(ticker, period=yf_period, interval="1d")
    df = result.get("df")
    source = result.get("source", "unknown")
    
    if df is None or len(df) < 60:
        return {"error": f"Insufficient data for {ticker} — need at least 60 days", "ticker": ticker}
    
    closes = df["Close"].values.astype(float)
    dates = [d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d)[:10] for d in df.index]
    
    # Calculate indicators
    rsi = _calc_rsi_series(closes)
    sma20 = _calc_sma(closes, 20)
    sma50 = _calc_sma(closes, 50)
    vols = _calc_volatility(closes) if use_volatility_filter else [None] * len(closes)
    
    # Generate signals and track trades
    signals = []
    equity_curve = [initial_capital]
    capital = initial_capital
    
    start_idx = 50  # Need enough data for SMA50
    
    for i in range(start_idx, len(closes) - holding_days):
        signal = None
        rsi_val = rsi[i]
        
        # SMA filter
        if use_sma_filter:
            if sma20[i] is None or sma50[i] is None:
                continue
            sma_bullish = sma20[i] > sma50[i]
            sma_bearish = sma20[i] < sma50[i]
        else:
            sma_bullish = True
            sma_bearish = True
        
        # Volatility filter
        if use_volatility_filter and vols[i] is not None:
            if vols[i] > vol_threshold:
                # Skip BUY signals in extreme volatility
                if rsi_val < rsi_buy:
                    continue
        
        # Signal generation
        if rsi_val < rsi_buy and sma_bullish:
            signal = "BUY"
        elif rsi_val > rsi_sell and sma_bearish:
            signal = "SELL"
        else:
            continue
        
        # Calculate trade result
        entry_price = closes[i]
        exit_price = closes[i + holding_days]
        
        if signal == "BUY":
            actual_return_pct = (exit_price - entry_price) / entry_price * 100
        else:  # SELL (short)
            actual_return_pct = (entry_price - exit_price) / entry_price * 100
        
        correct = actual_return_pct > 0
        
        # Position sizing (2% risk per trade)
        position_size = capital * 0.02  # Risk 2% per trade
        pnl = position_size * (actual_return_pct / 100)
        capital += pnl
        
        signals.append({
            "date": dates[i],
            "signal": signal,
            "entry_price": round(float(entry_price), 2),
            "exit_price": round(float(exit_price), 2),
            "return_pct": round(float(actual_return_pct), 2),
            "pnl": round(float(pnl), 2),
            "correct": correct,
            "rsi": round(float(rsi_val), 1),
            "sma20": round(float(sma20[i]), 2) if sma20[i] else None,
            "sma50": round(float(sma50[i]), 2) if sma50[i] else None,
            "volatility": round(float(vols[i]), 3) if vols[i] else None,
        })
        equity_curve.append(capital)
    
    if not signals:
        return {"error": "No signals generated in backtest period", "ticker": ticker,
                "note": "Try adjusting RSI thresholds or disabling SMA filter"}
    
    # ═══════════════════════════════════════
    # PERFORMANCE METRICS
    # ═══════════════════════════════════════
    
    total_signals = len(signals)
    correct_signals = sum(1 for s in signals if s["correct"])
    accuracy = correct_signals / total_signals * 100
    
    buy_signals = [s for s in signals if s["signal"] == "BUY"]
    sell_signals = [s for s in signals if s["signal"] == "SELL"]
    buy_accuracy = sum(1 for s in buy_signals if s["correct"]) / len(buy_signals) * 100 if buy_signals else 0
    sell_accuracy = sum(1 for s in sell_signals if s["correct"]) / len(sell_signals) * 100 if sell_signals else 0
    
    returns = [s["return_pct"] for s in signals]
    avg_return = float(np.mean(returns))
    total_return = (capital - initial_capital) / initial_capital * 100
    max_return = float(max(returns))
    min_return = float(min(returns))
    std_return = float(np.std(returns))
    
    # Sharpe ratio
    sharpe = (avg_return / std_return * np.sqrt(252 / holding_days)) if std_return > 0 else 0
    
    # Max drawdown
    equity_arr = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - peak) / peak * 100
    max_drawdown = float(np.min(drawdowns))
    
    # Profit factor
    win_returns = [s["pnl"] for s in signals if s["pnl"] > 0]
    loss_returns = [s["pnl"] for s in signals if s["pnl"] <= 0]
    profit_factor = abs(sum(win_returns) / sum(loss_returns)) if loss_returns and sum(loss_returns) != 0 else float('inf')
    
    # Buy & Hold comparison
    buy_hold_return = (closes[-1] - closes[start_idx]) / closes[start_idx] * 100
    
    avg_win = float(np.mean([r for r in returns if r > 0])) if any(r > 0 for r in returns) else 0
    avg_loss = float(np.mean([r for r in returns if r <= 0])) if any(r <= 0 for r in returns) else 0
    
    return {
        "ticker": ticker.upper(),
        "data_source": source,
        "backtest_period": f"{period_years} year(s)",
        "holding_days": holding_days,
        "initial_capital": initial_capital,
        "final_capital": round(capital, 2),
        
        "parameters": {
            "rsi_buy_threshold": rsi_buy,
            "rsi_sell_threshold": rsi_sell,
            "sma_filter": use_sma_filter,
            "volatility_filter": use_volatility_filter,
            "volatility_threshold": vol_threshold,
        },
        
        "summary": {
            "total_signals": total_signals,
            "buy_signals": len(buy_signals),
            "sell_signals": len(sell_signals),
            "accuracy_pct": round(accuracy, 1),
            "buy_accuracy_pct": round(buy_accuracy, 1),
            "sell_accuracy_pct": round(sell_accuracy, 1),
        },
        
        "returns": {
            "total_return_pct": round(float(total_return), 2),
            "avg_return_per_trade": round(avg_return, 2),
            "max_single_return": round(max_return, 2),
            "min_single_return": round(min_return, 2),
            "std_return": round(std_return, 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "max_drawdown_pct": round(float(max_drawdown), 2),
            "profit_factor": round(float(profit_factor), 2) if profit_factor != float('inf') else "∞",
            "avg_win_pct": round(avg_win, 2),
            "avg_loss_pct": round(avg_loss, 2),
        },
        
        "comparison": {
            "strategy_return": round(float(total_return), 2),
            "buy_hold_return": round(float(buy_hold_return), 2),
            "outperformance": round(float(total_return - buy_hold_return), 2),
        },
        
        "equity_curve": [round(e, 2) for e in equity_curve],
        "signals": signals,
        "disclaimer": "Past performance does not predict future results. Yatırım tavsiyesi değildir.",
    }
