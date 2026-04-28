"""
📊 Backtest Engine
===================
Tests technical indicator signals against historical price data.
Calculates accuracy, returns, max drawdown, Sharpe ratio.
"""

import json
import datetime
import numpy as np


def _calc_rsi(closes, period=14):
    """Calculate RSI for an array of closes."""
    if len(closes) < period + 1:
        return [50.0] * len(closes)
    deltas = np.diff(closes)
    rsi_values = [50.0] * period
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi_values.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))
    return rsi_values


def _calc_sma(closes, period):
    """Calculate SMA."""
    sma = []
    for i in range(len(closes)):
        if i < period - 1:
            sma.append(None)
        else:
            sma.append(np.mean(closes[i - period + 1:i + 1]))
    return sma


def run_backtest(ticker: str, period_years: float = 1, holding_days: int = 10,
                  rsi_buy: float = 35, rsi_sell: float = 65,
                  use_sma_filter: bool = True) -> dict:
    """
    Run a backtest on historical data using RSI + SMA signals.

    Args:
        ticker: Stock ticker symbol
        period_years: Years of historical data to test (default 1)
        holding_days: Number of days to hold after signal (default 10)
        rsi_buy: RSI threshold for buy signal (default 35)
        rsi_sell: RSI threshold for sell signal (default 65)
        use_sma_filter: Whether to use SMA20 > SMA50 as trend filter (default True)

    Returns:
        Dict with signals, accuracy metrics, returns, and performance stats.
    """
    import yfinance as yf

    period_map = {0.5: "6mo", 1: "1y", 2: "2y", 3: "3y", 5: "5y"}
    yf_period = period_map.get(period_years, "1y")

    stock = yf.Ticker(ticker)
    hist = stock.history(period=yf_period, interval="1d")

    if hist.empty or len(hist) < 60:
        return {"error": f"Insufficient data for {ticker} — need at least 60 days"}

    closes = hist["Close"].values
    dates = [d.strftime("%Y-%m-%d") for d in hist.index]

    # Calculate indicators
    rsi = _calc_rsi(closes)
    sma20 = _calc_sma(closes, 20)
    sma50 = _calc_sma(closes, 50)

    # Generate signals
    signals = []
    for i in range(50, len(closes) - holding_days):
        signal = None
        rsi_val = rsi[i]

        if use_sma_filter:
            if sma20[i] is None or sma50[i] is None:
                continue
            sma_bullish = sma20[i] > sma50[i]
            sma_bearish = sma20[i] < sma50[i]
        else:
            sma_bullish = True
            sma_bearish = True

        if rsi_val < rsi_buy and sma_bullish:
            signal = "BUY"
        elif rsi_val > rsi_sell and sma_bearish:
            signal = "SELL"
        else:
            continue

        # Calculate actual return after holding_days
        entry_price = closes[i]
        exit_price = closes[i + holding_days]
        actual_return = (exit_price - entry_price) / entry_price * 100

        # Was signal correct?
        if signal == "BUY":
            correct = actual_return > 0
        else:  # SELL
            correct = actual_return < 0

        signals.append({
            "date": dates[i],
            "signal": signal,
            "entry_price": round(float(entry_price), 2),
            "exit_price": round(float(exit_price), 2),
            "return_pct": round(float(actual_return), 2),
            "correct": correct,
            "rsi": round(float(rsi_val), 1),
        })

    if not signals:
        return {"error": "No signals generated in backtest period", "ticker": ticker}

    # Calculate metrics
    total_signals = len(signals)
    correct_signals = sum(1 for s in signals if s["correct"])
    accuracy = correct_signals / total_signals * 100

    buy_signals = [s for s in signals if s["signal"] == "BUY"]
    sell_signals = [s for s in signals if s["signal"] == "SELL"]
    buy_accuracy = sum(1 for s in buy_signals if s["correct"]) / len(buy_signals) * 100 if buy_signals else 0
    sell_accuracy = sum(1 for s in sell_signals if s["correct"]) / len(sell_signals) * 100 if sell_signals else 0

    returns = [s["return_pct"] for s in signals]
    avg_return = np.mean(returns)
    total_return = sum(returns)  # Simplified — assumes sequential non-overlapping trades
    max_return = max(returns)
    min_return = min(returns)
    std_return = np.std(returns)

    # Sharpe ratio (simplified, annualized)
    sharpe = (avg_return / std_return * np.sqrt(252 / holding_days)) if std_return > 0 else 0

    # Max drawdown of cumulative returns
    cumulative = np.cumsum(returns) / 100
    peak = np.maximum.accumulate(1 + cumulative)
    drawdowns = (1 + cumulative - peak) / peak * 100
    max_drawdown = float(np.min(drawdowns))

    # Buy & Hold comparison
    buy_hold_return = (closes[-1] - closes[50]) / closes[50] * 100

    # Win rate by signal type
    win_returns = [r for r in returns if r > 0]
    loss_returns = [r for r in returns if r <= 0]
    avg_win = np.mean(win_returns) if win_returns else 0
    avg_loss = np.mean(loss_returns) if loss_returns else 0
    profit_factor = abs(sum(win_returns) / sum(loss_returns)) if loss_returns and sum(loss_returns) != 0 else float('inf')

    result = {
        "ticker": ticker.upper(),
        "backtest_period": f"{period_years} year(s)",
        "holding_days": holding_days,
        "parameters": {
            "rsi_buy_threshold": rsi_buy,
            "rsi_sell_threshold": rsi_sell,
            "sma_filter": use_sma_filter,
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
            "avg_return_per_trade": round(float(avg_return), 2),
            "total_return_pct": round(float(total_return), 2),
            "max_single_return": round(float(max_return), 2),
            "min_single_return": round(float(min_return), 2),
            "std_return": round(float(std_return), 2),
            "sharpe_ratio": round(float(sharpe), 2),
            "max_drawdown_pct": round(float(max_drawdown), 2),
            "profit_factor": round(float(profit_factor), 2),
            "avg_win": round(float(avg_win), 2),
            "avg_loss": round(float(avg_loss), 2),
        },
        "comparison": {
            "strategy_return": round(float(total_return), 2),
            "buy_hold_return": round(float(buy_hold_return), 2),
            "outperformance": round(float(total_return - buy_hold_return), 2),
        },
        "signals": signals[-20:],  # Last 20 signals
        "disclaimer": "Past performance does not predict future results. Yatırım tavsiyesi değildir.",
    }
    return result


def run_backtest_json(ticker: str, period_years: float = 1, holding_days: int = 10) -> str:
    """Run backtest and return JSON string (for Gradio/API use)."""
    result = run_backtest(ticker, period_years, holding_days)
    return json.dumps(result, indent=2, ensure_ascii=False)


def format_backtest_markdown(result: dict) -> str:
    """Format backtest result as readable markdown."""
    if "error" in result:
        return f"❌ {result['error']}"

    s = result["summary"]
    r = result["returns"]
    c = result["comparison"]

    accuracy_emoji = "🟢" if s["accuracy_pct"] >= 60 else ("🟡" if s["accuracy_pct"] >= 50 else "🔴")

    md = f"""
## 📊 Backtest Sonuçları — {result['ticker']}

**Dönem:** {result['backtest_period']} | **Tutma süresi:** {result['holding_days']} gün

### {accuracy_emoji} Doğruluk

| Metrik | Değer |
|---|---|
| 🎯 Toplam Sinyal | {s['total_signals']} ({s['buy_signals']} AL, {s['sell_signals']} SAT) |
| ✅ Genel Doğruluk | **{s['accuracy_pct']}%** |
| 📈 AL Doğruluğu | {s['buy_accuracy_pct']}% |
| 📉 SAT Doğruluğu | {s['sell_accuracy_pct']}% |

### 💰 Getiri

| Metrik | Değer |
|---|---|
| 📊 Toplam Getiri | **{r['total_return_pct']:+.1f}%** |
| 📊 Ort. Trade Getiri | {r['avg_return_per_trade']:+.2f}% |
| 🏆 En İyi Trade | {r['max_single_return']:+.2f}% |
| 💀 En Kötü Trade | {r['min_single_return']:+.2f}% |
| 📈 Sharpe Ratio | {r['sharpe_ratio']} |
| 📉 Max Drawdown | {r['max_drawdown_pct']:.1f}% |
| 💪 Profit Factor | {r['profit_factor']} |

### 🔄 Karşılaştırma

| Strateji | Getiri |
|---|---|
| 📊 Sinyal Stratejisi | **{c['strategy_return']:+.1f}%** |
| 🏠 Buy & Hold | {c['buy_hold_return']:+.1f}% |
| 🆚 Fark | **{c['outperformance']:+.1f}%** |

### Son Sinyaller

| Tarih | Sinyal | Giriş | Çıkış | Getiri | Sonuç |
|---|---|---|---|---|---|
"""
    for sig in result.get("signals", [])[-10:]:
        emoji = "✅" if sig["correct"] else "❌"
        sig_emoji = "📈" if sig["signal"] == "BUY" else "📉"
        md += f"| {sig['date']} | {sig_emoji} {sig['signal']} | ${sig['entry_price']} | ${sig['exit_price']} | {sig['return_pct']:+.2f}% | {emoji} |\n"

    md += f"\n⚠️ *{result['disclaimer']}*"
    return md
