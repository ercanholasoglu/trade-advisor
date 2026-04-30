"""
🤖 Trade Bot Advisor v3.0 — Signal-First Architecture
=========================================================
Architecture:
  Signal Engine (deterministic) → THE decision maker
  LLM → interpreter/explainer ONLY (never generates signals)
  Backtesting → proves the strategy works on historical data
  Structured output → JSON, not "Looks bullish..."
  Sample data fallback → Space ALWAYS works

v3.0 Changes:
  ❌ LLM = decision maker → ✅ LLM = interpreter
  ❌ "Looks bullish..." → ✅ {"signal": "BUY", "confidence": 0.62, ...}
  ✅ Real signal system: SMA crossover + RSI + volatility filter
  ✅ Full backtesting engine with equity curves
  ✅ Explainability: "Why does this trade make sense?"
  ✅ Sample data fallback — demo always works
"""

import os
import json
import datetime
import gradio as gr
import numpy as np

from core.signal_engine import generate_signal
from core.backtester import run_backtest
from core.data_ingest import fetch_ohlcv, get_ticker_name
from core.charts import (
    build_candlestick_chart, build_indicator_panel,
    build_equity_curve, build_backtest_signals_chart,
    build_portfolio_equity_chart, build_pnl_distribution_chart,
    build_drawdown_chart,
)
from core.llm_interpreter import interpret_signal, _template_interpret
from core.portfolio import (
    get_portfolio, reset_portfolio, run_multi_ticker_simulation,
)


# ═══════════════════════════════════════════════════════
# TAB 1: QUICK ANALYSIS — Signal + Charts + Structured Output
# ═══════════════════════════════════════════════════════

def run_analysis(ticker, portfolio_value, risk_tolerance):
    """Main analysis pipeline: Signal Engine → Charts → Structured JSON → LLM Interpretation."""
    ticker = ticker.strip().upper()
    if not ticker:
        empty_fig = _empty_plot("Enter a ticker")
        return (
            empty_fig, empty_fig, 
            {"error": "No ticker provided"},
            "❌ Ticker girin (örn: AAPL, THYAO.IS, BTC-USD)",
            "N/A", "N/A", "N/A", "N/A"
        )
    
    # 1. Generate signal (deterministic — no LLM)
    signal = generate_signal(ticker, portfolio_value, risk_tolerance.lower(), period="3mo")
    
    # 2. Fetch data for charts
    result = fetch_ohlcv(ticker, period="3mo", interval="1d")
    df = result.get("df")
    
    # 3. Build charts
    if df is not None and len(df) >= 10:
        candlestick = build_candlestick_chart(df, ticker)
        indicators = build_indicator_panel(df, ticker)
    else:
        candlestick = _empty_plot(f"No chart data for {ticker}")
        indicators = _empty_plot("No indicator data")
    
    # 4. Structured JSON output
    structured = {
        "signal": signal["signal"],
        "confidence": signal["confidence"] / 100,  # 0-1 scale
        "reason": signal.get("reason", ""),
        "risk": signal["risk"],
        "ticker": signal["ticker"],
        "price": signal["price"],
        "indicators": {
            "rsi": signal["indicators"]["rsi"]["value"],
            "rsi_signal": signal["indicators"]["rsi"]["signal"],
            "sma_crossover": signal["indicators"]["sma_crossover"]["signal"],
            "macd": signal["indicators"]["macd"]["signal"],
            "bollinger": signal["indicators"]["bollinger"]["signal"],
            "volatility": signal["indicators"]["volatility"]["regime"],
        },
        "trade": signal["trade"],
        "votes": signal["votes"],
        "explainability": signal["explainability"],
    }
    
    # 5. LLM Interpretation (or template fallback)
    interpretation = _template_interpret(signal)
    
    # 6. Quick stats
    sig_emoji = {"STRONG_BUY": "🟢🟢", "BUY": "🟢", "HOLD": "🟡", "SELL": "🔴", "STRONG_SELL": "🔴🔴"}.get(signal["signal"], "⚪")
    signal_display = f"{sig_emoji} {signal['signal']}"
    confidence_display = f"{signal['confidence']}%"
    risk_display = signal["risk"].upper()
    price_display = f"${signal['price']:,.2f}" if signal.get("price") else "N/A"
    
    return (
        candlestick, indicators,
        structured,
        interpretation,
        signal_display, confidence_display, risk_display, price_display,
    )


# ═══════════════════════════════════════════════════════
# TAB 2: BACKTEST — Prove the strategy works
# ═══════════════════════════════════════════════════════

def run_backtest_ui(ticker, period_years, holding_days, rsi_buy, rsi_sell,
                     use_sma, use_vol_filter, vol_threshold):
    """Run backtest and return results + charts."""
    ticker = ticker.strip().upper()
    if not ticker:
        empty_fig = _empty_plot("Enter a ticker")
        return "❌ Ticker girin", empty_fig, empty_fig, empty_fig, {}
    
    result = run_backtest(
        ticker,
        period_years=period_years,
        holding_days=int(holding_days),
        rsi_buy=rsi_buy,
        rsi_sell=rsi_sell,
        use_sma_filter=use_sma,
        use_volatility_filter=use_vol_filter,
        vol_threshold=vol_threshold,
    )
    
    if "error" in result:
        empty_fig = _empty_plot(result["error"])
        return f"❌ {result['error']}", empty_fig, empty_fig, empty_fig, result
    
    # Build charts
    equity_chart = build_equity_curve(result)
    signal_chart = build_backtest_signals_chart(result)
    
    # Fetch price data for candlestick
    period_map = {0.5: "6mo", 1: "1y", 2: "2y", 3: "3y", 5: "5y"}
    data = fetch_ohlcv(ticker, period=period_map.get(period_years, "1y"))
    if data["df"] is not None:
        price_chart = build_candlestick_chart(data["df"], ticker)
    else:
        price_chart = _empty_plot("No price data")
    
    # Format summary markdown
    s = result["summary"]
    r = result["returns"]
    c = result["comparison"]
    
    acc_emoji = "🟢" if s["accuracy_pct"] >= 60 else ("🟡" if s["accuracy_pct"] >= 50 else "🔴")
    perf_emoji = "🟢" if r["total_return_pct"] > 0 else "🔴"
    vs_emoji = "🟢" if c["outperformance"] > 0 else "🔴"
    
    md = f"""## 📊 Backtest Sonuçları — {result['ticker']}
**Veri:** {result['data_source']} | **Dönem:** {result['backtest_period']} | **Tutma:** {result['holding_days']} gün

### {acc_emoji} Doğruluk: **%{s['accuracy_pct']}** ({s['total_signals']} sinyal)

| Metrik | Değer |
|---|---|
| 📈 AL Sinyalleri | {s['buy_signals']} (doğruluk: %{s['buy_accuracy_pct']}) |
| 📉 SAT Sinyalleri | {s['sell_signals']} (doğruluk: %{s['sell_accuracy_pct']}) |
| {perf_emoji} Toplam Getiri | **{r['total_return_pct']:+.1f}%** |
| 📊 Ort. Trade | {r['avg_return_per_trade']:+.2f}% |
| 🏆 En İyi | {r['max_single_return']:+.2f}% |
| 💀 En Kötü | {r['min_single_return']:+.2f}% |
| 📈 Sharpe | {r['sharpe_ratio']} |
| 📉 Max Drawdown | {r['max_drawdown_pct']:.1f}% |
| 💪 Profit Factor | {r['profit_factor']} |

### {vs_emoji} vs Buy & Hold
| Strateji | Getiri |
|---|---|
| 📊 Sinyal | **{c['strategy_return']:+.1f}%** |
| 🏠 Buy & Hold | {c['buy_hold_return']:+.1f}% |
| 🆚 Fark | **{c['outperformance']:+.1f}%** |

**Başlangıç:** ${result['initial_capital']:,.0f} → **Son:** ${result['final_capital']:,.0f}

⚠️ *{result['disclaimer']}*"""
    
    return md, equity_chart, signal_chart, price_chart, result


# ═══════════════════════════════════════════════════════
# TAB 3: COMPARE — Multi-ticker comparison
# ═══════════════════════════════════════════════════════

def compare_tickers(tickers_str, portfolio_value, risk_tolerance):
    """Compare 2-5 tickers side by side."""
    if not tickers_str or not tickers_str.strip():
        return "❌ En az 2 ticker girin (virgülle ayırın)", {}
    
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    if len(tickers) < 2:
        return "❌ En az 2 ticker gerekli", {}
    if len(tickers) > 5:
        return "❌ Max 5 ticker", {}
    
    results = {}
    for t in tickers:
        sig = generate_signal(t, portfolio_value, risk_tolerance.lower())
        results[t] = sig
    
    # Build comparison table
    md = f"## 🔄 Karşılaştırma: {', '.join(tickers)}\n\n"
    md += "| Metrik | " + " | ".join(tickers) + " |\n"
    md += "|---|" + "|".join(["---"] * len(tickers)) + "|\n"
    
    rows = [
        ("Sinyal", lambda s: f"**{s['signal']}**"),
        ("Güven", lambda s: f"%{s['confidence']}"),
        ("Risk", lambda s: s['risk']),
        ("Fiyat", lambda s: f"${s['price']:,.2f}"),
        ("RSI", lambda s: f"{s['indicators']['rsi']['value']}"),
        ("SMA Cross", lambda s: s['indicators']['sma_crossover']['signal']),
        ("MACD", lambda s: s['indicators']['macd']['signal']),
        ("Volatilite", lambda s: s['indicators']['volatility']['regime']),
        ("R/R", lambda s: s['trade'].get('risk_reward', 'N/A')),
    ]
    
    for label, fn in rows:
        md += f"| {label} | " + " | ".join(fn(results[t]) for t in tickers) + " |\n"
    
    # Structured comparison JSON
    comparison_json = {
        t: {
            "signal": results[t]["signal"],
            "confidence": results[t]["confidence"] / 100,
            "risk": results[t]["risk"],
            "price": results[t]["price"],
        }
        for t in tickers
    }
    
    md += "\n⚠️ *Yatırım tavsiyesi değildir.*"
    
    return md, comparison_json


# ═══════════════════════════════════════════════════════
# TAB 4: EXPLAINABILITY — "Why does this trade make sense?"
# ═══════════════════════════════════════════════════════

def explain_trade(ticker, portfolio_value, risk_tolerance):
    """Deep explainability: Why does this trade make sense?"""
    ticker = ticker.strip().upper()
    if not ticker:
        return "❌ Ticker girin"
    
    signal = generate_signal(ticker, portfolio_value, risk_tolerance.lower())
    
    exp = signal.get("explainability", {})
    votes = signal.get("votes", {})
    reasons = signal.get("reasons", [])
    trade = signal.get("trade", {})
    indicators = signal.get("indicators", {})
    
    sig_tr = {"STRONG_BUY": "GÜÇLÜ AL", "BUY": "AL", "HOLD": "BEKLE",
              "SELL": "SAT", "STRONG_SELL": "GÜÇLÜ SAT"}.get(signal["signal"], signal["signal"])
    
    md = f"""## 🧠 Explainability Report — {signal['ticker']}

### Bu trade neden {'mantıklı' if signal['signal'] in ('BUY','STRONG_BUY','SELL','STRONG_SELL') else 'şu an yapılmamalı'}?

**Tez:** {exp.get('thesis', 'N/A')}

**Destekleyen kanıt:** {exp.get('supporting_evidence', 'N/A')}

**Risk değerlendirmesi:** {exp.get('risk_assessment', 'N/A')}

---

### 📊 İndikatör Dökümü

| İndikatör | Değer | Sinyal | Yön |
|---|---|---|---|
| RSI(14) | {indicators['rsi']['value']} | {indicators['rsi']['signal']} | {'🐂' if 'BULLISH' in indicators['rsi']['signal'] or indicators['rsi']['signal'] == 'OVERSOLD' else '🐻' if 'BEARISH' in indicators['rsi']['signal'] or indicators['rsi']['signal'] == 'OVERBOUGHT' else '➖'} |
| SMA(20/50) | {indicators['sma_crossover']['sma20']}/{indicators['sma_crossover']['sma50']} | {indicators['sma_crossover']['signal']} | {'🐂' if 'BULLISH' in indicators['sma_crossover']['signal'] else '🐻' if 'BEARISH' in indicators['sma_crossover']['signal'] else '➖'} |
| MACD(12,26,9) | {indicators['macd']['histogram']} | {indicators['macd']['signal']} | {'🐂' if 'BULLISH' in indicators['macd']['signal'] else '🐻' if 'BEARISH' in indicators['macd']['signal'] else '➖'} |
| Bollinger(20,2) | — | {indicators['bollinger']['signal']} | {'🐂' if indicators['bollinger']['signal'] in ('BULLISH','OVERSOLD') else '🐻' if indicators['bollinger']['signal'] in ('BEARISH','OVERBOUGHT') else '➖'} |
| Volatilite | {indicators['volatility']['annualized_pct']}% | {indicators['volatility']['regime']} | {'⚠️' if indicators['volatility']['regime'] in ('HIGH','EXTREME') else '✅'} |

**Oylama:** {signal['bullish_count']} 🐂 boğa / {signal['bearish_count']} 🐻 ayı

---

### 📝 Karar Süreci (Kural Tabanlı)

1. RSI < 30 ve SMA20 > SMA50 → **AL** sinyali
2. RSI > 70 ve SMA20 < SMA50 → **SAT** sinyali
3. Extreme volatility → **AL sinyali filtrelenir** (HOLD'a düşer)
4. 5 indikatörden 3+ aynı yönde → sinyal güçlenir

**Sonuç:** {sig_tr} (Güven: %{signal['confidence']}, Risk: {signal['risk']})

---

### 💡 Nedenler

"""
    for i, r in enumerate(reasons, 1):
        md += f"{i}. {r}\n"
    
    if signal["signal"] in ("BUY", "STRONG_BUY") and trade.get("entry_price"):
        md += f"""
---

### 💰 Trade Detayları

| | |
|---|---|
| Giriş | **${trade['entry_price']:,.2f}** |
| Stop-Loss | **${trade['stop_loss']:,.2f}** |
| Take-Profit | **${trade['take_profit']:,.2f}** |
| Pozisyon | **{trade['shares']}** hisse (${trade['position_value']:,.0f}) |
| Risk/Ödül | **{trade['risk_reward']}** |
| Portföy %si | **%{trade['position_pct']}** |
"""
    
    md += "\n⚠️ *Algoritmik sinyal — yatırım tavsiyesi değildir.*"
    return md


# ═══════════════════════════════════════════════════════
# TAB: AI TRADE DECISION — signal → explanation → backtest → execute
# ═══════════════════════════════════════════════════════

def run_ai_trade_decision(tickers_str, portfolio_value, risk_tolerance):
    """Full AI Trade Decision pipeline: multi-ticker scan → signal → backtest → portfolio."""
    if not tickers_str or not tickers_str.strip():
        return "❌ Ticker girin", _empty_plot("No data"), {}

    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    if not tickers:
        return "❌ Ticker girin", _empty_plot("No data"), {}

    result = run_multi_ticker_simulation(tickers, portfolio_value, risk_tolerance.lower())
    decisions = result["decisions"]
    metrics = result["portfolio_metrics"]

    # Build decisions table
    md = "## 🤖 AI Trade Decisions\n\n"
    md += "| Ticker | Sinyal | Güven | Risk | Fiyat | BT Doğruluk | BT Sharpe | Aksiyon |\n"
    md += "|---|---|---|---|---|---|---|---|\n"

    for d in decisions:
        sig_e = {"STRONG_BUY": "🟢🟢", "BUY": "🟢", "HOLD": "🟡", "SELL": "🔴", "STRONG_SELL": "🔴🔴"}.get(d["signal"], "⚪")
        action = d["execution"].get("action", "NONE")
        act_e = "✅" if action == "OPENED" else ("🔒" if action == "CLOSED" else "➖")
        md += f"| **{d['ticker']}** | {sig_e} {d['signal']} | %{d['confidence']} | {d['risk']} | ${d['price']:,.2f} | %{d['backtest_accuracy']:.0f} | {d['backtest_sharpe']:.1f} | {act_e} {action} |\n"

    md += f"\n---\n### 💼 Portföy Durumu\n"
    md += f"| | |\n|---|---|\n"
    pnl_e = "🟢" if metrics["total_pnl"] >= 0 else "🔴"
    md += f"| 💰 Nakit | **${metrics['cash']:,.2f}** |\n"
    md += f"| 📊 Toplam Değer | **${metrics['equity']:,.2f}** |\n"
    md += f"| {pnl_e} P&L | **${metrics['total_pnl']:+,.2f}** ({metrics['total_pnl_pct']:+.1f}%) |\n"
    md += f"| 📈 Açık Pozisyon | **{metrics['open_positions']}** |\n"
    md += f"| 🔄 Toplam Trade | **{metrics['total_trades']}** |\n"
    md += f"| 🎯 Win Rate | **%{metrics['win_rate']}** |\n"
    md += f"| 📉 Max Drawdown | **{metrics['max_drawdown_pct']}%** |\n"

    # Explanation per decision
    md += "\n---\n### 🧠 Kararların Açıklaması\n\n"
    for d in decisions:
        sig_e = {"STRONG_BUY": "🟢🟢", "BUY": "🟢", "HOLD": "🟡", "SELL": "🔴", "STRONG_SELL": "🔴🔴"}.get(d["signal"], "⚪")
        md += f"**{sig_e} {d['ticker']}** — {d['signal']} (%{d['confidence']} güven)\n"
        md += f"- Neden: {d['reason']}\n"
        md += f"- İndikatörler: RSI={d['indicators']['rsi']:.1f}, SMA={d['indicators']['sma']}, MACD={d['indicators']['macd']}, Vol={d['indicators']['volatility']}\n"
        md += f"- Backtest: %{d['backtest_accuracy']:.0f} doğruluk, Sharpe {d['backtest_sharpe']:.2f}\n"
        act = d["execution"]
        if act.get("action") == "OPENED":
            md += f"- ✅ Pozisyon açıldı: {act['shares']} hisse @ ${act['price']:,.2f} (maliyet: ${act['cost']:,.2f})\n"
        elif act.get("action") == "CLOSED":
            md += f"- 🔒 Pozisyon kapatıldı: PnL ${act.get('pnl',0):+,.2f}\n"
        else:
            md += f"- ➖ İşlem yok: {act.get('reason', 'N/A')}\n"
        md += "\n"

    md += "⚠️ *Yatırım tavsiyesi değildir.*"

    # Portfolio chart
    portfolio = get_portfolio()
    port_chart = build_portfolio_equity_chart(portfolio.equity_snapshots)

    return md, port_chart, result


# ═══════════════════════════════════════════════════════
# TAB: EVALUATION DASHBOARD — son 50 sinyal, başarı oranı, PnL, drawdown
# ═══════════════════════════════════════════════════════

def run_evaluation_dashboard(tickers_str, period_years):
    """
    Run backtests on multiple tickers and build a comprehensive evaluation dashboard.
    Shows: last 50 signals, accuracy, PnL, drawdown.
    """
    if not tickers_str or not tickers_str.strip():
        ef = _empty_plot("Ticker girin")
        return "❌ Ticker girin", ef, ef, ef, {}

    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    if not tickers:
        ef = _empty_plot("Ticker girin")
        return "❌ Ticker girin", ef, ef, ef, {}

    all_signals = []
    combined_equity = [100000.0]
    capital = 100000.0
    ticker_stats = []

    for ticker in tickers:
        bt = run_backtest(ticker, period_years=period_years, holding_days=10)
        if "error" in bt:
            ticker_stats.append({"ticker": ticker, "error": bt["error"]})
            continue

        # Collect signals
        for s in bt["signals"]:
            s["ticker"] = ticker
            all_signals.append(s)

        # Track combined equity
        for s in bt["signals"]:
            pnl = capital * 0.02 * (s["return_pct"] / 100)
            capital += pnl
            combined_equity.append(capital)

        ticker_stats.append({
            "ticker": ticker,
            "signals": bt["summary"]["total_signals"],
            "accuracy": bt["summary"]["accuracy_pct"],
            "return": bt["returns"]["total_return_pct"],
            "sharpe": bt["returns"]["sharpe_ratio"],
            "max_dd": bt["returns"]["max_drawdown_pct"],
            "profit_factor": bt["returns"]["profit_factor"],
            "buy_acc": bt["summary"]["buy_accuracy_pct"],
            "sell_acc": bt["summary"]["sell_accuracy_pct"],
        })

    # Sort by date, take last 50
    all_signals.sort(key=lambda x: x["date"])
    last_50 = all_signals[-50:]

    # Stats
    total = len(all_signals)
    correct = sum(1 for s in all_signals if s["correct"])
    accuracy = correct / total * 100 if total else 0
    total_pnl_pct = (capital - 100000) / 100000 * 100

    # Drawdown
    eq = np.array(combined_equity)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak * 100
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0

    acc_e = "🟢" if accuracy >= 60 else ("🟡" if accuracy >= 50 else "🔴")
    pnl_e = "🟢" if total_pnl_pct > 0 else "🔴"

    # Build markdown
    md = f"## 📊 Evaluation Dashboard\n\n"
    md += f"### {acc_e} Genel Başarı: **%{accuracy:.1f}** ({correct}/{total} sinyal)\n"
    md += f"### {pnl_e} Toplam PnL: **{total_pnl_pct:+.1f}%** | Max Drawdown: **{max_dd:.1f}%**\n\n"

    # Per-ticker table
    md += "### Ticker Bazında Sonuçlar\n\n"
    md += "| Ticker | Sinyaller | Doğruluk | Getiri | Sharpe | Max DD | PF |\n"
    md += "|---|---|---|---|---|---|---|\n"
    for ts in ticker_stats:
        if "error" in ts:
            md += f"| {ts['ticker']} | ❌ | — | — | — | — | — |\n"
        else:
            a_e = "🟢" if ts["accuracy"] >= 60 else ("🟡" if ts["accuracy"] >= 50 else "🔴")
            md += f"| **{ts['ticker']}** | {ts['signals']} | {a_e} %{ts['accuracy']} | {ts['return']:+.1f}% | {ts['sharpe']} | {ts['max_dd']:.1f}% | {ts['profit_factor']} |\n"

    # Last 50 signals table
    md += f"\n### Son {len(last_50)} Sinyal\n\n"
    md += "| Tarih | Ticker | Sinyal | Giriş | Çıkış | Getiri | RSI | Sonuç |\n"
    md += "|---|---|---|---|---|---|---|---|\n"
    for s in last_50[-20:]:  # Show last 20 in table for readability
        sig_e = "📈" if s["signal"] == "BUY" else "📉"
        res_e = "✅" if s["correct"] else "❌"
        md += f"| {s['date']} | {s['ticker']} | {sig_e} {s['signal']} | ${s['entry_price']} | ${s['exit_price']} | {s['return_pct']:+.1f}% | {s['rsi']:.0f} | {res_e} |\n"

    if len(last_50) > 20:
        md += f"\n*... ve {len(last_50) - 20} sinyal daha*\n"

    md += "\n⚠️ *Geçmiş performans gelecek sonuçları garanti etmez.*"

    # Build charts
    # 1. Equity curve
    import plotly.graph_objects as go
    eq_fig = go.Figure()
    eq_fig.add_trace(go.Scatter(
        y=combined_equity, mode='lines',
        line=dict(color="#03DAC6" if capital >= 100000 else "#ef5350", width=2),
        fill='tozeroy', fillcolor='rgba(3,218,198,0.08)',
    ))
    eq_fig.add_hline(y=100000, line_dash="dash", line_color="gray", opacity=0.4)
    eq_fig.update_layout(
        title=f"💰 Birleşik Equity Curve — {total_pnl_pct:+.1f}%",
        height=300, template="plotly_dark",
        paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e",
        font=dict(color="#e0e0e0"), yaxis_title="$",
        margin=dict(l=50, r=20, t=60, b=30),
    )

    # 2. Drawdown
    dd_fig = go.Figure()
    dd_fig.add_trace(go.Scatter(
        y=dd, mode='lines', fill='tozeroy',
        line=dict(color="#ef5350", width=1.5),
        fillcolor='rgba(239,83,80,0.15)',
    ))
    dd_fig.update_layout(
        title=f"📉 Drawdown — Max: {max_dd:.1f}%",
        height=250, template="plotly_dark",
        paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e",
        font=dict(color="#e0e0e0"), yaxis_title="%",
        margin=dict(l=50, r=20, t=60, b=30),
    )

    # 3. Signal distribution
    buy_correct = sum(1 for s in all_signals if s["signal"] == "BUY" and s["correct"])
    buy_wrong = sum(1 for s in all_signals if s["signal"] == "BUY" and not s["correct"])
    sell_correct = sum(1 for s in all_signals if s["signal"] == "SELL" and s["correct"])
    sell_wrong = sum(1 for s in all_signals if s["signal"] == "SELL" and not s["correct"])

    dist_fig = go.Figure(data=[
        go.Bar(name='Doğru', x=['BUY', 'SELL'], y=[buy_correct, sell_correct], marker_color='#26a69a'),
        go.Bar(name='Yanlış', x=['BUY', 'SELL'], y=[buy_wrong, sell_wrong], marker_color='#ef5350'),
    ])
    dist_fig.update_layout(
        barmode='stack', title="📊 Sinyal Dağılımı",
        height=250, template="plotly_dark",
        paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e",
        font=dict(color="#e0e0e0"),
        margin=dict(l=50, r=20, t=60, b=30),
    )

    eval_json = {
        "total_signals": total,
        "accuracy_pct": round(accuracy, 1),
        "total_pnl_pct": round(total_pnl_pct, 1),
        "max_drawdown_pct": round(max_dd, 1),
        "tickers": ticker_stats,
        "last_50_signals": last_50,
    }

    return md, eq_fig, dd_fig, dist_fig, eval_json


# ═══════════════════════════════════════════════════════
# TAB: PORTFOLIO SIMULATION — stateful tracking
# ═══════════════════════════════════════════════════════

def run_portfolio_action(ticker, portfolio_value, risk_tolerance):
    """Generate signal and execute in portfolio."""
    ticker = ticker.strip().upper()
    if not ticker:
        return "❌ Ticker girin", _empty_plot("No data"), _empty_plot("No data"), ""

    sig = generate_signal(ticker, portfolio_value, risk_tolerance.lower())
    portfolio = get_portfolio(portfolio_value)
    exec_result = portfolio.execute_signal(sig)
    metrics = portfolio.get_metrics()

    sig_e = {"STRONG_BUY": "🟢🟢", "BUY": "🟢", "HOLD": "🟡", "SELL": "🔴", "STRONG_SELL": "🔴🔴"}.get(sig["signal"], "⚪")

    md = f"## {sig_e} {ticker} — {sig['signal']} (Güven: %{sig['confidence']})\n\n"
    md += f"**Neden:** {sig.get('reason', 'N/A')}\n\n"

    act = exec_result.get("action", "NONE")
    if act == "OPENED":
        md += f"✅ **Pozisyon açıldı:** {exec_result['shares']} hisse @ ${exec_result['price']:,.2f}\n"
        md += f"Maliyet: ${exec_result['cost']:,.2f} | Kalan nakit: ${exec_result['remaining_cash']:,.2f}\n"
    elif act == "CLOSED":
        md += f"🔒 **Pozisyon kapatıldı:** PnL ${exec_result.get('pnl',0):+,.2f} ({exec_result.get('pnl_pct',0):+.1f}%)\n"
    elif act == "REJECTED":
        md += f"⚠️ **Reddedildi:** {exec_result.get('reason', 'N/A')}\n"
    else:
        md += f"➖ **İşlem yok:** {exec_result.get('reason', sig['signal'])}\n"

    md += _build_portfolio_summary_md(metrics, portfolio)

    port_chart = build_portfolio_equity_chart(portfolio.equity_snapshots)
    pnl_chart = build_pnl_distribution_chart(portfolio.get_trade_history())
    dd_chart_html = _build_portfolio_positions_md(portfolio)

    return md, port_chart, pnl_chart, dd_chart_html


def get_portfolio_status():
    """Get current portfolio status."""
    portfolio = get_portfolio()
    metrics = portfolio.get_metrics()
    md = _build_portfolio_summary_md(metrics, portfolio)

    port_chart = build_portfolio_equity_chart(portfolio.equity_snapshots)
    pnl_chart = build_pnl_distribution_chart(portfolio.get_trade_history())
    positions_md = _build_portfolio_positions_md(portfolio)

    return md, port_chart, pnl_chart, positions_md


def reset_portfolio_ui():
    """Reset portfolio to initial state."""
    reset_portfolio(100000)
    return "✅ Portföy sıfırlandı ($100,000)", _empty_plot("Reset"), _empty_plot("Reset"), "*Portföy sıfırlandı*"


def _build_portfolio_summary_md(metrics, portfolio):
    pnl_e = "🟢" if metrics["total_pnl"] >= 0 else "🔴"
    md = f"\n---\n### 💼 Portföy Özeti\n\n"
    md += f"| Metrik | Değer |\n|---|---|\n"
    md += f"| 💰 Nakit | ${metrics['cash']:,.2f} |\n"
    md += f"| 📊 Toplam Değer | ${metrics['equity']:,.2f} |\n"
    md += f"| {pnl_e} P&L | ${metrics['total_pnl']:+,.2f} ({metrics['total_pnl_pct']:+.1f}%) |\n"
    md += f"| 📈 Açık Pozisyon | {metrics['open_positions']} |\n"
    md += f"| 🔄 Toplam Trade | {metrics['total_trades']} |\n"
    md += f"| 🎯 Win Rate | %{metrics['win_rate']} |\n"
    md += f"| 📉 Max Drawdown | {metrics['max_drawdown_pct']}% |\n"
    md += f"| 💪 Profit Factor | {metrics['profit_factor']} |\n"
    return md


def _build_portfolio_positions_md(portfolio):
    positions = portfolio.get_positions_detail()
    history = portfolio.get_trade_history(20)

    md = "### 📋 Açık Pozisyonlar\n\n"
    if positions:
        md += "| Ticker | Adet | Giriş | Maliyet | SL | TP | Sinyal |\n|---|---|---|---|---|---|---|\n"
        for p in positions:
            md += f"| **{p['ticker']}** | {p['shares']} | ${p['entry_price']:,.2f} | ${p['cost']:,.2f} | {p.get('stop_loss') or '—'} | {p.get('take_profit') or '—'} | {p.get('signal', '—')} |\n"
    else:
        md += "*Açık pozisyon yok.*\n"

    md += "\n### 📜 Son Trade'ler\n\n"
    if history:
        md += "| Ticker | Giriş | Çıkış | PnL | PnL% | Neden |\n|---|---|---|---|---|---|\n"
        for t in history[-10:]:
            pnl_e = "🟢" if t["pnl"] > 0 else "🔴"
            md += f"| **{t['ticker']}** | ${t['entry_price']:,.2f} | ${t['exit_price']:,.2f} | {pnl_e} ${t['pnl']:+,.2f} | {t['pnl_pct']:+.1f}% | {t['reason']} |\n"
    else:
        md += "*Henüz kapatılmış trade yok.*\n"

    return md



# ═══════════════════════════════════════════════════════
# HELPER
# ═══════════════════════════════════════════════════════

def _empty_plot(message="No data"):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16, color="gray"))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e",
        height=300,
    )
    return fig


# ═══════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════

HEADER = """# 🤖 Trade Bot Advisor v3.5 — Signal-First + Evaluation + Portfolio

| Bileşen | Rol |
|---|---|
| 📊 **Signal Engine** | Kural tabanlı indikatör oyu → sinyal üretir (RSI + SMA crossover + MACD + Bollinger + Volume) |
| 🤖 **AI Trade Decision** | Sinyal + açıklama + backtest + portföy execution — tam pipeline |
| 📈 **Evaluation Dashboard** | Son 50 sinyal, başarı oranı, PnL, drawdown — tek ekranda |
| 💼 **Portfolio Simulation** | Stateful portföy: pozisyon aç/kapat, P&L takibi, trade geçmişi |
| 🧠 **LLM Interpreter** | Sinyali yorumlar — **asla sinyal üretmez** |

> Demo her zaman çalışır (sample data fallback). **LLM = yorumlayıcı**, sinyal motoru = karar verici.
"""

TICKERS_HELP = """**Desteklenen:** 🇺🇸 AAPL, NVDA, MSFT | 🇹🇷 THYAO.IS, GARAN.IS | ₿ BTC-USD, ETH-USD | 🪙 GC=F (Altın)"""

with gr.Blocks(title="🤖 Trade Bot Advisor v3.0") as demo:
    gr.Markdown(HEADER)
    
    with gr.Tabs():
        
        # ════════════ TAB 1: QUICK ANALYSIS ════════════
        with gr.Tab("📊 Analiz", id="analysis"):
            gr.Markdown(f"### Tek hisse analizi — Sinyal + Grafikler + Yapılandırılmış Çıktı\n{TICKERS_HELP}")
            
            with gr.Row():
                with gr.Column(scale=1):
                    ticker_input = gr.Textbox(label="📌 Ticker", value="AAPL", placeholder="AAPL, THYAO.IS, BTC-USD…")
                    portfolio_input = gr.Number(label="💰 Portföy ($)", value=100000, minimum=1000)
                    risk_input = gr.Radio(label="⚖️ Risk Toleransı", choices=["Conservative", "Moderate", "Aggressive"], value="Moderate")
                    analyze_btn = gr.Button("🔍 Analiz Et", variant="primary", size="lg")
                    
                    with gr.Row():
                        signal_display = gr.Textbox(label="Sinyal", interactive=False)
                        confidence_display = gr.Textbox(label="Güven", interactive=False)
                    with gr.Row():
                        risk_display = gr.Textbox(label="Risk", interactive=False)
                        price_display = gr.Textbox(label="Fiyat", interactive=False)
                
                with gr.Column(scale=2):
                    candlestick_plot = gr.Plot(label="📈 Fiyat Grafiği")
                    indicator_plot = gr.Plot(label="📊 İndikatörler (RSI + MACD)")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🧾 Yapılandırılmış Çıktı (JSON)")
                    signal_json = gr.JSON(label="Signal JSON")
                with gr.Column(scale=1):
                    gr.Markdown("### 🧠 Yorum")
                    interpretation_md = gr.Markdown(value="*Analiz için butona tıklayın…*")
            
            analyze_btn.click(
                fn=run_analysis,
                inputs=[ticker_input, portfolio_input, risk_input],
                outputs=[candlestick_plot, indicator_plot, signal_json, interpretation_md,
                         signal_display, confidence_display, risk_display, price_display],
            )
        
        # ════════════ TAB 2: BACKTEST ════════════
        with gr.Tab("📊 Backtest", id="backtest"):
            gr.Markdown("### 📊 Strateji Backtesti — Geçmiş verilerde sinyal doğruluğu\nAynı RSI + SMA crossover + volatility filter stratejisini geçmiş veride test edin.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    bt_ticker = gr.Textbox(label="📌 Ticker", value="AAPL")
                    bt_period = gr.Slider(label="📅 Dönem (yıl)", minimum=0.5, maximum=5, step=0.5, value=1)
                    bt_holding = gr.Slider(label="⏱️ Tutma süresi (gün)", minimum=1, maximum=30, step=1, value=10)
                    
                    gr.Markdown("**RSI Eşikleri:**")
                    bt_rsi_buy = gr.Slider(label="📈 RSI AL eşiği", minimum=15, maximum=45, step=1, value=35)
                    bt_rsi_sell = gr.Slider(label="📉 RSI SAT eşiği", minimum=55, maximum=85, step=1, value=65)
                    
                    gr.Markdown("**Filtreler:**")
                    bt_sma = gr.Checkbox(label="✅ SMA(20/50) crossover filtresi", value=True)
                    bt_vol_filter = gr.Checkbox(label="✅ Volatility filtresi", value=True)
                    bt_vol_threshold = gr.Slider(label="Volatilite eşiği", minimum=0.2, maximum=0.8, step=0.05, value=0.50)
                    
                    bt_btn = gr.Button("📊 Backtest Çalıştır", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    bt_summary = gr.Markdown(value="*Parametreleri ayarlayın ve Backtest Çalıştır'a tıklayın…*")
                    bt_equity = gr.Plot(label="💰 Equity Curve")
                    bt_signals = gr.Plot(label="📍 Sinyal Haritası")
                    bt_price = gr.Plot(label="📈 Fiyat Grafiği")
                    bt_json = gr.JSON(label="📋 Detaylı Sonuçlar (JSON)")
            
            bt_btn.click(
                fn=run_backtest_ui,
                inputs=[bt_ticker, bt_period, bt_holding, bt_rsi_buy, bt_rsi_sell,
                        bt_sma, bt_vol_filter, bt_vol_threshold],
                outputs=[bt_summary, bt_equity, bt_signals, bt_price, bt_json],
            )
        
        # ════════════ TAB 3: COMPARE ════════════
        with gr.Tab("🔄 Karşılaştırma", id="compare"):
            gr.Markdown("### 2–5 ticker karşılaştırma\nVirgülle ayırın: `AAPL, NVDA, MSFT`")
            
            with gr.Row():
                with gr.Column(scale=1):
                    cmp_tickers = gr.Textbox(label="📌 Ticker'lar", value="AAPL, NVDA, MSFT")
                    cmp_portfolio = gr.Number(label="💰 Portföy ($)", value=100000, minimum=1000)
                    cmp_risk = gr.Radio(label="⚖️ Risk", choices=["Conservative", "Moderate", "Aggressive"], value="Moderate")
                    cmp_btn = gr.Button("🔄 Karşılaştır", variant="primary", size="lg")
                with gr.Column(scale=2):
                    cmp_md = gr.Markdown(value="*Ticker'ları girin ve karşılaştırın…*")
                    cmp_json = gr.JSON(label="Karşılaştırma JSON")
            
            cmp_btn.click(
                fn=compare_tickers,
                inputs=[cmp_tickers, cmp_portfolio, cmp_risk],
                outputs=[cmp_md, cmp_json],
            )
        
        # ════════════ TAB 4: EXPLAINABILITY ════════════
        with gr.Tab("🧠 Explainability", id="explain"):
            gr.Markdown("### Bu trade neden mantıklı?\nHer sinyalin arkasındaki mantığı detaylı görün.")
            
            with gr.Row():
                with gr.Column(scale=1):
                    exp_ticker = gr.Textbox(label="📌 Ticker", value="AAPL")
                    exp_portfolio = gr.Number(label="💰 Portföy ($)", value=100000, minimum=1000)
                    exp_risk = gr.Radio(label="⚖️ Risk", choices=["Conservative", "Moderate", "Aggressive"], value="Moderate")
                    exp_btn = gr.Button("🧠 Açıkla", variant="primary", size="lg")
                with gr.Column(scale=2):
                    exp_md = gr.Markdown(value="*Ticker girin ve Açıkla'ya tıklayın…*")
            
            exp_btn.click(
                fn=explain_trade,
                inputs=[exp_ticker, exp_portfolio, exp_risk],
                outputs=exp_md,
            )
        
        # ════════════ TAB: AI TRADE DECISION ════════════
        with gr.Tab("🤖 AI Decision", id="ai_decision"):
            gr.Markdown("### 🤖 AI Trade Decision Pipeline\nSinyal → Açıklama → Backtest doğrulama → Portföy execution — hepsi tek tuşla.\nVirgülle birden fazla ticker girin.")
            with gr.Row():
                with gr.Column(scale=1):
                    ai_tickers = gr.Textbox(label="📌 Ticker(lar)", value="AAPL, NVDA, MSFT", placeholder="Virgülle ayırın…")
                    ai_portfolio = gr.Number(label="💰 Portföy ($)", value=100000, minimum=1000)
                    ai_risk = gr.Radio(label="⚖️ Risk", choices=["Conservative", "Moderate", "Aggressive"], value="Moderate")
                    ai_btn = gr.Button("🤖 Trade Kararlarını Çalıştır", variant="primary", size="lg")
                with gr.Column(scale=2):
                    ai_md = gr.Markdown(value="*Ticker girin ve çalıştırın…*")
                    ai_chart = gr.Plot(label="💼 Portföy Equity")
                    ai_json = gr.JSON(label="📋 Detaylı Sonuçlar")
            ai_btn.click(
                fn=run_ai_trade_decision,
                inputs=[ai_tickers, ai_portfolio, ai_risk],
                outputs=[ai_md, ai_chart, ai_json],
            )

        # ════════════ TAB: EVALUATION DASHBOARD ════════════
        with gr.Tab("📊 Evaluation", id="evaluation"):
            gr.Markdown("### 📊 Evaluation Dashboard — Son 50 sinyal, başarı oranı, PnL, drawdown\nBirden fazla ticker backtest edip birleşik performans görün.")
            with gr.Row():
                with gr.Column(scale=1):
                    eval_tickers = gr.Textbox(label="📌 Ticker(lar)", value="AAPL, NVDA, MSFT, GOOGL, TSLA")
                    eval_period = gr.Slider(label="📅 Dönem (yıl)", minimum=0.5, maximum=5, step=0.5, value=2)
                    eval_btn = gr.Button("📊 Dashboard Oluştur", variant="primary", size="lg")
                with gr.Column(scale=2):
                    eval_md = gr.Markdown(value="*Ticker girin ve dashboard oluşturun…*")
            with gr.Row():
                eval_equity = gr.Plot(label="💰 Birleşik Equity Curve")
                eval_dd = gr.Plot(label="📉 Drawdown")
            with gr.Row():
                eval_dist = gr.Plot(label="📊 Sinyal Dağılımı")
                eval_json = gr.JSON(label="📋 Detaylı Veriler")
            eval_btn.click(
                fn=run_evaluation_dashboard,
                inputs=[eval_tickers, eval_period],
                outputs=[eval_md, eval_equity, eval_dd, eval_dist, eval_json],
            )

        # ════════════ TAB: PORTFOLIO SIMULATION ════════════
        with gr.Tab("💼 Portföy Sim", id="portfolio"):
            gr.Markdown("### 💼 Portfolio Simulation — Stateful trade tracking\nHer sinyal portföye işlenir. Pozisyonlar, P&L, drawdown takip edilir.")
            with gr.Row():
                with gr.Column(scale=1):
                    ps_ticker = gr.Textbox(label="📌 Ticker", value="AAPL")
                    ps_portfolio = gr.Number(label="💰 Portföy ($)", value=100000, minimum=1000)
                    ps_risk = gr.Radio(label="⚖️ Risk", choices=["Conservative", "Moderate", "Aggressive"], value="Moderate")
                    ps_trade_btn = gr.Button("📝 Sinyal + Trade Aç", variant="primary", size="lg")
                    ps_status_btn = gr.Button("📊 Portföy Durumu", variant="secondary")
                    ps_reset_btn = gr.Button("🗑️ Portföyü Sıfırla", variant="stop")
                with gr.Column(scale=2):
                    ps_md = gr.Markdown(value="*Ticker girin ve trade açın…*")
                    with gr.Row():
                        ps_equity = gr.Plot(label="💼 Portföy Equity")
                        ps_pnl = gr.Plot(label="📊 PnL Dağılımı")
                    ps_positions = gr.Markdown(value="*Portföy boş*")
            ps_trade_btn.click(
                fn=run_portfolio_action,
                inputs=[ps_ticker, ps_portfolio, ps_risk],
                outputs=[ps_md, ps_equity, ps_pnl, ps_positions],
            )
            ps_status_btn.click(
                fn=get_portfolio_status,
                outputs=[ps_md, ps_equity, ps_pnl, ps_positions],
            )
            ps_reset_btn.click(
                fn=reset_portfolio_ui,
                outputs=[ps_md, ps_equity, ps_pnl, ps_positions],
            )

        # ════════════ TAB 5: REHBER ════════════
        with gr.Tab("📖 Rehber"):
            gr.Markdown("""
# 📖 Trade Bot Advisor v3.0 — Rehber

## Mimari: Signal-First

```
Kullanıcı → Ticker girer
              ↓
         Signal Engine (kural tabanlı)
         ├── RSI(14) — momentum
         ├── SMA(20/50) crossover — trend
         ├── MACD(12,26,9) — momentum değişimi
         ├── Bollinger Bands(20,2) — fiyat bandı
         ├── Volume analizi — hacim onayı
         └── Volatility filter — extreme filtreleme
              ↓
         5 indikatör oyu → BUY/SELL/HOLD
              ↓
         Structured JSON output
              ↓
         LLM Interpreter (opsiyonel)
         └── Sinyali Türkçe/İngilizce yorumlar
              ↓
         Backtest ile doğrulama
```

## Sinyal Kuralları

| Koşul | Sinyal |
|---|---|
| RSI < 30 + SMA20 > SMA50 | **STRONG_BUY** |
| RSI < 35 + SMA20 > SMA50 | **BUY** |
| RSI > 70 + SMA20 < SMA50 | **STRONG_SELL** |
| RSI > 65 + SMA20 < SMA50 | **SELL** |
| Extreme volatility + BUY | → **HOLD** (filtre) |
| 4+/5 bullish | Sinyal güçlenir |

## Desteklenen Varlıklar

| Tür | Format | Örnekler |
|---|---|---|
| 🇺🇸 US/NASDAQ | `TICKER` | AAPL, NVDA, MSFT, TSLA |
| 🇹🇷 BIST | `TICKER.IS` | THYAO.IS, GARAN.IS, AKBNK.IS |
| ₿ Kripto | `TICKER-USD` | BTC-USD, ETH-USD, SOL-USD |
| 🪙 Emtia | Futures | GC=F (Altın), SI=F (Gümüş) |

## v3.0 vs v2.0

| | v2.0 | v3.0 |
|---|---|---|
| Sinyal kaynağı | LLM council | Kural tabanlı motor |
| Output | Serbest metin | Yapılandırılmış JSON |
| Backtest | Basit | Equity curve + signal map |
| Explainability | Yok | Tam dökümanlı |
| Demo reliability | API'ye bağımlı | Sample data fallback |
| LLM rolü | Karar verici | Sadece yorumlayıcı |

## Structured Output Formatı

```json
{
  "signal": "BUY",
  "confidence": 0.62,
  "reason": "RSI oversold at 28.5 — potential bounce opportunity",
  "risk": "medium",
  "indicators": {
    "rsi": 28.5,
    "rsi_signal": "OVERSOLD",
    "sma_crossover": "BULLISH",
    "macd": "BULLISH_CROSSOVER",
    "bollinger": "OVERSOLD",
    "volatility": "MODERATE"
  },
  "trade": {
    "entry_price": 230.50,
    "stop_loss": 222.40,
    "take_profit": 242.65,
    "shares": 25,
    "risk_reward": "1:1.5"
  }
}
```

⚠️ **DISCLAIMER:** Bu sistem eğitim ve araştırma amaçlıdır. Yatırım tavsiyesi değildir.
""")
    
    gr.Markdown("---\n**v3.5** | Signal Engine + AI Decision + Evaluation Dashboard + Portfolio Sim + Backtester + Explainability | ⚠️ Yatırım tavsiyesi değildir")


if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
