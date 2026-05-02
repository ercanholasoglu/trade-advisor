"""
🤖 Trade Bot Advisor v3.5 — Signal-First + Evaluation + Portfolio
====================================================================
Architecture:
  Signal Engine (deterministic) → THE decision maker
  LLM → interpreter/explainer ONLY (never generates signals)
  Backtesting → proves the strategy works on historical data
  Evaluation Dashboard → son 50 sinyal, başarı oranı, PnL, drawdown
  Portfolio Simulation → stateful trade tracking
  Structured output → JSON, not "Looks bullish..."
  Sample data fallback → Space ALWAYS works
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
from core.evaluator import run_ground_truth_evaluation
from core.agents import run_multi_agent_analysis
from core.optimizer import compute_indicator_accuracy, optimize_for_tickers
from core.reasoning import generate_advanced_reasoning
from core.council_ui import run_llm_council_ui


# ═══════════════════════════════════════════════════════
# TAB 1: QUICK ANALYSIS
# ═══════════════════════════════════════════════════════

def run_analysis(ticker, portfolio_value, risk_tolerance):
    ticker = ticker.strip().upper()
    if not ticker:
        empty_fig = _empty_plot("Enter a ticker")
        return (empty_fig, empty_fig, {"error": "No ticker provided"},
                "❌ Ticker girin", "N/A", "N/A", "N/A", "N/A")

    signal = generate_signal(ticker, portfolio_value, risk_tolerance.lower(), period="3mo")
    result = fetch_ohlcv(ticker, period="3mo", interval="1d")
    df = result.get("df")

    if df is not None and len(df) >= 10:
        candlestick = build_candlestick_chart(df, ticker)
        indicators = build_indicator_panel(df, ticker)
    else:
        candlestick = _empty_plot(f"No chart data for {ticker}")
        indicators = _empty_plot("No indicator data")

    structured = {
        "signal": signal["signal"],
        "confidence": signal["confidence"] / 100,
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

    interpretation = _template_interpret(signal)
    sig_emoji = {"STRONG_BUY": "🟢🟢", "BUY": "🟢", "HOLD": "🟡", "SELL": "🔴", "STRONG_SELL": "🔴🔴"}.get(signal["signal"], "⚪")

    return (
        candlestick, indicators, structured, interpretation,
        f"{sig_emoji} {signal['signal']}", f"{signal['confidence']}%",
        signal["risk"].upper(),
        f"${signal['price']:,.2f}" if signal.get("price") else "N/A",
    )


# ═══════════════════════════════════════════════════════
# TAB 2: BACKTEST
# ═══════════════════════════════════════════════════════

def run_backtest_ui(ticker, period_years, holding_days, rsi_buy, rsi_sell,
                     use_sma, use_vol_filter, vol_threshold):
    ticker = ticker.strip().upper()
    if not ticker:
        ef = _empty_plot("Enter a ticker")
        return "❌ Ticker girin", ef, ef, ef, {}

    result = run_backtest(ticker, period_years=period_years, holding_days=int(holding_days),
                          rsi_buy=rsi_buy, rsi_sell=rsi_sell, use_sma_filter=use_sma,
                          use_volatility_filter=use_vol_filter, vol_threshold=vol_threshold)

    if "error" in result:
        ef = _empty_plot(result["error"])
        return f"❌ {result['error']}", ef, ef, ef, result

    equity_chart = build_equity_curve(result)
    signal_chart = build_backtest_signals_chart(result)
    period_map = {0.5: "6mo", 1: "1y", 2: "2y", 3: "3y", 5: "5y"}
    data = fetch_ohlcv(ticker, period=period_map.get(period_years, "1y"))
    price_chart = build_candlestick_chart(data["df"], ticker) if data["df"] is not None else _empty_plot("No price data")

    s, r, c = result["summary"], result["returns"], result["comparison"]
    acc_e = "🟢" if s["accuracy_pct"] >= 60 else ("🟡" if s["accuracy_pct"] >= 50 else "🔴")
    perf_e = "🟢" if r["total_return_pct"] > 0 else "🔴"
    vs_e = "🟢" if c["outperformance"] > 0 else "🔴"

    md = f"""## 📊 Backtest — {result['ticker']}
**Veri:** {result['data_source']} | **Dönem:** {result['backtest_period']} | **Tutma:** {result['holding_days']} gün

### {acc_e} Doğruluk: **%{s['accuracy_pct']}** ({s['total_signals']} sinyal)

| Metrik | Değer |
|---|---|
| 📈 AL | {s['buy_signals']} (%{s['buy_accuracy_pct']}) |
| 📉 SAT | {s['sell_signals']} (%{s['sell_accuracy_pct']}) |
| {perf_e} Toplam | **{r['total_return_pct']:+.1f}%** |
| 📊 Ort. Trade | {r['avg_return_per_trade']:+.2f}% |
| 📈 Sharpe | {r['sharpe_ratio']} |
| 📉 Max DD | {r['max_drawdown_pct']:.1f}% |
| 💪 PF | {r['profit_factor']} |

### {vs_e} vs Buy & Hold: Sinyal **{c['strategy_return']:+.1f}%** vs B&H {c['buy_hold_return']:+.1f}% (fark: **{c['outperformance']:+.1f}%**)

**${result['initial_capital']:,.0f} → ${result['final_capital']:,.0f}**

⚠️ *{result['disclaimer']}*"""
    return md, equity_chart, signal_chart, price_chart, result


# ═══════════════════════════════════════════════════════
# TAB 3: COMPARE
# ═══════════════════════════════════════════════════════

def compare_tickers(tickers_str, portfolio_value, risk_tolerance):
    if not tickers_str or not tickers_str.strip():
        return "❌ En az 2 ticker girin", {}
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    if len(tickers) < 2: return "❌ En az 2 ticker gerekli", {}
    if len(tickers) > 5: return "❌ Max 5 ticker", {}

    results = {t: generate_signal(t, portfolio_value, risk_tolerance.lower()) for t in tickers}
    md = f"## 🔄 Karşılaştırma: {', '.join(tickers)}\n\n"
    md += "| Metrik | " + " | ".join(tickers) + " |\n|---|" + "|".join(["---"]*len(tickers)) + "|\n"
    rows = [("Sinyal", lambda s: f"**{s['signal']}**"), ("Güven", lambda s: f"%{s['confidence']}"),
            ("Risk", lambda s: s['risk']), ("Fiyat", lambda s: f"${s['price']:,.2f}"),
            ("RSI", lambda s: f"{s['indicators']['rsi']['value']}"),
            ("SMA", lambda s: s['indicators']['sma_crossover']['signal']),
            ("MACD", lambda s: s['indicators']['macd']['signal']),
            ("Vol", lambda s: s['indicators']['volatility']['regime']),
            ("R/R", lambda s: s['trade'].get('risk_reward', 'N/A'))]
    for label, fn in rows:
        md += f"| {label} | " + " | ".join(fn(results[t]) for t in tickers) + " |\n"
    md += "\n⚠️ *Yatırım tavsiyesi değildir.*"
    cj = {t: {"signal": results[t]["signal"], "confidence": results[t]["confidence"]/100, "risk": results[t]["risk"], "price": results[t]["price"]} for t in tickers}
    return md, cj


# ═══════════════════════════════════════════════════════
# TAB 4: EXPLAINABILITY
# ═══════════════════════════════════════════════════════

def explain_trade(ticker, portfolio_value, risk_tolerance):
    ticker = ticker.strip().upper()
    if not ticker: return "❌ Ticker girin"
    signal = generate_signal(ticker, portfolio_value, risk_tolerance.lower())
    exp, votes, reasons = signal.get("explainability",{}), signal.get("votes",{}), signal.get("reasons",[])
    trade, ind = signal.get("trade",{}), signal.get("indicators",{})
    sig_tr = {"STRONG_BUY":"GÜÇLÜ AL","BUY":"AL","HOLD":"BEKLE","SELL":"SAT","STRONG_SELL":"GÜÇLÜ SAT"}.get(signal["signal"], signal["signal"])

    def d(s, pos, neg): return '🐂' if any(x in s for x in pos) else ('🐻' if any(x in s for x in neg) else '➖')

    md = f"""## 🧠 Explainability — {signal['ticker']}

### Bu trade neden {'mantıklı' if signal['signal'] in ('BUY','STRONG_BUY','SELL','STRONG_SELL') else 'şu an yapılmamalı'}?

**Tez:** {exp.get('thesis','N/A')}
**Kanıt:** {exp.get('supporting_evidence','N/A')}
**Risk:** {exp.get('risk_assessment','N/A')}

---

| İndikatör | Değer | Sinyal | Yön |
|---|---|---|---|
| RSI(14) | {ind['rsi']['value']} | {ind['rsi']['signal']} | {d(ind['rsi']['signal'],['BULLISH','OVERSOLD'],['BEARISH','OVERBOUGHT'])} |
| SMA(20/50) | {ind['sma_crossover']['sma20']}/{ind['sma_crossover']['sma50']} | {ind['sma_crossover']['signal']} | {d(ind['sma_crossover']['signal'],['BULLISH'],['BEARISH'])} |
| MACD | {ind['macd']['histogram']} | {ind['macd']['signal']} | {d(ind['macd']['signal'],['BULLISH'],['BEARISH'])} |
| Bollinger | — | {ind['bollinger']['signal']} | {d(ind['bollinger']['signal'],['BULLISH','OVERSOLD'],['BEARISH','OVERBOUGHT'])} |
| Vol | {ind['volatility']['annualized_pct']}% | {ind['volatility']['regime']} | {'⚠️' if ind['volatility']['regime'] in ('HIGH','EXTREME') else '✅'} |

**Oy:** {signal['bullish_count']} 🐂 / {signal['bearish_count']} 🐻 → **{sig_tr}** (%{signal['confidence']}, risk: {signal['risk']})

"""
    for i, r in enumerate(reasons, 1): md += f"{i}. {r}\n"
    if signal["signal"] in ("BUY","STRONG_BUY") and trade.get("entry_price"):
        md += f"\n**Trade:** Giriş ${trade['entry_price']:,.2f} | SL ${trade['stop_loss']:,.2f} | TP ${trade['take_profit']:,.2f} | {trade['shares']} hisse | R/R {trade['risk_reward']}\n"
    md += "\n⚠️ *Algoritmik sinyal — yatırım tavsiyesi değildir.*"
    return md


# ═══════════════════════════════════════════════════════
# TAB: AI TRADE DECISION
# ═══════════════════════════════════════════════════════

def run_ai_trade_decision(tickers_str, portfolio_value, risk_tolerance):
    if not tickers_str or not tickers_str.strip():
        return "❌ Ticker girin", _empty_plot("No data"), {}
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    if not tickers: return "❌ Ticker girin", _empty_plot("No data"), {}

    result = run_multi_ticker_simulation(tickers, portfolio_value, risk_tolerance.lower())
    decisions, metrics = result["decisions"], result["portfolio_metrics"]

    md = "## 🤖 AI Trade Decisions\n\n"
    md += "| Ticker | Sinyal | Güven | Risk | Fiyat | BT Doğ | BT Sharpe | Aksiyon |\n|---|---|---|---|---|---|---|---|\n"
    for d in decisions:
        se = {"STRONG_BUY":"🟢🟢","BUY":"🟢","HOLD":"🟡","SELL":"🔴","STRONG_SELL":"🔴🔴"}.get(d["signal"],"⚪")
        act = d["execution"].get("action","NONE")
        ae = "✅" if act=="OPENED" else ("🔒" if act=="CLOSED" else "➖")
        md += f"| **{d['ticker']}** | {se} {d['signal']} | %{d['confidence']} | {d['risk']} | ${d['price']:,.2f} | %{d['backtest_accuracy']:.0f} | {d['backtest_sharpe']:.1f} | {ae} {act} |\n"

    pe = "🟢" if metrics["total_pnl"]>=0 else "🔴"
    md += f"\n---\n### 💼 Portföy\n| | |\n|---|---|\n| Nakit | ${metrics['cash']:,.2f} |\n| Değer | ${metrics['equity']:,.2f} |\n| {pe} P&L | ${metrics['total_pnl']:+,.2f} ({metrics['total_pnl_pct']:+.1f}%) |\n| Pozisyon | {metrics['open_positions']} |\n| Trade | {metrics['total_trades']} |\n| Win Rate | %{metrics['win_rate']} |\n| Max DD | {metrics['max_drawdown_pct']}% |\n"

    md += "\n---\n### 🧠 Açıklamalar\n\n"
    for d in decisions:
        se = {"STRONG_BUY":"🟢🟢","BUY":"🟢","HOLD":"🟡","SELL":"🔴","STRONG_SELL":"🔴🔴"}.get(d["signal"],"⚪")
        md += f"**{se} {d['ticker']}** — {d['signal']} (%{d['confidence']})\n"
        md += f"- {d['reason']}\n- RSI={d['indicators']['rsi']:.1f}, SMA={d['indicators']['sma']}, MACD={d['indicators']['macd']}, Vol={d['indicators']['volatility']}\n"
        md += f"- Backtest: %{d['backtest_accuracy']:.0f}, Sharpe {d['backtest_sharpe']:.2f}\n"
        a = d["execution"]
        if a.get("action")=="OPENED": md += f"- ✅ {a['shares']} hisse @ ${a['price']:,.2f}\n"
        elif a.get("action")=="CLOSED": md += f"- 🔒 PnL ${a.get('pnl',0):+,.2f}\n"
        else: md += f"- ➖ {a.get('reason','N/A')}\n"
        md += "\n"
    md += "⚠️ *Yatırım tavsiyesi değildir.*"

    port_chart = build_portfolio_equity_chart(get_portfolio().equity_snapshots)
    return md, port_chart, result


# ═══════════════════════════════════════════════════════
# TAB: EVALUATION DASHBOARD
# ═══════════════════════════════════════════════════════

def run_evaluation_dashboard(tickers_str, period_years):
    if not tickers_str or not tickers_str.strip():
        ef = _empty_plot("Ticker girin")
        return "❌ Ticker girin", ef, ef, ef, {}
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    if not tickers:
        ef = _empty_plot("Ticker girin")
        return "❌ Ticker girin", ef, ef, ef, {}

    import plotly.graph_objects as go
    all_signals, combined_equity, capital, ticker_stats = [], [100000.0], 100000.0, []

    for ticker in tickers:
        bt = run_backtest(ticker, period_years=period_years, holding_days=10)
        if "error" in bt:
            ticker_stats.append({"ticker": ticker, "error": bt["error"]}); continue
        for s in bt["signals"]: s["ticker"] = ticker; all_signals.append(s)
        for s in bt["signals"]:
            pnl = capital * 0.02 * (s["return_pct"] / 100); capital += pnl; combined_equity.append(capital)
        ticker_stats.append({"ticker": ticker, "signals": bt["summary"]["total_signals"],
            "accuracy": bt["summary"]["accuracy_pct"], "return": bt["returns"]["total_return_pct"],
            "sharpe": bt["returns"]["sharpe_ratio"], "max_dd": bt["returns"]["max_drawdown_pct"],
            "profit_factor": bt["returns"]["profit_factor"],
            "buy_acc": bt["summary"]["buy_accuracy_pct"], "sell_acc": bt["summary"]["sell_accuracy_pct"]})

    all_signals.sort(key=lambda x: x["date"]); last_50 = all_signals[-50:]
    total = len(all_signals); correct = sum(1 for s in all_signals if s["correct"])
    accuracy = correct/total*100 if total else 0; total_pnl_pct = (capital-100000)/100000*100
    eq = np.array(combined_equity); peak = np.maximum.accumulate(eq)
    dd = (eq-peak)/peak*100; max_dd = float(np.min(dd)) if len(dd)>0 else 0
    acc_e = "🟢" if accuracy>=60 else ("🟡" if accuracy>=50 else "🔴")
    pnl_e = "🟢" if total_pnl_pct>0 else "🔴"

    md = f"## 📊 Evaluation Dashboard\n\n### {acc_e} Başarı: **%{accuracy:.1f}** ({correct}/{total})\n### {pnl_e} PnL: **{total_pnl_pct:+.1f}%** | Max DD: **{max_dd:.1f}%**\n\n"
    md += "### Ticker Sonuçları\n\n| Ticker | Sinyal | Doğruluk | Getiri | Sharpe | Max DD | PF |\n|---|---|---|---|---|---|---|\n"
    for ts in ticker_stats:
        if "error" in ts: md += f"| {ts['ticker']} | ❌ | — | — | — | — | — |\n"
        else:
            ae = "🟢" if ts["accuracy"]>=60 else ("🟡" if ts["accuracy"]>=50 else "🔴")
            md += f"| **{ts['ticker']}** | {ts['signals']} | {ae} %{ts['accuracy']} | {ts['return']:+.1f}% | {ts['sharpe']} | {ts['max_dd']:.1f}% | {ts['profit_factor']} |\n"
    md += f"\n### Son {len(last_50)} Sinyal\n\n| Tarih | Ticker | Sinyal | Giriş | Çıkış | Getiri | RSI | Sonuç |\n|---|---|---|---|---|---|---|---|\n"
    for s in last_50[-20:]:
        se = "📈" if s["signal"]=="BUY" else "📉"; re = "✅" if s["correct"] else "❌"
        md += f"| {s['date']} | {s['ticker']} | {se} {s['signal']} | ${s['entry_price']} | ${s['exit_price']} | {s['return_pct']:+.1f}% | {s['rsi']:.0f} | {re} |\n"
    if len(last_50)>20: md += f"\n*...ve {len(last_50)-20} sinyal daha*\n"
    md += "\n⚠️ *Geçmiş performans gelecek sonuçları garanti etmez.*"

    eq_fig = go.Figure(); eq_fig.add_trace(go.Scatter(y=combined_equity, mode='lines', line=dict(color="#03DAC6" if capital>=100000 else "#ef5350", width=2), fill='tozeroy', fillcolor='rgba(3,218,198,0.08)'))
    eq_fig.add_hline(y=100000, line_dash="dash", line_color="gray", opacity=0.4)
    eq_fig.update_layout(title=f"💰 Equity — {total_pnl_pct:+.1f}%", height=300, template="plotly_dark", paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e", font=dict(color="#e0e0e0"), yaxis_title="$", margin=dict(l=50,r=20,t=60,b=30))

    dd_fig = go.Figure(); dd_fig.add_trace(go.Scatter(y=dd, mode='lines', fill='tozeroy', line=dict(color="#ef5350", width=1.5), fillcolor='rgba(239,83,80,0.15)'))
    dd_fig.update_layout(title=f"📉 Drawdown — Max: {max_dd:.1f}%", height=250, template="plotly_dark", paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e", font=dict(color="#e0e0e0"), yaxis_title="%", margin=dict(l=50,r=20,t=60,b=30))

    bc = sum(1 for s in all_signals if s["signal"]=="BUY" and s["correct"]); bw = sum(1 for s in all_signals if s["signal"]=="BUY" and not s["correct"])
    sc = sum(1 for s in all_signals if s["signal"]=="SELL" and s["correct"]); sw = sum(1 for s in all_signals if s["signal"]=="SELL" and not s["correct"])
    dist_fig = go.Figure(data=[go.Bar(name='Doğru', x=['BUY','SELL'], y=[bc,sc], marker_color='#26a69a'), go.Bar(name='Yanlış', x=['BUY','SELL'], y=[bw,sw], marker_color='#ef5350')])
    dist_fig.update_layout(barmode='stack', title="📊 Sinyal Dağılımı", height=250, template="plotly_dark", paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e", font=dict(color="#e0e0e0"), margin=dict(l=50,r=20,t=60,b=30))

    return md, eq_fig, dd_fig, dist_fig, {"total_signals": total, "accuracy_pct": round(accuracy,1), "total_pnl_pct": round(total_pnl_pct,1), "max_drawdown_pct": round(max_dd,1), "tickers": ticker_stats}


# ═══════════════════════════════════════════════════════
# TAB: PORTFOLIO SIMULATION
# ═══════════════════════════════════════════════════════

def run_portfolio_action(ticker, portfolio_value, risk_tolerance):
    ticker = ticker.strip().upper()
    if not ticker: return "❌ Ticker girin", _empty_plot("No data"), _empty_plot("No data"), ""
    sig = generate_signal(ticker, portfolio_value, risk_tolerance.lower())
    portfolio = get_portfolio(portfolio_value)
    exec_result = portfolio.execute_signal(sig)
    metrics = portfolio.get_metrics()
    se = {"STRONG_BUY":"🟢🟢","BUY":"🟢","HOLD":"🟡","SELL":"🔴","STRONG_SELL":"🔴🔴"}.get(sig["signal"],"⚪")
    md = f"## {se} {ticker} — {sig['signal']} (%{sig['confidence']})\n\n**Neden:** {sig.get('reason','N/A')}\n\n"
    act = exec_result.get("action","NONE")
    if act=="OPENED": md += f"✅ **Açıldı:** {exec_result['shares']} hisse @ ${exec_result['price']:,.2f} | Maliyet: ${exec_result['cost']:,.2f}\n"
    elif act=="CLOSED": md += f"🔒 **Kapatıldı:** PnL ${exec_result.get('pnl',0):+,.2f}\n"
    elif act=="REJECTED": md += f"⚠️ **Red:** {exec_result.get('reason','N/A')}\n"
    else: md += f"➖ **İşlem yok:** {exec_result.get('reason', sig['signal'])}\n"
    md += _build_portfolio_md(metrics, portfolio)
    return md, build_portfolio_equity_chart(portfolio.equity_snapshots), build_pnl_distribution_chart(portfolio.get_trade_history()), _build_positions_md(portfolio)

def get_portfolio_status():
    p = get_portfolio(); m = p.get_metrics()
    return _build_portfolio_md(m, p), build_portfolio_equity_chart(p.equity_snapshots), build_pnl_distribution_chart(p.get_trade_history()), _build_positions_md(p)

def reset_portfolio_ui():
    reset_portfolio(100000)
    return "✅ Portföy sıfırlandı ($100,000)", _empty_plot("Reset"), _empty_plot("Reset"), "*Portföy sıfırlandı*"

def _build_portfolio_md(m, p):
    pe = "🟢" if m["total_pnl"]>=0 else "🔴"
    return f"\n---\n### 💼 Portföy\n| Metrik | Değer |\n|---|---|\n| Nakit | ${m['cash']:,.2f} |\n| Değer | ${m['equity']:,.2f} |\n| {pe} P&L | ${m['total_pnl']:+,.2f} ({m['total_pnl_pct']:+.1f}%) |\n| Pozisyon | {m['open_positions']} |\n| Trade | {m['total_trades']} |\n| Win Rate | %{m['win_rate']} |\n| Max DD | {m['max_drawdown_pct']}% |\n| PF | {m['profit_factor']} |\n"

def _build_positions_md(portfolio):
    pos = portfolio.get_positions_detail(); hist = portfolio.get_trade_history(20)
    md = "### 📋 Pozisyonlar\n\n"
    if pos:
        md += "| Ticker | Adet | Giriş | Maliyet | SL | TP |\n|---|---|---|---|---|---|\n"
        for p in pos: md += f"| **{p['ticker']}** | {p['shares']} | ${p['entry_price']:,.2f} | ${p['cost']:,.2f} | {p.get('stop_loss') or '—'} | {p.get('take_profit') or '—'} |\n"
    else: md += "*Boş*\n"
    md += "\n### 📜 Trade Geçmişi\n\n"
    if hist:
        md += "| Ticker | Giriş | Çıkış | PnL | PnL% | Neden |\n|---|---|---|---|---|---|\n"
        for t in hist[-10:]:
            pe = "🟢" if t["pnl"]>0 else "🔴"
            md += f"| **{t['ticker']}** | ${t['entry_price']:,.2f} | ${t['exit_price']:,.2f} | {pe} ${t['pnl']:+,.2f} | {t['pnl_pct']:+.1f}% | {t['reason']} |\n"
    else: md += "*Henüz yok*\n"
    return md


# ═══════════════════════════════════════════════════════
# TAB: GROUND TRUTH — Model ne dedi? Gerçek ne oldu?
# ═══════════════════════════════════════════════════════

def run_ground_truth_ui(tickers_str, period_years, horizon_5, horizon_10, horizon_20):
    """Run ground truth evaluation: compare predictions vs actual outcomes."""
    if not tickers_str or not tickers_str.strip():
        ef = _empty_plot("Ticker girin")
        return "❌ Ticker girin", ef, ef, {}

    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    horizons = []
    if horizon_5: horizons.append(5)
    if horizon_10: horizons.append(10)
    if horizon_20: horizons.append(20)
    if not horizons: horizons = [10]

    import plotly.graph_objects as go

    all_evals = []
    ticker_results = []

    for ticker in tickers:
        r = run_ground_truth_evaluation(ticker, period_years=period_years, eval_horizons=horizons)
        if "error" in r:
            ticker_results.append({"ticker": ticker, "error": r["error"]})
            continue
        ticker_results.append(r)
        for ev in r["evaluations"]:
            ev["ticker"] = ticker
            all_evals.append(ev)

    if not all_evals:
        ef = _empty_plot("Sinyal bulunamadı")
        return "❌ Hiç sinyal bulunamadı — dönem veya parametreleri değiştirin.", ef, ef, {}

    # Pick primary horizon for display
    prim = f"{horizons[len(horizons)//2]}d"

    # ── Build the main table: Model ne dedi? Gerçek ne oldu? ──
    md = "## 🎯 Ground Truth Evaluation\n### Model ne dedi? Gerçek ne oldu?\n\n"

    # Aggregate stats per horizon
    for h in horizons:
        hk = f"{h}d"
        total = sum(1 for e in all_evals if hk in e["outcomes"])
        correct = sum(1 for e in all_evals if hk in e["outcomes"] and e["outcomes"][hk]["correct"])
        acc = correct / total * 100 if total else 0
        dir_rets = []
        for e in all_evals:
            if hk not in e["outcomes"]: continue
            chg = e["outcomes"][hk]["change_pct"]
            dir_rets.append(chg if e["signal"] == "BUY" else -chg)
        avg_dir = float(np.mean(dir_rets)) if dir_rets else 0
        ae = "🟢" if acc >= 60 else ("🟡" if acc >= 50 else "🔴")
        md += f"**{h} gün horizonu:** {ae} **%{acc:.1f}** doğruluk ({correct}/{total}) | Ort. yönlü getiri: **{avg_dir:+.2f}%**\n\n"

    # Per-ticker breakdown
    md += "---\n### Ticker Bazında\n\n"
    for tr in ticker_results:
        if "error" in tr:
            md += f"**{tr['ticker']}**: ❌ {tr['error']}\n\n"
            continue
        md += f"**{tr['ticker']}** ({tr['total_evaluations']} sinyal, kaynak: {tr['data_source']})\n\n"
        md += "| Horizon | Doğruluk | AL Doğ | SAT Doğ | Ort. Yönlü Getiri |\n|---|---|---|---|---|\n"
        for hk, hs in tr["horizon_stats"].items():
            ae = "🟢" if hs["accuracy_pct"] >= 60 else ("🟡" if hs["accuracy_pct"] >= 50 else "🔴")
            md += f"| {hk} | {ae} %{hs['accuracy_pct']} | %{hs['buy_accuracy_pct']} | %{hs['sell_accuracy_pct']} | {hs['avg_directional_return_pct']:+.2f}% |\n"
        cal = tr.get("calibration", {})
        if cal.get("high_conviction_signals", 0) > 0:
            md += f"\n*Kalibrasyon ({cal['reference_horizon']}): Yüksek kanaat %{cal['high_conviction_accuracy']}, Düşük kanaat %{cal['low_conviction_accuracy']}*\n"
        md += "\n"

    # ── Detailed signals table (last 30) ──
    all_evals.sort(key=lambda x: x["date"])
    show = all_evals[-30:]

    md += f"---\n### Son {len(show)} Sinyal — Tahmin vs Gerçek\n\n"
    md += "| Tarih | Ticker | Model Dedi | Fiyat | RSI |"
    for h in horizons:
        md += f" {h}G Gerçek | {h}G Sonuç |"
    md += "\n|---|---|---|---|---|"
    for _ in horizons:
        md += "---|---|"
    md += "\n"

    for e in show:
        se = "🟢 AL" if e["signal"] == "BUY" else "🔴 SAT"
        row = f"| {e['date']} | {e['ticker']} | **{se}** | ${e['price']:,.2f} | {e['rsi']:.0f} |"
        for h in horizons:
            hk = f"{h}d"
            o = e["outcomes"].get(hk, {})
            if o:
                chg = o["change_pct"]
                emoji = "✅" if o["correct"] else "❌"
                row += f" {chg:+.1f}% ({o['direction']}) | {emoji} |"
            else:
                row += " — | — |"
        md += row + "\n"

    md += "\n⚠️ *Geçmiş performans gelecek sonuçları garanti etmez.*"

    # ── Charts ──
    # 1. Accuracy per horizon bar chart
    acc_data = []
    for h in horizons:
        hk = f"{h}d"
        t = sum(1 for e in all_evals if hk in e["outcomes"])
        c = sum(1 for e in all_evals if hk in e["outcomes"] and e["outcomes"][hk]["correct"])
        acc_data.append({"horizon": f"{h} gün", "accuracy": c / t * 100 if t else 0})

    acc_fig = go.Figure(data=[go.Bar(
        x=[a["horizon"] for a in acc_data],
        y=[a["accuracy"] for a in acc_data],
        marker_color=["#26a69a" if a["accuracy"] >= 55 else "#ef5350" for a in acc_data],
        text=[f"%{a['accuracy']:.1f}" for a in acc_data],
        textposition="outside",
    )])
    acc_fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="Coin flip (50%)")
    acc_fig.update_layout(title="🎯 Doğruluk vs Horizon", height=300, template="plotly_dark",
                          paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e", font=dict(color="#e0e0e0"),
                          yaxis_title="Doğruluk %", yaxis_range=[0, 100],
                          margin=dict(l=50, r=20, t=60, b=30))

    # 2. Scatter: prediction outcome per signal (using primary horizon)
    buy_evals = [e for e in all_evals if e["signal"] == "BUY" and prim in e["outcomes"]]
    sell_evals = [e for e in all_evals if e["signal"] == "SELL" and prim in e["outcomes"]]

    scatter_fig = go.Figure()
    if buy_evals:
        scatter_fig.add_trace(go.Scatter(
            x=[e["rsi"] for e in buy_evals],
            y=[e["outcomes"][prim]["change_pct"] for e in buy_evals],
            mode="markers",
            marker=dict(size=8, color=["#26a69a" if e["outcomes"][prim]["correct"] else "#ef5350" for e in buy_evals],
                        symbol="triangle-up", line=dict(width=1, color="white")),
            name="BUY",
            text=[f"{e['ticker']} {e['date']}<br>Fiyat: ${e['price']}<br>{prim}: {e['outcomes'][prim]['change_pct']:+.1f}%<br>{'✅' if e['outcomes'][prim]['correct'] else '❌'}" for e in buy_evals],
            hoverinfo="text",
        ))
    if sell_evals:
        scatter_fig.add_trace(go.Scatter(
            x=[e["rsi"] for e in sell_evals],
            y=[e["outcomes"][prim]["change_pct"] for e in sell_evals],
            mode="markers",
            marker=dict(size=8, color=["#26a69a" if e["outcomes"][prim]["correct"] else "#ef5350" for e in sell_evals],
                        symbol="triangle-down", line=dict(width=1, color="white")),
            name="SELL",
            text=[f"{e['ticker']} {e['date']}<br>Fiyat: ${e['price']}<br>{prim}: {e['outcomes'][prim]['change_pct']:+.1f}%<br>{'✅' if e['outcomes'][prim]['correct'] else '❌'}" for e in sell_evals],
            hoverinfo="text",
        ))
    scatter_fig.add_hline(y=0, line_color="gray", opacity=0.3)
    scatter_fig.update_layout(title=f"📊 RSI vs Gerçek Getiri ({prim})", height=350, template="plotly_dark",
                              paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e", font=dict(color="#e0e0e0"),
                              xaxis_title="RSI (sinyal anında)", yaxis_title=f"Gerçek {prim} getiri %",
                              margin=dict(l=50, r=20, t=60, b=40))

    # JSON output
    out_json = {
        "tickers": [tr["ticker"] if "ticker" in tr else tr.get("ticker","?") for tr in ticker_results],
        "horizons": horizons,
        "total_signals": len(all_evals),
        "horizon_accuracy": {f"{h}d": round(sum(1 for e in all_evals if f"{h}d" in e["outcomes"] and e["outcomes"][f"{h}d"]["correct"]) / max(1, sum(1 for e in all_evals if f"{h}d" in e["outcomes"])) * 100, 1) for h in horizons},
    }

    return md, acc_fig, scatter_fig, out_json



# ═══════════════════════════════════════════════════════
# TAB: MULTI-AGENT SYSTEM
# ═══════════════════════════════════════════════════════

def run_multi_agent_ui(ticker, portfolio_value, risk_tolerance, w_analyst, w_sentiment, w_risk):
    """Run 4-agent analysis with custom weights."""
    ticker = ticker.strip().upper()
    if not ticker:
        return "Ticker girin", {}
    weights = {"Analyst": w_analyst, "Sentiment": w_sentiment, "Risk": w_risk}
    total = sum(weights.values())
    if total > 0:
        weights = {k: v/total for k, v in weights.items()}
    result = run_multi_agent_analysis(ticker, portfolio_value, risk_tolerance.lower(), weights=weights)
    d = result["decision"]
    agents = result["agents"]

    se = {"STRONG_BUY":"🟢🟢","BUY":"🟢","HOLD":"🟡","SELL":"🔴","STRONG_SELL":"🔴🔴"}.get(d["signal"],"⚪")
    md = f"## 🤖 Multi-Agent Karar: {se} **{d['signal']}** ({d['confidence']}%)\n"
    md += f"**Konsensus:** {d['consensus']} | **Risk Kararı:** {d['risk_verdict']} | **Skor:** {d['weighted_score']}\n"
    if d.get("override"):
        md += f"\n⚠️ **Override:** {d['override']}\n"

    md += "\n---\n### 🏛 Ajan Raporları\n\n| Ajan | Sinyal | Güven | Ağırlık | Katkı | Ana Neden |\n|---|---|---|---|---|---|\n"
    for a in d["agent_summaries"]:
        ae = "🟢" if a["signal"]=="BULLISH" else ("🔴" if a["signal"]=="BEARISH" else "🟡")
        md += f"| **{a['agent']}** | {ae} {a['signal']} | %{a['confidence']} | {a['weight']:.0%} | {a['contribution']:+.3f} | {a['top_reason'][:60]} |\n"

    for name, report in agents.items():
        icon = {"analyst":"📈","sentiment":"📰","risk":"⚠️"}.get(name,"🔹")
        md += f"\n#### {icon} {report['agent']}\n**Sinyal:** {report['signal']} ({report['confidence']}%)\n\n"
        for i, r in enumerate(report["reasoning"][:4], 1):
            md += f"{i}. {r}\n"

    md += "\n---\n### 🔗 Akıl Yürütme Zinciri\n\n"
    for line in d["reasoning_chain"]:
        md += f"- {line}\n"
    md += "\n⚠️ *Yatırım tavsiyesi değildir.*"
    return md, result


# ═══════════════════════════════════════════════════════
# TAB: STRATEGY OPTIMIZER
# ═══════════════════════════════════════════════════════

def run_optimizer_ui(tickers_str, period_years):
    """Run dynamic strategy optimization."""
    if not tickers_str or not tickers_str.strip():
        return "Ticker girin", _empty_plot("No data"), {}
    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

    import plotly.graph_objects as go

    if len(tickers) == 1:
        result = compute_indicator_accuracy(tickers[0], period_years)
        if "error" in result:
            return f"Hata: {result['error']}", _empty_plot("Error"), result
        ranking = result["ranking"]
        opt_w = result["optimized_weights"]
        def_w = result["default_weights"]
    else:
        result = optimize_for_tickers(tickers, period_years)
        if "error" in result:
            return f"Hata: {result['error']}", _empty_plot("Error"), result
        ranking = [{"indicator": k, "accuracy": v, "samples": "multi"} for k, v in result["average_accuracy"].items()]
        ranking.sort(key=lambda x: x["accuracy"], reverse=True)
        opt_w = result["average_weights"]
        def_w = {"RSI":0.2,"SMA":0.2,"MACD":0.2,"Bollinger":0.2,"Volume":0.2}

    md = f"## ⚙️ Dinamik Strateji Optimizasyonu\n\n"
    md += f"**Ticker:** {', '.join(tickers)} | **Dönem:** {period_years} yıl\n\n"
    md += "### İndikatör Performans Sıralaması\n\n| # | İndikatör | Doğruluk | Varsayılan Ağırlık | Optimize Ağırlık | Değişim |\n|---|---|---|---|---|---|\n"
    for i, r in enumerate(ranking, 1):
        name = r["indicator"]
        acc = r["accuracy"]
        dw = def_w.get(name, 0.2)
        ow = opt_w.get(name, 0.2)
        change = ow - dw
        ae = "🟢" if acc >= 55 else ("🟡" if acc >= 48 else "🔴")
        ce = "⬆️" if change > 0.02 else ("⬇️" if change < -0.02 else "➡️")
        md += f"| {i} | **{name}** | {ae} %{acc} | {dw:.0%} | **{ow:.0%}** | {ce} {change:+.0%} |\n"

    md += f"\n### Sonuç\n"
    best = ranking[0] if ranking else {"indicator":"?","accuracy":0}
    worst = ranking[-1] if ranking else {"indicator":"?","accuracy":0}
    md += f"- **En iyi:** {best['indicator']} (%{best['accuracy']} doğruluk) → ağırlık artırıldı\n"
    md += f"- **En kötü:** {worst['indicator']} (%{worst['accuracy']} doğruluk) → ağırlık azaltıldı\n"
    md += f"- Optimizasyon ground truth verisine dayanır — geçmiş {period_years} yıl analiz edildi\n"
    md += "\n⚠️ *Geçmiş performans gelecek sonuçları garanti etmez.*"

    # Chart: default vs optimized weights
    names = list(opt_w.keys())
    fig = go.Figure(data=[
        go.Bar(name='Varsayılan', x=names, y=[def_w.get(n,0.2)*100 for n in names], marker_color='#607D8B'),
        go.Bar(name='Optimize', x=names, y=[opt_w.get(n,0.2)*100 for n in names], marker_color='#03DAC6'),
    ])
    fig.update_layout(barmode='group', title="⚖️ İndikatör Ağırlıkları: Varsayılan vs Optimize",
                      height=350, template="plotly_dark", paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e",
                      font=dict(color="#e0e0e0"), yaxis_title="Ağırlık %",
                      margin=dict(l=50,r=20,t=60,b=30))
    return md, fig, result


# ═══════════════════════════════════════════════════════
# TAB: ADVANCED REASONING
# ═══════════════════════════════════════════════════════

def run_reasoning_ui(ticker):
    """Run advanced contextual reasoning."""
    ticker = ticker.strip().upper()
    if not ticker:
        return "Ticker girin", {}
    result = generate_advanced_reasoning(ticker)
    if "error" in result:
        return f"Hata: {result['error']}", result

    syn = result["synthesis"]
    se = {"STRONG_BUY":"🟢🟢","BUY":"🟢","HOLD":"🟡","SELL":"🔴","STRONG_SELL":"🔴🔴"}.get(result["signal"],"⚪")
    ve = {"STRENGTHENED":"💪","WEAKENED":"⚠️","CONFIRMED":"✅"}.get(syn["verdict"],"🔹")

    md = f"## 🧠 Gelişmiş Akıl Yürütme — {ticker}\n\n"
    md += f"### {se} Sinyal: **{result['signal']}** | {ve} Güven: **{syn['original_confidence']}% → {syn['adjusted_confidence']}%** ({syn['confidence_change']:+d})\n\n"
    md += f"**Sentez:** {syn['verdict']} — {syn['narrative']}\n\n"

    md += "---\n### 📋 Kontekstüel Analizler\n\n"
    for a in result["analyses"]:
        md += f"#### {a['icon']} {a['category']}: **{a['finding']}**\n"
        md += f"{a['detail']}\n\n"
        md += f"**Etki:** {a['impact']}\n\n"

    md += f"---\n### 📊 Özet\n"
    md += f"| | |\n|---|---|\n"
    md += f"| Boğa Faktörleri | {syn['bullish_factors']} |\n"
    md += f"| Ayı Faktörleri | {syn['bearish_factors']} |\n"
    md += f"| Uyarılar | {syn['warnings_count']} |\n"
    md += f"| Güven Değişimi | {syn['confidence_change']:+d}% |\n"
    md += "\n⚠️ *Yatırım tavsiyesi değildir.*"
    return md, result



# ═══════════════════════════════════════════════════════
# HELPER
# ═══════════════════════════════════════════════════════

def _empty_plot(message="No data"):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16, color="gray"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e", height=300)
    return fig


# ═══════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════

HEADER = """# 🤖 Trade Bot Advisor v3.5 — Signal-First + Evaluation + Portfolio

| Bileşen | Rol |
|---|---|
| 📊 **Signal Engine** | Kural tabanlı sinyal (RSI + SMA crossover + MACD + Bollinger + Volume) |
| 🤖 **AI Trade Decision** | Sinyal + açıklama + backtest + portföy execution |
| 📊 **Evaluation Dashboard** | Son 50 sinyal, başarı oranı, PnL, drawdown |
| 💼 **Portfolio Simulation** | Stateful portföy: pozisyon aç/kapat, P&L takibi |

> Demo her zaman çalışır (sample data fallback). **LLM = yorumlayıcı**, sinyal motoru = karar verici.
"""
TH = """**Desteklenen:** 🇺🇸 AAPL, NVDA, MSFT | 🇹🇷 THYAO.IS, GARAN.IS | ₿ BTC-USD | 🪙 GC=F"""

with gr.Blocks(title="🤖 Trade Bot Advisor v3.5") as demo:
    gr.Markdown(HEADER)
    with gr.Tabs():

        with gr.Tab("📊 Analiz"):
            gr.Markdown(f"### Tek hisse analiz\n{TH}")
            with gr.Row():
                with gr.Column(scale=1):
                    t1=gr.Textbox(label="Ticker",value="AAPL"); p1=gr.Number(label="Portföy ($)",value=100000,minimum=1000)
                    r1=gr.Radio(label="Risk",choices=["Conservative","Moderate","Aggressive"],value="Moderate")
                    b1=gr.Button("🔍 Analiz Et",variant="primary",size="lg")
                    with gr.Row(): sd=gr.Textbox(label="Sinyal",interactive=False); cd=gr.Textbox(label="Güven",interactive=False)
                    with gr.Row(): rd=gr.Textbox(label="Risk",interactive=False); pd_=gr.Textbox(label="Fiyat",interactive=False)
                with gr.Column(scale=2):
                    cp=gr.Plot(label="Fiyat"); ip=gr.Plot(label="İndikatörler")
            with gr.Row():
                with gr.Column(): sj=gr.JSON(label="Signal JSON")
                with gr.Column(): im=gr.Markdown(value="*Butona tıklayın…*")
            b1.click(run_analysis,[t1,p1,r1],[cp,ip,sj,im,sd,cd,rd,pd_])

        with gr.Tab("📊 Backtest"):
            gr.Markdown("### Strateji Backtesti")
            with gr.Row():
                with gr.Column(scale=1):
                    bt1=gr.Textbox(label="Ticker",value="AAPL"); bt2=gr.Slider(label="Dönem (yıl)",minimum=0.5,maximum=5,step=0.5,value=1)
                    bt3=gr.Slider(label="Tutma (gün)",minimum=1,maximum=30,step=1,value=10)
                    bt4=gr.Slider(label="RSI AL",minimum=15,maximum=45,step=1,value=35); bt5=gr.Slider(label="RSI SAT",minimum=55,maximum=85,step=1,value=65)
                    bt6=gr.Checkbox(label="SMA filtresi",value=True); bt7=gr.Checkbox(label="Vol filtresi",value=True)
                    bt8=gr.Slider(label="Vol eşik",minimum=0.2,maximum=0.8,step=0.05,value=0.50)
                    bb1=gr.Button("📊 Backtest",variant="primary",size="lg")
                with gr.Column(scale=2):
                    bm=gr.Markdown(value="*Parametreleri ayarlayın…*"); be=gr.Plot(label="Equity"); bs=gr.Plot(label="Sinyaller")
                    bpc=gr.Plot(label="Fiyat"); bj=gr.JSON(label="Sonuçlar")
            bb1.click(run_backtest_ui,[bt1,bt2,bt3,bt4,bt5,bt6,bt7,bt8],[bm,be,bs,bpc,bj])

        with gr.Tab("🔄 Karşılaştırma"):
            gr.Markdown("### 2–5 ticker karşılaştır")
            with gr.Row():
                with gr.Column(scale=1):
                    ct=gr.Textbox(label="Ticker'lar",value="AAPL, NVDA, MSFT"); cp2=gr.Number(label="Portföy ($)",value=100000,minimum=1000)
                    cr=gr.Radio(label="Risk",choices=["Conservative","Moderate","Aggressive"],value="Moderate")
                    cb=gr.Button("🔄 Karşılaştır",variant="primary",size="lg")
                with gr.Column(scale=2): cm=gr.Markdown(value="*Girin…*"); cjs=gr.JSON(label="JSON")
            cb.click(compare_tickers,[ct,cp2,cr],[cm,cjs])

        with gr.Tab("🧠 Explainability"):
            gr.Markdown("### Bu trade neden mantıklı?")
            with gr.Row():
                with gr.Column(scale=1):
                    et=gr.Textbox(label="Ticker",value="AAPL"); ep=gr.Number(label="Portföy ($)",value=100000,minimum=1000)
                    er=gr.Radio(label="Risk",choices=["Conservative","Moderate","Aggressive"],value="Moderate")
                    eb=gr.Button("🧠 Açıkla",variant="primary",size="lg")
                with gr.Column(scale=2): em=gr.Markdown(value="*Girin…*")
            eb.click(explain_trade,[et,ep,er],em)

        with gr.Tab("🤖 AI Decision"):
            gr.Markdown("### AI Trade Decision Pipeline\nSinyal → Backtest → Execute")
            with gr.Row():
                with gr.Column(scale=1):
                    at=gr.Textbox(label="Ticker(lar)",value="AAPL, NVDA, MSFT"); ap=gr.Number(label="Portföy ($)",value=100000,minimum=1000)
                    ar=gr.Radio(label="Risk",choices=["Conservative","Moderate","Aggressive"],value="Moderate")
                    ab=gr.Button("🤖 Çalıştır",variant="primary",size="lg")
                with gr.Column(scale=2): am=gr.Markdown(value="*Girin…*"); ac=gr.Plot(label="Portföy"); aj=gr.JSON(label="Sonuçlar")
            ab.click(run_ai_trade_decision,[at,ap,ar],[am,ac,aj])

        with gr.Tab("📊 Evaluation"):
            gr.Markdown("### Evaluation Dashboard — Son 50 sinyal, başarı, PnL, drawdown")
            with gr.Row():
                with gr.Column(scale=1):
                    evt=gr.Textbox(label="Ticker(lar)",value="AAPL, NVDA, MSFT, GOOGL, TSLA")
                    evp=gr.Slider(label="Dönem (yıl)",minimum=0.5,maximum=5,step=0.5,value=2)
                    evb=gr.Button("📊 Dashboard",variant="primary",size="lg")
                with gr.Column(scale=2): evm=gr.Markdown(value="*Girin…*")
            with gr.Row(): eve=gr.Plot(label="Equity"); evd=gr.Plot(label="Drawdown")
            with gr.Row(): evs=gr.Plot(label="Sinyal Dağılımı"); evj=gr.JSON(label="Veriler")
            evb.click(run_evaluation_dashboard,[evt,evp],[evm,eve,evd,evs,evj])

        with gr.Tab("💼 Portföy Sim"):
            gr.Markdown("### Portfolio Simulation — Stateful trading")
            with gr.Row():
                with gr.Column(scale=1):
                    pst=gr.Textbox(label="Ticker",value="AAPL"); psp=gr.Number(label="Portföy ($)",value=100000,minimum=1000)
                    psr=gr.Radio(label="Risk",choices=["Conservative","Moderate","Aggressive"],value="Moderate")
                    psb=gr.Button("📝 Trade Aç",variant="primary",size="lg")
                    pss=gr.Button("📊 Durum",variant="secondary"); psx=gr.Button("🗑️ Sıfırla",variant="stop")
                with gr.Column(scale=2):
                    psm=gr.Markdown(value="*Girin…*")
                    with gr.Row(): pse=gr.Plot(label="Equity"); pspnl=gr.Plot(label="PnL")
                    pspos=gr.Markdown(value="*Boş*")
            psb.click(run_portfolio_action,[pst,psp,psr],[psm,pse,pspnl,pspos])
            pss.click(get_portfolio_status,outputs=[psm,pse,pspnl,pspos])
            psx.click(reset_portfolio_ui,outputs=[psm,pse,pspnl,pspos])


        with gr.Tab("🎯 Ground Truth"):
            gr.Markdown("### 🎯 Model ne dedi? Gerçek ne oldu?\nGeçmiş sinyalleri gerçek fiyat hareketleriyle karşılaştır. Her sinyal için 5/10/20 gün sonra fiyat ne yaptı?")
            with gr.Row():
                with gr.Column(scale=1):
                    gt_tickers = gr.Textbox(label="Ticker(lar)", value="AAPL, NVDA, MSFT")
                    gt_period = gr.Slider(label="Dönem (yıl)", minimum=0.5, maximum=5, step=0.5, value=2)
                    gr.Markdown("**Değerlendirme horizonları:**")
                    gt_h5 = gr.Checkbox(label="5 gün", value=True)
                    gt_h10 = gr.Checkbox(label="10 gün", value=True)
                    gt_h20 = gr.Checkbox(label="20 gün", value=True)
                    gt_btn = gr.Button("🎯 Ground Truth Analizi", variant="primary", size="lg")
                with gr.Column(scale=2):
                    gt_md = gr.Markdown(value="*Ticker girin ve çalıştırın…*")
            with gr.Row():
                gt_acc = gr.Plot(label="🎯 Doğruluk vs Horizon")
                gt_scatter = gr.Plot(label="📊 RSI vs Gerçek Getiri")
            gt_json = gr.JSON(label="📋 Sonuçlar")
            gt_btn.click(run_ground_truth_ui, [gt_tickers, gt_period, gt_h5, gt_h10, gt_h20],
                        [gt_md, gt_acc, gt_scatter, gt_json])


        with gr.Tab("🤖 Multi-Agent"):
            gr.Markdown("### 4 Ajan Sistemi: Analist + Sentiment + Risk + Orchestrator\nHer ajan bağımsız analiz yapar, Orchestrator sentezler.")
            with gr.Row():
                with gr.Column(scale=1):
                    ma_t=gr.Textbox(label="Ticker",value="AAPL"); ma_p=gr.Number(label="Portfoy ($)",value=100000,minimum=1000)
                    ma_r=gr.Radio(label="Risk",choices=["Conservative","Moderate","Aggressive"],value="Moderate")
                    gr.Markdown("**Ajan Agirliklari:**")
                    ma_wa=gr.Slider(label="Analist",minimum=0.1,maximum=1.0,step=0.05,value=0.50)
                    ma_ws=gr.Slider(label="Sentiment",minimum=0.0,maximum=1.0,step=0.05,value=0.25)
                    ma_wr=gr.Slider(label="Risk",minimum=0.0,maximum=1.0,step=0.05,value=0.25)
                    ma_btn=gr.Button("🤖 Multi-Agent Analiz",variant="primary",size="lg")
                with gr.Column(scale=2):
                    ma_md=gr.Markdown(value="*Girin...*"); ma_json=gr.JSON(label="Detay")
            ma_btn.click(run_multi_agent_ui,[ma_t,ma_p,ma_r,ma_wa,ma_ws,ma_wr],[ma_md,ma_json])

        with gr.Tab("⚙️ Optimizer"):
            gr.Markdown("### Dinamik Strateji Optimizasyonu\nGround truth'a dayanarak hangi indikator daha basarili? Agirliklari otomatik ayarla.")
            with gr.Row():
                with gr.Column(scale=1):
                    op_t=gr.Textbox(label="Ticker(lar)",value="AAPL, NVDA, MSFT")
                    op_p=gr.Slider(label="Donem (yil)",minimum=0.5,maximum=5,step=0.5,value=2)
                    op_btn=gr.Button("⚙️ Optimize Et",variant="primary",size="lg")
                with gr.Column(scale=2):
                    op_md=gr.Markdown(value="*Girin...*"); op_fig=gr.Plot(label="Agirliklar"); op_json=gr.JSON(label="Detay")
            op_btn.click(run_optimizer_ui,[op_t,op_p],[op_md,op_fig,op_json])

        with gr.Tab("🧠 Reasoning"):
            gr.Markdown("### Gelismis Akil Yurutme\nLikidite tuzagi? Piyasa rejimi? Makro baglam? Teknik + dis faktorler birlestirilir.")
            with gr.Row():
                with gr.Column(scale=1):
                    re_t=gr.Textbox(label="Ticker",value="AAPL")
                    re_btn=gr.Button("🧠 Derin Analiz",variant="primary",size="lg")
                with gr.Column(scale=2):
                    re_md=gr.Markdown(value="*Girin...*"); re_json=gr.JSON(label="Detay")
            re_btn.click(run_reasoning_ui,[re_t],[re_md,re_json])

        with gr.Tab("🏛️ LLM Council"):
            gr.Markdown("### 🏛️ Gercek LLM Council — Bull vs Bear Tartismasi\n6 LLM ajan gercek zamanli tartisir, Fund Manager karar verir.\n**HF_TOKEN gerekli. ~30-60 sn surer.**")
            with gr.Row():
                with gr.Column(scale=1):
                    lc_t=gr.Textbox(label="Ticker",value="AAPL")
                    lc_ctx=gr.Textbox(label="Kontekst (opsiyonel)",placeholder="$100K, moderate risk...")
                    lc_btn=gr.Button("🏛️ Council Calistir",variant="primary",size="lg")
                with gr.Column(scale=2):
                    lc_md=gr.Markdown(value="*HF_TOKEN gerekli. Ticker girin...*")
                    lc_json=gr.JSON(label="Council JSON")
            lc_btn.click(run_llm_council_ui,[lc_t,lc_ctx],[lc_md,lc_json])

        with gr.Tab("📖 Rehber"):
            gr.Markdown("""# 📖 Rehber

## Signal Engine
RSI(14) + SMA(20/50) crossover + MACD(12,26,9) + Bollinger(20,2) + Volume → 5 indikatör oyu

## Sinyal Kuralları
| Koşul | Sinyal |
|---|---|
| 3+/5 bullish | BUY |
| 4+/5 bullish | STRONG_BUY |
| 3+/5 bearish | SELL |
| Extreme vol + BUY | → HOLD |

## Desteklenen
🇺🇸 AAPL, NVDA, MSFT | 🇹🇷 THYAO.IS, GARAN.IS | ₿ BTC-USD, ETH-USD | 🪙 GC=F

## 8 Tab
1. **Analiz** — Tek hisse sinyal + chart + JSON
2. **Backtest** — Geçmiş veride test
3. **Karşılaştırma** — Multi-ticker
4. **Explainability** — Neden bu trade?
5. **AI Decision** — Tam pipeline
6. **Evaluation** — Dashboard + drawdown
7. **Portföy Sim** — Stateful tracking
8. **Ground Truth** — Model ne dedi? Gerçek ne oldu?
9. **Rehber** — Bu sayfa

⚠️ Yatırım tavsiyesi değildir.
""")

    gr.Markdown("---\n**v3.5** | Signal Engine + AI Decision + Evaluation + Portfolio Sim + **Ground Truth** | ⚠️ Yatırım tavsiyesi değildir")

if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
        mcp_server=True,
    )
