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
)
from core.llm_interpreter import interpret_signal, _template_interpret


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

HEADER = """# 🤖 Trade Bot Advisor v3.0 — Signal-First Architecture

| Bileşen | Rol |
|---|---|
| 📊 **Signal Engine** | Kural tabanlı indikatör oyu → sinyal üretir (RSI + SMA crossover + MACD + Bollinger + Volume) |
| 🧠 **LLM Interpreter** | Sinyali Türkçe/İngilizce yorumlar — **asla sinyal üretmez** |
| 📈 **Backtester** | Stratejiyi geçmiş veride test eder — Sharpe, drawdown, vs buy&hold |
| 🛡️ **Volatility Filter** | Extreme volatilitede AL sinyallerini filtreler |
| 🧾 **Structured Output** | Her sinyal JSON formatında: `{"signal": "BUY", "confidence": 0.62, ...}` |

> **LLM = yorumlayıcı**, sinyal motoru = karar verici. Demo her zaman çalışır (sample data fallback).
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
    
    gr.Markdown("---\n**v3.0** | Signal Engine + Backtester + Structured Output + Explainability | ⚠️ Yatırım tavsiyesi değildir")


if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
