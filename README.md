---
title: 🤖 Trade Bot Advisor v3.0
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.29.1"
app_file: app.py
pinned: true
license: mit
tags:
  - trading
  - finance
  - technical-analysis
  - backtesting
  - signals
---

# 🤖 Trade Bot Advisor v3.0 — Signal-First Architecture

> **Kural tabanlı sinyal motoru** • **Backtesting** • **Yapılandırılmış JSON çıktı** • **Explainability** • **LLM = sadece yorumlayıcı**

## v3.0 Mimari

```
Kullanıcı → Ticker girer
              ↓
         Signal Engine (kural tabanlı, deterministic)
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
         {"signal": "BUY", "confidence": 0.62, "reason": "...", "risk": "medium"}
              ↓
         LLM Interpreter (opsiyonel, sadece açıklama)
              ↓
         Backtest ile geçmiş veride doğrulama
```

## v2.0 → v3.0 Değişiklikler

| | v2.0 | v3.0 |
|---|---|---|
| **Sinyal kaynağı** | LLM council (karar verici) | Kural tabanlı motor (deterministic) |
| **LLM rolü** | Karar verici | Sadece yorumlayıcı |
| **Output** | Serbest metin ("Looks bullish...") | `{"signal": "BUY", "confidence": 0.62}` |
| **Backtest** | Basit doğruluk | Equity curve + signal map + Sharpe + drawdown |
| **Explainability** | Yok | "Bu trade neden mantıklı?" detaylı dökümanlı |
| **Demo reliability** | API'ye bağımlı (crash riski) | Sample data fallback (her zaman çalışır) |
| **Volatility filter** | Yok | Extreme volatilitede AL filtrelenir |

## 5 Tab

1. **📊 Analiz** — Sinyal + Candlestick + RSI/MACD + JSON + Yorum
2. **📊 Backtest** — Stratejiyi geçmiş veride test et (equity curve, signal map)
3. **🔄 Karşılaştırma** — 2-5 ticker yan yana
4. **🧠 Explainability** — "Bu trade neden mantıklı?" detaylı rapor
5. **📖 Rehber** — Kullanım kılavuzu

## Sinyal Kuralları

| Koşul | Sinyal |
|---|---|
| 3+/5 bullish indikatör | **BUY** |
| 4+/5 bullish | **STRONG_BUY** |
| 3+/5 bearish | **SELL** |
| 4+/5 bearish | **STRONG_SELL** |
| Extreme volatility + BUY → | **HOLD** (filtre) |

## Desteklenen Varlıklar

| Tür | Format | Örnekler |
|---|---|---|
| 🇺🇸 US Stocks | `TICKER` | AAPL, NVDA, MSFT, TSLA |
| 🇹🇷 BIST | `TICKER.IS` | THYAO.IS, GARAN.IS |
| ₿ Kripto | `TICKER-USD` | BTC-USD, ETH-USD |
| 🪙 Emtia | Futures | GC=F (Altın), SI=F (Gümüş) |

## Tech Stack

Python + Gradio + yfinance + NumPy + Pandas + Plotly

⚠️ **DISCLAIMER:** Eğitim ve araştırma amaçlıdır. Yatırım tavsiyesi değildir.
