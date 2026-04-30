---
title: 🤖 Trade Bot Advisor v3.5
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
  - portfolio-simulation
  - signals
---

# 🤖 Trade Bot Advisor v3.5 — Signal-First + Evaluation + Portfolio

> **Signal Engine** • **AI Trade Decision** • **Evaluation Dashboard** • **Portfolio Simulation** • **Backtesting** • **Explainability**

## 8 Tab

| Tab | Özellik |
|---|---|
| 📊 **Analiz** | Tek hisse sinyal + candlestick + RSI/MACD + JSON + yorum |
| 📊 **Backtest** | RSI+SMA stratejisini geçmiş veride test — equity curve, Sharpe, drawdown |
| 🔄 **Karşılaştırma** | 2-5 ticker yan yana karşılaştır |
| 🧠 **Explainability** | "Bu trade neden mantıklı?" — tam indikatör dökümü |
| 🤖 **AI Decision** | **YENİ** — Sinyal → açıklama → backtest doğrulama → portföy execution |
| 📊 **Evaluation** | **YENİ** — Son 50 sinyal, başarı oranı, PnL, drawdown dashboard |
| 💼 **Portföy Sim** | **YENİ** — Stateful portföy: pozisyon aç/kapat, P&L, trade geçmişi |
| 📖 **Rehber** | Kullanım kılavuzu |

## Yeni: AI Trade Decision Pipeline

```
Ticker listesi gir (AAPL, NVDA, MSFT)
              ↓
    ┌─────────────────────────────┐
    │  Her ticker için:           │
    │  1. Signal Engine → sinyal  │
    │  2. Backtest → doğrulama    │
    │  3. Explanation → neden     │
    │  4. Portfolio → execute     │
    └─────────────────────────────┘
              ↓
    Sonuç: Hangi ticker'a girdi, neden, backtest doğruluğu,
           portföy P&L, equity curve, drawdown
```

## Yeni: Evaluation Dashboard

- **Son 50 sinyal** tablosu (tarih, ticker, sinyal, giriş/çıkış, PnL, doğruluk)
- **Birleşik equity curve** (tüm ticker'lar)
- **Drawdown chart**
- **Sinyal dağılımı** (BUY vs SELL doğruluk)
- **Ticker bazında başarı oranı**

## Yeni: Portfolio Simulation

- Stateful portföy — pozisyonlar session boyunca korunur
- Her sinyal otomatik execute edilir
- Açık pozisyonlar, kapalı trade'ler, P&L takibi
- Win rate, profit factor, max drawdown metrikleri
- Portföy sıfırlama

## Signal Engine

```
RSI(14) + SMA(20/50) crossover + MACD(12,26,9) + Bollinger(20,2) + Volume
                        ↓
              5 indikatör oylama
                        ↓
         3+/5 bullish = BUY, 4+/5 = STRONG_BUY
         3+/5 bearish = SELL, 4+/5 = STRONG_SELL
         Extreme volatility → BUY filtrelenir
```

## Tech Stack

Python + Gradio + yfinance + Binance API + NumPy + Pandas + Plotly

⚠️ **DISCLAIMER:** Eğitim ve araştırma amaçlıdır. Yatırım tavsiyesi değildir.
