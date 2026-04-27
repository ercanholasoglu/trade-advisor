---
title: "🤖 Agentic Trade Bot Advisor"
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: true
short_description: "Multi-agent AI trade advisor with 5 specialized agents"
---

# 🤖 Agentic Trade Bot Advisor

**TradingAgents** paper ([arxiv: 2412.20138](https://arxiv.org/abs/2412.20138)) mimarisinden ilham alan multi-agent trade advisor sistemi.

## 🏗️ Mimari

```
┌─────────────────────────────────────────────────────────┐
│                   🎯 FUND MANAGER                       │
│              (CodeAgent - Orchestrator)                  │
│         Tüm raporları alır, final karar verir           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │📈Technical│ │📰News    │ │📊Fundamen│ │⚠️Risk    │  │
│  │ Analyst  │ │ Analyst  │ │tal Analyst│ │ Manager  │  │
│  │          │ │          │ │          │ │          │  │
│  │• RSI     │ │• News    │ │• P/E     │ │• VaR     │  │
│  │• MACD    │ │• Sentimen│ │• Earnings│ │• Position│  │
│  │• Bollinge│ │• DuckDuck│ │• Balance │ │• Stop-Los│  │
│  │• SMA/EMA │ │  Go News │ │  Sheet   │ │• Market  │  │
│  │• ATR     │ │          │ │• Analyst │ │  Overview│  │
│  │• Stoch   │ │          │ │  Targets │ │          │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 🛠️ Agent'lar ve Tool'ları

| Agent | Tool'lar | Görev |
|---|---|---|
| 📈 **Technical Analyst** | `get_price_history`, `get_technical_indicators` | RSI, MACD, Bollinger Bands, SMA, EMA, ATR, Stochastic analizi |
| 📰 **Sentiment Analyst** | `get_news_sentiment` | DuckDuckGo ile haber arama + keyword sentiment scoring |
| 📊 **Fundamental Analyst** | `get_fundamental_data` | P/E, PEG, marjlar, büyüme, ROE, bilanço, analist hedefleri |
| ⚠️ **Risk Manager** | `calculate_risk_metrics`, `get_market_overview` | VaR, position sizing, stop-loss/take-profit, piyasa rejimi |
| 🎯 **Fund Manager** | *Managed agents* | Tüm raporları sentezler, final BUY/SELL/HOLD kararı verir |

## ⚡ Teknoloji Stack

- **Framework**: [smolagents](https://huggingface.co/docs/smolagents) (HuggingFace)
- **LLM**: [Qwen/Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) via HF Inference API
- **Market Data**: [yfinance](https://github.com/ranaroussi/yfinance)
- **Technical Analysis**: [pandas-ta](https://github.com/twopirllc/pandas-ta)
- **News**: [DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search)
- **UI**: [Gradio](https://gradio.app/)

## 📊 Örnek Output

```
═══════════════════════════════════════════════════
🤖 TRADE ADVISOR REPORT
═══════════════════════════════════════════════════
📊 TICKER: NVDA | COMPANY: NVIDIA Corporation
📈 SIGNAL: BUY | 🎯 CONFIDENCE: 72%
───────────────────────────────────────────────────
💰 TRADE PARAMETERS:
• Entry: $135.40 | Stop-Loss: $124.12 (-8.3%)
• Take-Profit: $152.32 (+12.5%) | R/R: 1:1.5
• Position: 176 shares ($23,830)
═══════════════════════════════════════════════════
```

## 📚 Referanslar

- [TradingAgents: Multi-Agents LLM Financial Trading Framework](https://arxiv.org/abs/2412.20138) (53K ⭐)
- [QuantAgent: HFT Multi-Agent System](https://arxiv.org/abs/2509.09995)
- [ContestTrade: Internal Contest Mechanism](https://arxiv.org/abs/2508.00554)
- [FINSABER: Evaluation Framework](https://arxiv.org/abs/2505.07078)

## ⚠️ Disclaimer

Bu yapay zeka destekli bir analiz aracıdır. **Yatırım tavsiyesi değildir.** 
Kendi araştırmanızı yapın ve finansal kararlarınızı profesyonel danışmanlarınızla görüşerek alın.
