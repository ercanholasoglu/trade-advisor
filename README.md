---
title: 🤖 Agentic Trade Bot Advisor v2.0
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.29.1"
app_file: app.py
pinned: true
license: mit
tags:
  - agent
  - trading
  - finance
  - multi-agent
  - smolagents
  - bist
  - stocks
  - crypto
---

# 🤖 Agentic Trade Bot Advisor v2.0

> **12 AI Agent** • **LLM Council (Bull vs Bear Debate)** • BIST + NASDAQ + Kripto + Altın + KAP + SEC + Makro

## Mimari

```
                              ┌─────────────────┐
                              │   Kullanıcı      │
                              │   "NVDA analiz"  │
                              └────────┬─────────┘
                                       │
                              ┌────────▼─────────┐
                              │  🎯 Fund Manager  │ ← CodeAgent (deep model)
                              │  (Orchestrator)    │
                              └────────┬─────────┘
                    ┌──────────────────┼──────────────────┐
         ┌──────────┤   Phase 1: Data   ├──────────┐       │
         │          └──────────────────┘          │       │
    Fast Model                                Fast Model  │
         │                                        │       │
    ┌────▼────┐ ┌────▼────┐ ┌────▼────┐ ┌────▼────┐     │
    │📈Technical│ │📰Sentiment│ │📊Fundament│ │🇹🇷 BIST  │     │
    │  Analyst │ │ Analyst │ │  Analyst │ │ Analyst │     │
    └─────────┘ └─────────┘ └─────────┘ └─────────┘     │
    ┌────▼────┐ ┌────▼────┐                              │
    │⚠️ Risk   │ │🌍 Macro  │                              │
    │ Manager │ │ Analyst │                              │
    └─────────┘ └─────────┘                              │
                    ┌──────────────────┼──────────────────┐
         ┌──────────┤  Phase 2: Council ├──────────┐       │
         │          └──────────────────┘          │       │
    Deep Model                                Deep Model  │
         │                                        │       │
    ┌────▼────┐ ┌────▼────┐ ┌────▼────┐          │       │
    │🐂 Bullish│ │🐻 Bearish│ │⚖️ Neutral│          │       │
    │ Advisor │ │ Advisor │ │Mediator │          │       │
    └─────────┘ └─────────┘ └─────────┘          │       │
                    ┌──────────────────┼──────────────────┐
                    │  Phase 3: Final   │
                    │  Decision + JSON   │
                    └──────────────────┘
```

## v2.0 Yeni Özellikler

| Özellik | Açıklama |
|---|---|
| 🧠 **FinBERT Sentiment** | HF Inference API ile NLP sentiment (keyword fallback) |
| 🏛️ **LLM Council** | Bull + Bear tartışması + Mediator hakem |
| 📊 **Backtest Engine** | RSI+SMA sinyallerini geçmiş veride test |
| 📈 **Portfolio Optimizer** | Markowitz efficient frontier — max Sharpe & min vol |
| 👁️ **Watchlist + Alerts** | Otomatik fiyat/RSI/VIX alertleri (Telegram/Email/Discord) |
| 📜 **Analysis History** | SQLite'da kayıt, 7/30 gün sonra doğruluk kontrolü |
| ₿ **Crypto Fundamentals** | CoinGecko — market cap, dev activity, community |
| 🕵️ **SEC Insider Trades** | EDGAR Form 4 — içeriden işlemler |
| 📊 **Options Flow** | Put/Call ratio, unusual volume tespiti |
| 🌍 **Macro Analyst** | TCMB, Fed, yield curve, DXY |
| 🗄️ **TTL Cache** | API çağrıları cache'lenir (15dk-24saat) |
| ⚡ **Dual Model** | Data: fast model, Council: deep model |
| 📋 **Structured JSON** | Pydantic TradeReport schema |
| 🛡️ **Error Handling** | Graceful degradation — kısmi veriyle devam |
| ⏰ **APScheduler** | 15dk watchlist tarama, günlük fiyat güncelleme |

## 10 Tab UI

1. 📊 **Dashboard** — Günlük AL/SAT taraması
2. ⚡ **Hızlı Analiz** — Tek hisse LLM Council analizi
3. 🔄 **Karşılaştırma** — 2-5 ticker karşılaştırma
4. 📊 **Backtest** — Sinyal doğruluk testi
5. 📈 **Portföy** — Markowitz optimizasyonu
6. 👁️ **Watchlist** — Alert yönetimi
7. 📜 **Geçmiş** — Analiz geçmişi + doğruluk
8. 🔔 **Telegram** — Alert gönderimi
9. 💬 **Sohbet** — Doğal dil analiz
10. 📖 **Rehber** — Kullanım kılavuzu

## 14 Tool

| Tool | Veri Kaynağı |
|---|---|
| `get_price_history` | Yahoo Finance |
| `get_technical_indicators` | pandas-ta |
| `get_news_sentiment` | DuckDuckGo + FinBERT |
| `get_fundamental_data` | Yahoo Finance |
| `calculate_risk_metrics` | VaR, ATR, position sizing |
| `get_market_overview` | 4 endeks + VIX + 11 sektör |
| `get_bist_scanner` | BIST30 + bankalar + emtia TRY |
| `get_kap_disclosures` | KAP API + DuckDuckGo |
| `get_daily_dashboard` | BIST + US günlük tarama |
| `get_crypto_fundamentals` | CoinGecko |
| `get_insider_trades` | SEC EDGAR Form 4 |
| `get_options_flow` | yfinance options |
| `get_macro_data` | TCMB + yfinance yields |

## Kurulum

```bash
git clone https://huggingface.co/spaces/SutskeverFanBoy/trade-bot-advisor
cd trade-bot-advisor
pip install -r requirements.txt
export HF_TOKEN="hf_xxx"
python app.py
```

## Ortam Değişkenleri

| Değişken | Açıklama |
|---|---|
| `HF_TOKEN` | HuggingFace API token (**zorunlu**) |
| `FAST_MODEL_ID` | Data agent'lar için hızlı model (opsiyonel) |
| `DEEP_MODEL_ID` | Council + FM için derin model (opsiyonel) |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token |
| `TELEGRAM_CHAT_ID` | Telegram chat ID |
| `DISCORD_WEBHOOK_URL` | Discord webhook |

## Akademik Referanslar

- [TradingAgents](https://arxiv.org/abs/2412.20138) — Bull/bear debate mimarisi
- [ContestTrade](https://arxiv.org/abs/2508.00554) — Rekabetçi agent sıralaması
- [FINSABER](https://arxiv.org/abs/2505.07078) — "Framework koordinasyonu > model büyüklüğü"

## Tech Stack

smolagents + Qwen/Qwen2.5-72B-Instruct + yfinance + pandas-ta + FinBERT + CoinGecko + SEC EDGAR + KAP + SQLite + APScheduler + Gradio

---

⚠️ **DISCLAIMER:** Bu sistem eğitim ve araştırma amaçlıdır. Yatırım tavsiyesi değildir.
