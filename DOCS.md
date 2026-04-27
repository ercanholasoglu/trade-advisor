# 🤖 Agentic Trade Bot Advisor — Ürün Dokümanı

> **Versiyon:** 1.0 | **Tarih:** Nisan 2026  
> **Space:** [huggingface.co/spaces/SutskeverFanBoy/trade-bot-advisor](https://huggingface.co/spaces/SutskeverFanBoy/trade-bot-advisor)  
> **Akademik Referans:** [TradingAgents Paper (arxiv: 2412.20138)](https://arxiv.org/abs/2412.20138)

---

## İçindekiler

1. [Ürüne Genel Bakış](#1-ürüne-genel-bakış)
2. [Sistem Mimarisi](#2-sistem-mimarisi)
3. [Veri Kaynakları — Neyi Nereden Alıyoruz?](#3-veri-kaynakları--neyi-nereden-alıyoruz)
4. [Tool Kataloğu — Her Tool Ne Yapar?](#4-tool-kataloğu--her-tool-ne-yapar)
5. [Agent Kataloğu — Her Agent Ne Yapar?](#5-agent-kataloğu--her-agent-ne-yapar)
6. [Karar Mekanizması — Sinyal Nasıl Üretiliyor?](#6-karar-mekanizması--sinyal-nasıl-üretiliyor)
7. [Alert ve Uyarı Sistemi](#7-alert-ve-uyarı-sistemi)
8. [Kullanım Kılavuzu](#8-kullanım-kılavuzu)
9. [Arayüz Geliştirme Yol Haritası](#9-arayüz-geliştirme-yol-haritası)
10. [Bilinen Kısıtlamalar](#10-bilinen-kısıtlamalar)
11. [Geliştirme Fikirleri](#11-geliştirme-fikirleri)

---

## 1. Ürüne Genel Bakış

### Ne?
5 otonom AI agent'ın bir fon yöneticisi gibi koordineli çalışarak herhangi bir hisse, kripto veya ETF için **kapsamlı trade analizi** ürettiği bir sistem.

### Neden Multi-Agent?
Tek bir LLM'e "AAPL'yi analiz et" dersen hallüsinasyon yapar — gerçek veriye erişimi yok, sayıları uydurur. Bizim sistemde:
- Her agent **kendi tool'u** ile **gerçek veri** çeker (yfinance, DuckDuckGo)
- Veriyi **kendi uzmanlık alanında** yorumlar
- Fund Manager tüm raporları alıp **çapraz doğrulama** yaparak final kararı verir

### Akış (30–90 saniye)
```
Kullanıcı: "NVDA analiz et, $100K portföy, moderate risk"
    │
    ▼
🎯 Fund Manager: "Tamam, sırayla herkesi çağırıyorum"
    │
    ├─→ 📈 Technical Analyst
    │     ├─ get_price_history("NVDA") → yfinance API → OHLCV verisi
    │     ├─ get_technical_indicators("NVDA") → pandas-ta → RSI, MACD, BB...
    │     └─ Rapor: "HOLD — RSI 53.6 nötr, MACD bullish ama BB overbought"
    │
    ├─→ 📰 Sentiment Analyst
    │     ├─ get_news_sentiment("NVDA", "NVIDIA") → DuckDuckGo News → 8 haber
    │     └─ Rapor: "BULLISH — 5/8 pozitif, AI chip demand güçlü"
    │
    ├─→ 📊 Fundamental Analyst
    │     ├─ get_fundamental_data("NVDA") → yfinance info → P/E, marjlar...
    │     └─ Rapor: "FAIR — Forward P/E 35 yüksek ama %122 earnings growth"
    │
    ├─→ ⚠️ Risk Manager
    │     ├─ get_market_overview() → 4 endeks + VIX + 11 sektör + emtia
    │     ├─ calculate_risk_metrics("NVDA", 100000, "moderate") → VaR, sizing
    │     └─ Rapor: "APPROVE_WITH_CAUTION — Vol moderate, R/R 1:1.5"
    │
    ▼
🎯 Fund Manager: TÜM raporları sentezler
    │
    ▼
📋 TRADE ADVISOR REPORT
   Signal: BUY | Confidence: 68%
   Entry: $135.40 | SL: $124.12 | TP: $152.32
   Position: 176 shares ($23,830)
```

---

## 2. Sistem Mimarisi

### Teknoloji Stack

| Katman | Teknoloji | Rol |
|---|---|---|
| **LLM** | Qwen/Qwen2.5-72B-Instruct | Tüm agent'ların beyni (HF Inference API üzerinden) |
| **Agent Framework** | smolagents 1.24+ | Agent orkestrasyon, tool yönetimi, kod çalıştırma |
| **Market Data** | yfinance 1.3+ | Fiyat, temel veriler, finansal tablolar (Yahoo Finance API) |
| **Teknik Analiz** | pandas-ta 0.4+ | 60+ teknik indikatör hesaplama |
| **Haber** | duckduckgo-search 6+ | Finansal haber arama |
| **UI** | Gradio 6.13+ | Web arayüzü |

### Agent Tipleri

| Agent | smolagents Tipi | Neden Bu Tip? |
|---|---|---|
| Technical Analyst | `ToolCallingAgent` | Yapılandırılmış API çağrıları — tool'u çağır, sonucu yorumla |
| Sentiment Analyst | `ToolCallingAgent` | Aynı — tek tool, düz veri akışı |
| Fundamental Analyst | `ToolCallingAgent` | Aynı — tek tool, düz veri akışı |
| Risk Manager | `ToolCallingAgent` | 2 tool'u sıralı çağırıp birleştirmesi gerekiyor |
| **Fund Manager** | `CodeAgent` | **Python kodu yazarak** 4 sub-agent'ı dinamik çağırır, sonuçları programatik birleştirir |

**Neden Fund Manager `CodeAgent`?** Çünkü 4 agent'ın sonuçlarını karşılaştırıp, çelişkileri çözüp, ağırlıklı karar vermesi gerekiyor. Bu tür karmaşık mantık, `ToolCallingAgent`'ın JSON-tabanlı yapısına sığmaz — Python kodu ile çok daha güçlü.

### Dosya Yapısı
```
trade-bot-advisor/
├── app.py                          # Gradio UI (2 tab)
├── agent.py                        # 5 agent tanımı + system prompt'lar
├── requirements.txt                # Bağımlılıklar
├── DOCS.md                         # ← Bu doküman
├── README.md                       # Space açıklama kartı
└── tools/
    ├── __init__.py
    ├── price_history.py            # Tool 1: OHLCV fiyat verileri
    ├── technical_indicators.py     # Tool 2: RSI, MACD, BB, SMA, ATR, Stoch
    ├── news_sentiment.py           # Tool 3: Haber arama + sentiment scoring
    ├── fundamental_analysis.py     # Tool 4: P/E, PEG, marjlar, bilanço
    ├── risk_calculator.py          # Tool 5: VaR, position sizing, SL/TP
    └── market_overview.py          # Tool 6: Endeksler, VIX, sektörler
```

---

## 3. Veri Kaynakları — Neyi Nereden Alıyoruz?

### 3.1 Fiyat Verisi (yfinance → Yahoo Finance)

| Veri | Kaynak | Güncelleme | Kapsam |
|---|---|---|---|
| OHLCV (Open/High/Low/Close/Volume) | Yahoo Finance API | Gerçek zamanlı (~15dk gecikme) | Tüm US hisseleri, ETF'ler, kripto, emtia, endeksler |
| Geçmiş fiyat | Yahoo Finance API | Günlük kapanış | 1 dakikadan max 30 yıla kadar |

**Nasıl çalışıyor:**
```python
stock = yf.Ticker("AAPL")
hist = stock.history(period="3mo", interval="1d")
# → DataFrame: Date, Open, High, Low, Close, Volume
```

**Desteklenen ticker formatları:**
- Hisseler: `AAPL`, `NVDA`, `TSLA`, `MSFT`
- Kripto: `BTC-USD`, `ETH-USD`, `SOL-USD`
- ETF: `SPY`, `QQQ`, `IWM`
- Emtia: `GC=F` (Altın), `CL=F` (Petrol)
- Endeksler: `^GSPC` (S&P 500), `^VIX`

**Kısıtlamalar:**
- 15 dakika gecikme (gerçek zamanlı değil)
- Aşırı istek → rate limit (günde ~2000 istek)
- Bazı uluslararası borsalar sınırlı

---

### 3.2 Temel Veriler / Finansallar (yfinance → Yahoo Finance)

| Veri | Detay |
|---|---|
| **Şirket Profili** | İsim, sektör, endüstri, ülke, çalışan sayısı, açıklama |
| **Değerleme Oranları** | P/E (trailing + forward), PEG, P/B, P/S, EV/EBITDA, EV/Revenue |
| **Gelir Tablosu** | Revenue, revenue growth, gross/operating/profit margin |
| **Bilanço** | Debt-to-equity, current ratio, free cash flow |
| **Performans** | ROE, ROA, earnings growth |
| **Temettü** | Yield, rate, payout ratio |
| **Analist Hedefleri** | Target mean/high/low, recommendation (buy/hold/sell), analist sayısı |

**Nasıl çalışıyor:**
```python
stock = yf.Ticker("AAPL")
info = stock.info
# → dict: marketCap, trailingPE, forwardPE, profitMargins, returnOnEquity, ...
```

**Kısıtlama:** Kripto ticker'larında temel veri yok (sadece hisseler).

---

### 3.3 Haber Verisi (DuckDuckGo News Search)

| Veri | Detay |
|---|---|
| **Kaynak** | DuckDuckGo News API (ücretsiz, API key gerektirmez) |
| **Arama sorgusu** | `"{ticker} {company_name} stock news financial"` |
| **Sonuç sayısı** | Sorgu başına 8 haber |
| **İçerik** | Başlık, kaynak, tarih, URL, özet (ilk 200 karakter) |

**Sentiment nasıl hesaplanıyor?**

Şu an **keyword-based** basit bir yöntem kullanıyoruz (FinBERT yerine lightweight):

```
Pozitif kelimeler (27 adet):
  surge, soar, jump, gain, rally, rise, bull, bullish, record, high,
  growth, profit, beat, exceed, upgrade, buy, outperform, strong,
  positive, boom, breakout, optimistic, upbeat, recovery, innovation, expand

Negatif kelimeler (27 adet):
  drop, fall, crash, plunge, decline, loss, bear, bearish, low, weak,
  miss, downgrade, sell, underperform, risk, fear, concern, warning,
  cut, layoff, recession, debt, slump, volatile, uncertainty, lawsuit, investigation
```

**Scoring:**
- Her haberde pozitif ve negatif kelime sayısı karşılaştırılır
- `score = pozitif_sayısı × 0.2` (max +1.0) veya `score = -negatif_sayısı × 0.2` (min -1.0)
- Tüm haberlerin ortalaması → overall sentiment
- `avg > 0.2` → BULLISH, `avg < -0.2` → BEARISH, arada → NEUTRAL

---

### 3.4 Piyasa Genel Durumu (yfinance → çoklu sembol)

Tek seferde 20+ sembol çekiliyor:

| Kategori | Semboller |
|---|---|
| **Endeksler** | S&P 500 (`^GSPC`), NASDAQ (`^IXIC`), DOW (`^DJI`), Russell 2000 (`^RUT`) |
| **Korku Endeksi** | VIX (`^VIX`) |
| **Emtia** | Altın (`GC=F`), Petrol (`CL=F`) |
| **Kripto** | Bitcoin (`BTC-USD`) |
| **Tahvil** | 10Y Treasury (`^TNX`) |
| **Dolar** | DXY (`DX-Y.NYB`) |
| **Sektör ETF'leri (11)** | XLK, XLV, XLF, XLE, XLY, XLI, XLU, XLRE, XLC, XLB, XLP |

Her sembol için son 5 günlük veri çekilir, önceki güne göre % değişim hesaplanır.

---

## 4. Tool Kataloğu — Her Tool Ne Yapar?

### Tool 1: `get_price_history`
📁 `tools/price_history.py`

| Özellik | Detay |
|---|---|
| **Girdi** | `ticker` (str), `period` (str, default "1mo"), `interval` (str, default "1d") |
| **Çıktı** | JSON: OHLCV kayıtları + temel istatistikler |
| **Veri kaynağı** | Yahoo Finance (yfinance) |
| **Kullanan agent** | Technical Analyst |

**Ne döndürür:**
```json
{
  "ticker": "AAPL",
  "current_price": 266.41,
  "price_change": 0.09,
  "price_change_pct": 0.09,
  "period_high": 270.50,
  "period_low": 245.22,
  "avg_volume": 48523100,
  "data_points": 22,
  "ohlcv": [
    {"date": "2026-03-27 00:00", "open": 265.10, "high": 267.80, "low": 264.50, "close": 266.41, "volume": 52340000},
    ...
  ]
}
```

**Ne zaman kullanılır:** Agent fiyat trendini, destek/direnç seviyelerini ve hacim profilini anlamak istediğinde.

---

### Tool 2: `get_technical_indicators`
📁 `tools/technical_indicators.py`

| Özellik | Detay |
|---|---|
| **Girdi** | `ticker` (str), `period` (str, default "3mo") |
| **Çıktı** | JSON: 7 indikatör + sinyal + genel değerlendirme |
| **Veri kaynağı** | Yahoo Finance → pandas-ta hesaplama |
| **Kullanan agent** | Technical Analyst |

**Hesaplanan İndikatörler:**

| İndikatör | Parametreler | Sinyal Mantığı |
|---|---|---|
| **RSI** | Period: 14 | >70 OVERBOUGHT, <30 OVERSOLD, >60 BULLISH, <40 BEARISH, arası NEUTRAL |
| **MACD** | Fast:12, Slow:26, Signal:9 | Histogram >0 BULLISH, <0 BEARISH. Ayrıca crossover tespiti |
| **Bollinger Bands** | Period:20, StdDev:2 | Fiyat > üst bant: OVERBOUGHT, < alt bant: OVERSOLD |
| **SMA** | 20-gün ve 50-gün | SMA20 > SMA50 + fiyat > SMA20: STRONG_BULLISH |
| **EMA** | 12-gün ve 26-gün | Trend takibi için |
| **ATR** | Period: 14 | Volatilite ölçümü ($ ve % olarak) |
| **Stochastic** | K:14, D:3, Smooth:3 | >80 OVERBOUGHT, <20 OVERSOLD |

**Genel sinyal nasıl hesaplanıyor:**
```
5 indikatörden kaç tanesi bullish/bearish?
  ≥4 bullish → STRONG_BUY
  ≥3 bullish → BUY
  ≥4 bearish → STRONG_SELL
  ≥3 bearish → SELL
  aksi halde  → HOLD
```

---

### Tool 3: `get_news_sentiment`
📁 `tools/news_sentiment.py`

| Özellik | Detay |
|---|---|
| **Girdi** | `ticker` (str), `company_name` (str, opsiyonel) |
| **Çıktı** | JSON: 8 haber + bireysel sentiment + genel özet |
| **Veri kaynağı** | DuckDuckGo News Search |
| **Kullanan agent** | Sentiment Analyst |

**Çıktı örneği:**
```json
{
  "ticker": "NVDA",
  "news_count": 8,
  "overall_sentiment": "BULLISH",
  "avg_sentiment_score": 0.225,
  "sentiment_breakdown": {"positive": 5, "negative": 2, "neutral": 1},
  "articles": [
    {
      "title": "NVIDIA Shares Surge on Record AI Chip Demand",
      "source": "Reuters",
      "sentiment": "POSITIVE",
      "sentiment_score": 0.6,
      "snippet": "NVIDIA Corp reported record quarterly revenue..."
    }
  ]
}
```

---

### Tool 4: `get_fundamental_data`
📁 `tools/fundamental_analysis.py`

| Özellik | Detay |
|---|---|
| **Girdi** | `ticker` (str) |
| **Çıktı** | JSON: Profil + değerleme + finansallar + temettü + analist hedefleri |
| **Veri kaynağı** | Yahoo Finance (stock.info) |
| **Kullanan agent** | Fundamental Analyst |

**Döndürdüğü veri kategorileri:**
- **Profil:** İsim, sektör, market cap (okunabilir: "$3.91T"), çalışan sayısı
- **Değerleme:** P/E trailing & forward, PEG, P/B, P/S, EV/EBITDA
- **Finansallar:** Revenue (formatted), revenue growth %, gross/operating/profit margin %, ROE, ROA, D/E, free cash flow
- **Temettü:** Yield %, rate, payout ratio
- **Analist:** Target fiyat (mean/high/low), recommendation, upside %
- **Otomatik sinyaller:** `LOW_PE_UNDERVALUED` (P/E<15), `HIGH_PE_OVERVALUED` (P/E>30), `PEG_UNDERVALUED` (<1), `ANALYSTS_BULLISH/BEARISH`

---

### Tool 5: `calculate_risk_metrics`
📁 `tools/risk_calculator.py`

| Özellik | Detay |
|---|---|
| **Girdi** | `ticker` (str), `portfolio_value` (float), `risk_tolerance` (str) |
| **Çıktı** | JSON: Volatilite + VaR + position sizing + SL/TP + öneriler |
| **Veri kaynağı** | Yahoo Finance (6 aylık günlük veri) |
| **Kullanan agent** | Risk Manager |

**Hesaplamalar:**

| Metrik | Formül | Açıklama |
|---|---|---|
| **Günlük Volatilite** | `std(daily_returns)` | Fiyat oynaklığı |
| **Yıllık Volatilite** | `daily_vol × √252` | Yıllıklandırılmış |
| **VaR (Value at Risk)** | `vol × z_score` | Conservative: %99 (z=2.326), Moderate: %95 (z=1.645), Aggressive: %90 (z=1.282) |
| **Max Drawdown** | `min((price - peak) / peak)` | 6 ayda en büyük düşüş |
| **Sharpe Ratio** | `(avg_return / vol) × √252` | Risk-ayarlı getiri |
| **ATR (14)** | True Range ortalaması | Fiyat dalgalanma mesafesi |

**Position Sizing (Risk Toleransına Göre):**

| Tolerans | Trade Başına Risk | SL Çarpanı | TP Çarpanı | VaR Confidence |
|---|---|---|---|---|
| Conservative | Portföyün %1'i | 1.5 × ATR | 2.0 × ATR | %99 |
| Moderate | Portföyün %2'si | 2.0 × ATR | 3.0 × ATR | %95 |
| Aggressive | Portföyün %4'ü | 3.0 × ATR | 4.5 × ATR | %90 |

**Hisse sayısı:** `risk_amount / stop_loss_distance`

**Otomatik Uyarılar:**
- ⚠️ EXTREME volatilite → pozisyonu %50 küçült
- ⚠️ 6 ayda %20+ drawdown → dikkatli ol
- ⚠️ R/R < 1.5 → TP'yi genişlet
- ⚠️ Pozisyon > portföyün %20'si → küçült
- ✅ Sharpe > 1 → iyi risk-ayarlı getiri
- ✅ Düşük volatilite → giriş için uygun

---

### Tool 6: `get_market_overview`
📁 `tools/market_overview.py`

| Özellik | Detay |
|---|---|
| **Girdi** | Yok (argümansız) |
| **Çıktı** | JSON: Endeksler + VIX + emtia/kripto + 11 sektör + piyasa rejimi |
| **Veri kaynağı** | Yahoo Finance (20+ sembol) |
| **Kullanan agent** | Risk Manager |

**Piyasa rejimi nasıl belirleniyor:**
```
4 endeksten (S&P 500, NASDAQ, DOW, Russell 2000) kaçı pozitif?
  4/4 pozitif → BULLISH
  3/4 pozitif → MODERATELY_BULLISH
  1/4 pozitif → MODERATELY_BEARISH
  0/4 pozitif → BEARISH
  
VIX > 25 ise → "+ HIGH_VOLATILITY" eklenir
```

**VIX Rejim Sınıflandırması:**
```
VIX < 15  → LOW_FEAR (Complacency) — piyasa aşırı rahat, dikkat
VIX 15-20 → NORMAL — standart piyasa koşulları
VIX 20-30 → ELEVATED_FEAR — artan belirsizlik
VIX > 30  → HIGH_FEAR (Panic) — panik modu, trade tavsiye edilmez
```

---

## 5. Agent Kataloğu — Her Agent Ne Yapar?

### 📈 Agent 1: Technical Analyst
```
Tip:         ToolCallingAgent
Tool'ları:   get_price_history, get_technical_indicators
Max Adım:    6
```

**İş akışı:**
1. `get_price_history(ticker)` çağır → son 1 ayın OHLCV verisini al
2. `get_technical_indicators(ticker)` çağır → 7 indikatörü hesaplat
3. İkisini birleştirip teknik rapor yaz

**Raporunda olması gereken:**
- Trend yönü: UPTREND / DOWNTREND / SIDEWAYS
- Destek ve direnç seviyeleri (fiyattan çıkarır)
- Momentum değerlendirmesi (RSI + MACD)
- Volatilite değerlendirmesi (BB width + ATR)
- Genel teknik sinyal: STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL

---

### 📰 Agent 2: Sentiment Analyst
```
Tip:         ToolCallingAgent
Tool'ları:   get_news_sentiment
Max Adım:    4
```

**İş akışı:**
1. `get_news_sentiment(ticker, company_name)` çağır → 8 haber + sentiment
2. Haberleri piyasa etkisine göre değerlendir
3. Sentiment raporu yaz

**Raporunda olması gereken:**
- Genel sentiment: BULLISH / NEUTRAL / BEARISH
- Ana pozitif katalizörler (earnings beat, yeni ürün, upgrade, vs.)
- Ana riskler (yasal sorun, downgrade, yavaşlama, vs.)
- Yaklaşan önemli olaylar (earnings tarihi, FDA kararı, vs.)
- Güven seviyesi: HIGH / MEDIUM / LOW

---

### 📊 Agent 3: Fundamental Analyst
```
Tip:         ToolCallingAgent
Tool'ları:   get_fundamental_data
Max Adım:    4
```

**İş akışı:**
1. `get_fundamental_data(ticker)` çağır → tüm finansal veriler
2. Değerleme, büyüme, karlılık, bilanço sağlığını değerlendir
3. Temel analiz raporu yaz

**Raporunda olması gereken:**
- Değerleme: UNDERVALUED / FAIR / OVERVALUED (P/E, PEG, P/S ile)
- Büyüme: Revenue ve earnings growth trajectory
- Karlılık: Margin kalitesi, ROE, ROA
- Finansal sağlık: Borç seviyeleri, nakit akışı
- Analist konsensüsü: Target fiyat ve upside/downside %

---

### ⚠️ Agent 4: Risk Manager
```
Tip:         ToolCallingAgent
Tool'ları:   calculate_risk_metrics, get_market_overview
Max Adım:    6
```

**İş akışı:**
1. `get_market_overview()` çağır → genel piyasa durumu, VIX, sektörler
2. `calculate_risk_metrics(ticker, portfolio, tolerance)` çağır → pozisyon boyutlama
3. Risk raporu yaz

**Raporunda olması gereken:**
- Piyasa ortamı: Bull/Bear/Neutral + VIX rejimi
- Volatilite değerlendirmesi: Hissenin vol rejimi
- Pozisyon boyutlama: Kaç hisse, kaç dolar
- Giriş/çıkış seviyeleri: Stop-loss ve take-profit fiyatları
- Risk/Reward oranı: Trade'e değer mi?
- Risk verdikti: APPROVE / APPROVE_WITH_CAUTION / REJECT

---

### 🎯 Agent 5: Fund Manager (Orchestrator)
```
Tip:         CodeAgent
Tool'ları:   Yok (4 managed agent'ı kullanır)
Max Adım:    15
Planning:    Her 3 adımda bir yeniden planlar
```

**İş akışı:**
1. Technical Analyst'ı çağır → teknik rapor al
2. Sentiment Analyst'ı çağır → sentiment rapor al
3. Fundamental Analyst'ı çağır → temel analiz rapor al
4. Risk Manager'ı çağır → risk rapor al
5. **4 raporu sentezle** → final trade kararı üret

**Karar kuralları:**
- Tüm 4 agent'a MUTLAKA danışır (atlamaz)
- Sinyaller çelişirse → Risk Manager'ın verdikti en ağır
- VIX > 30 → ne olursa olsun HOLD önerir
- Confidence < %50 → HOLD önerir
- Her trade'de SL ve TP ZORUNLU
- Disclaimer her zaman eklenir

---

## 6. Karar Mekanizması — Sinyal Nasıl Üretiliyor?

### Sinyal Oluşturma Akışı

```
                    ┌─────────────────┐
                    │  4 Agent Raporu  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Fund Manager  │
                    │   (CodeAgent)   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        Sinyal Uyumu?   VIX Kontrolü   Confidence
              │              │          Hesaplama
              │              │              │
              ▼              ▼              ▼
         ┌─────────────────────────────────────┐
         │          FINAL KARAR                 │
         │                                      │
         │  BUY:  ≥3 agent bullish + risk OK    │
         │  SELL: ≥3 agent bearish              │
         │  HOLD: Karışık sinyaller / VIX>30    │
         │        / confidence < %50            │
         └─────────────────────────────────────┘
```

### Ağırlıklandırma (Fund Manager'ın implicit prioritesi)

Fund Manager bir LLM olduğu için katı bir "puan sistemi" yerine doğal dil mantığı kullanır. Ancak system prompt'taki kurallar şöyle bir hiyerarşi oluşturur:

```
1. 🚨 Risk Manager REJECT dedi → HOLD (en yüksek öncelik)
2. 🚨 VIX > 30 → HOLD (piyasa panikte)
3. ⚠️  Çelişkili sinyaller → Risk Manager'a ağırlık ver
4. ✅  3+ agent aynı yönde → o yönde sinyal ver
5. 📊  Confidence < %50 → HOLD
```

---

## 7. Alert ve Uyarı Sistemi

### Mevcut Uyarılar (Risk Calculator'dan otomatik)

| Uyarı | Koşul | Öneri |
|---|---|---|
| ⚠️ EXTREME Volatilite | Yıllık vol > %50 | Pozisyon %50 küçült |
| ⚠️ Büyük Drawdown | 6 ayda max drawdown > %20 | Dikkatli ol |
| ⚠️ Düşük R/R | Risk/Reward < 1.5 | Take-profit'i genişlet |
| ⚠️ Aşırı Pozisyon | Pozisyon > portföyün %20'si | Küçült |
| ✅ İyi Sharpe | Sharpe > 1 | Olumlu sinyal |
| ✅ Düşük Vol | Vol rejimi LOW | Giriş için uygun |

### VIX-Bazlı Market Alert'leri

| VIX Seviyesi | Rejim | Sistem Davranışı |
|---|---|---|
| < 15 | LOW_FEAR (Complacency) | Normal analiz, ama "piyasa aşırı rahat" uyarısı |
| 15-20 | NORMAL | Normal analiz |
| 20-30 | ELEVATED_FEAR | "Artan belirsizlik" uyarısı, pozisyon küçültme önerisi |
| > 30 | HIGH_FEAR (Panic) | **HOLD zorunlu** — VIX>30 kuralı devreye girer |

### Fund Manager Seviyesinde Uyarılar

Fund Manager şu durumlarda kendi başına HOLD önerir:
- 4 agent'tan çelişkili sinyaller geliyorsa
- Confidence skoru %50'nin altındaysa
- Risk Manager REJECT verdiyse
- Piyasa aşırı koşullardaysa

---

## 8. Kullanım Kılavuzu

### Tab 1: ⚡ Hızlı Analiz

**En basit kullanım.** 3 input gir, butona bas:

1. **Ticker Sembolü:** Analiz etmek istediğin varlık
   - `AAPL` → Apple
   - `BTC-USD` → Bitcoin
   - `GC=F` → Altın
   
2. **Portföy Değeri ($):** Toplam yatırım bütçen
   - Minimum: $1,000
   - Bu değer position sizing için kullanılır
   
3. **Risk Toleransı:** 3 seviye
   - **Conservative:** Portföyün %1'ini riskle, sıkı stop-loss
   - **Moderate:** Portföyün %2'sini riskle, dengeli
   - **Aggressive:** Portföyün %4'ünü riskle, geniş stop-loss

4. **"🔍 Analiz Başlat"** butonuna tıkla → 30-90 saniye bekle

### Tab 2: 💬 Serbest Sohbet

**Doğal dilde soru sor.** Agent'lar otomatik olarak gerekli tool'ları çağırır.

**Örnek sorular:**
```
"AAPL hissesini analiz et, portföyüm $50K ve orta risk toleransım var"
"Piyasa genel durumu nasıl? Şu an hisse almak mantıklı mı?"  
"NVDA ve AMD'yi karşılaştır, hangisi daha iyi?"
"BTC-USD için kısa vadeli teknik analiz yap"
"TSLA aşırı değerli mi? Temel veriler ne diyor?"
```

### Raporu Okuma Rehberi

```
═══════════════════════════════════════════════════
🤖 TRADE ADVISOR REPORT
═══════════════════════════════════════════════════

📊 TICKER: NVDA | COMPANY: NVIDIA Corporation     ← Ne analiz edildi
📅 DATE: 2026-04-27                                ← Analiz tarihi

📈 SIGNAL: BUY                ← ANA KARAR: BUY / SELL / HOLD
🎯 CONFIDENCE: 68%            ← Ne kadar emin (%50 altı → HOLD)

📋 TECHNICAL: BULLISH         ← Teknik indikatörler ne diyor
📰 SENTIMENT: BULLISH         ← Haberler ne diyor
📊 FUNDAMENTAL: FAIR          ← Değerleme makul mü
⚠️ RISK: APPROVE_WITH_CAUTION ← Risk yönetimi onaylıyor mu

💰 TRADE PARAMETERS:          ← İşlem parametreleri
• Entry: $135.40              ← Bu fiyattan gir
• Stop-Loss: $124.12 (-8.3%)  ← Zarar kes seviyesi
• Take-Profit: $152.32 (+12%) ← Kar al seviyesi  
• Position: 176 shares ($23K)  ← Kaç hisse, kaç $
• Risk/Reward: 1:1.5          ← Her 1$ risk için 1.5$ potansiyel kar
═══════════════════════════════════════════════════
```

---

## 9. Arayüz Geliştirme Yol Haritası

### 🔴 Faz 1 — Kısa Vade (Acil İyileştirmeler)

#### 1.1 Gerçek Zamanlı İlerleme Göstergesi
**Problem:** Kullanıcı butona basıyor, 30-90 saniye boyunca boş ekrana bakıyor.
**Çözüm:** Her agent çalışırken canlı progress göster:
```
⏳ [1/4] Technical Analyst çalışıyor... RSI, MACD hesaplanıyor
✅ [2/4] Sentiment Analyst tamamlandı — BULLISH
⏳ [3/4] Fundamental Analyst çalışıyor...
✅ [4/4] Risk Manager tamamlandı — APPROVE
🎯 Fund Manager final kararı hazırlıyor...
```
**Nasıl:** smolagents `step_callbacks` kullanarak her agent step'inde Gradio `yield` ile güncelle.

#### 1.2 Sonuç Kartları (Görsel Output)
**Problem:** Çıktı düz metin — zor okunuyor.
**Çözüm:** Gradio `gr.HTML` ile renkli kartlar:
- 🟢 BUY → yeşil kart
- 🔴 SELL → kırmızı kart  
- 🟡 HOLD → sarı kart
- Gauge chart ile confidence göster
- Mini sparkline ile fiyat trendi

#### 1.3 Çoklu Ticker Karşılaştırma
**Problem:** Tek seferde 1 ticker analiz edebiliyoruz.
**Çözüm:** "Karşılaştır" tab'ı ekle — 2-3 ticker yan yana analiz:
```
NVDA vs AMD vs INTC
├── Teknik Skor:    BUY / HOLD / SELL
├── Sentiment:      BULLISH / NEUTRAL / BEARISH
├── Değerleme:      FAIR / OVERVALUED / UNDERVALUED
└── Risk/Reward:    1:1.5 / 1:1.2 / 1:2.0
→ Kazanan: NVDA (en iyi R/R + teknik momentum)
```

---

### 🟡 Faz 2 — Orta Vade (Yeni Özellikler)

#### 2.1 FinBERT Entegrasyonu (Sentiment Upgrade)
**Problem:** Keyword-based sentiment çok basit — "the stock did not fall" gibi negation'ları yakalayamıyor.
**Çözüm:** `ProsusAI/finbert` modeli ile her haber cümlesine gerçek NLP sentiment:
```python
from transformers import pipeline
sentiment_model = pipeline("text-classification", model="ProsusAI/finbert")
# → {"label": "positive", "score": 0.94}
```
**Etki:** Sentiment doğruluğu ~%60 → ~%85'e çıkar.

#### 2.2 Interaktif Grafik Paneli
**Problem:** Fiyat verisi sadece JSON — görselleştirme yok.
**Çözüm:** `plotly` ile interaktif grafik tab'ı:
- Candlestick chart (OHLCV)
- Bollinger Bands overlay
- RSI alt panel
- MACD alt panel
- Volume bar chart
- Destek/direnç çizgileri

#### 2.3 Watchlist & Favoriler
**Problem:** Her seferinde ticker yazıyorsun.
**Çözüm:**
- Favori ticker listesi (localStorage ile)
- Önceden tanımlı watchlist'ler: "FAANG", "Magnificent 7", "Kripto Top 5"
- Tek tıkla analiz

#### 2.4 Tarihsel Analiz Logu
**Problem:** Geçmiş analizleri göremiyorsun.
**Çözüm:**
- Her analiz otomatik loglanır
- "Geçmiş Analizler" tab'ı
- Sinyal doğruluk oranı takibi (geriye dönük)

---

### 🟢 Faz 3 — Uzun Vade (İleri Özellikler)

#### 3.1 Backtesting Motoru
**Problem:** Sinyallerin gerçekten çalışıp çalışmadığını bilmiyoruz.
**Çözüm:** Geçmiş veriler üzerinde simülasyon:
```
Backtest Sonuçları (AAPL, son 1 yıl):
├── Toplam Sinyal: 24 (14 BUY, 6 SELL, 4 HOLD)
├── Doğru Sinyal: 16/24 (%67)
├── Kümülatif Getiri: +18.4%
├── Buy & Hold: +22.1%
├── Max Drawdown: -8.2%
└── Sharpe: 1.34
```

#### 3.2 Alert/Bildirim Sistemi
**Problem:** Sürekli kontrol etmen gerekiyor.
**Çözüm:**
- Fiyat alertleri: "AAPL $250'nin altına düşerse bildir"
- Teknik alertler: "RSI 30'un altına düşerse bildir"
- VIX alertleri: "VIX 25'i geçerse bildir"
- Haber alertleri: "NVDA hakkında negatif haber çıkarsa bildir"
- Bildirim: Email / Telegram / Discord webhook

#### 3.3 Portföy Takibi
**Problem:** Tek hisse analizi var, portföy görünümü yok.
**Çözüm:**
- Portföy girişi (ticker + miktar)
- Toplam portföy değeri, günlük P/L
- Korelasyon matrisi (diversifikasyon kontrolü)
- Sektör dağılımı pie chart
- Portföy beta ve Sharpe

#### 3.4 Multi-LLM Routing
**TradingAgents paper'dan ilham:** Farklı agent'lara farklı modeller ata:
```
Veri çeken agent'lar → hızlı model (Qwen2.5-7B veya gpt-4o-mini)
Karar veren agent'lar → derin model (Qwen3-235B veya gpt-4o)
```
**Etki:** Maliyet %70 düşer, kalite aynı kalır.

#### 3.5 Bull vs Bear Debate Agent'ları
**TradingAgents paper'ın en güçlü özelliği:**
```
Mevcut: 4 analist → Fund Manager
Gelişmiş: 4 analist → Bullish Researcher ↔ Bearish Researcher (N tur tartışma) → Fund Manager
```
2 araştırmacı agent, analist raporlarını alıp birbirleriyle tartışır. Bu, LLM'in aşırı güven (overconfidence) sorununu azaltır.

---

## 10. Bilinen Kısıtlamalar

| Kısıt | Detay | Çözüm Yolu |
|---|---|---|
| **15dk fiyat gecikmesi** | yfinance gerçek zamanlı değil | Polygon.io veya Alpaca API'ye geçiş |
| **Basit sentiment** | Keyword-based, negation yakalayamaz | FinBERT entegrasyonu (Faz 2) |
| **Sadece İngilizce haber** | DuckDuckGo İngilizce sonuç döner | Çoklu dil desteği ekle |
| **Kripto fundamental yok** | yfinance crypto'da sadece fiyat verir | CoinGecko API entegrasyonu |
| **Rate limit** | yfinance günde ~2000 istek | Cache katmanı ekle |
| **Tek seferlik analiz** | Canlı takip yok | WebSocket + scheduled jobs |
| **LLM maliyeti** | Her analiz ~5-15 HF Inference API çağrısı | Model caching, daha küçük modeller |
| **Backtesting yok** | Sinyallerin doğruluğunu test edemiyoruz | Backtest motoru (Faz 3) |

---

## 11. Geliştirme Fikirleri

### Yeni Tool Fikirleri

| Tool | Veri Kaynağı | Ne Yapar |
|---|---|---|
| `get_insider_trades` | SEC EDGAR / OpenInsider | Insider alım/satımları — büyük içeriden alım → bullish sinyal |
| `get_options_flow` | Yahoo Finance options | Put/Call ratio, unusual options activity |
| `get_earnings_calendar` | Yahoo Finance | Yaklaşan earnings tarihleri ve tahminler |
| `get_social_sentiment` | Reddit API (wallstreetbets) | Sosyal medya hype/fear ölçümü |
| `get_crypto_onchain` | Glassnode / CoinGecko | Whale hareketleri, exchange inflow/outflow |
| `get_economic_calendar` | FRED / Trading Economics | Fed faiz kararı, CPI, NFP tarihleri |
| `get_sector_rotation` | yfinance sektör ETF'leri | Sektör rotasyonu tespiti (defensive↔cyclical) |

### Yeni Agent Fikirleri

| Agent | Rol |
|---|---|
| **Macro Economist** | Faiz, enflasyon, GDP verileriyle makro ortam değerlendirmesi |
| **Bullish Researcher** | Analist raporlarından sadece pozitif argümanları savunur |
| **Bearish Researcher** | Analist raporlarından sadece negatif argümanları savunur |
| **Portfolio Manager** | Mevcut portföyle yeni trade'in uyumunu kontrol eder |
| **Execution Strategist** | Limit order vs market order, en iyi giriş zamanlaması |

---

> ⚠️ **DISCLAIMER:** Bu sistem eğitim ve araştırma amaçlıdır. Yatırım tavsiyesi değildir. Finansal kararlarınızı her zaman kendi araştırmanıza ve profesyonel danışmanlarınıza dayandırın.
