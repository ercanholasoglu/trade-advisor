"""
🤖 Trade Bot Advisor - Gradio UI (Faz 1)
==========================================
Canlı ilerleme göstergesi, görsel sonuç kartları, çoklu ticker
karşılaştırma, Telegram alert entegrasyonu.
"""

import os
import json
import time
import datetime
import gradio as gr
from agent import create_trade_advisor

from tools.price_history import get_price_history
from tools.technical_indicators import get_technical_indicators
from tools.news_sentiment import get_news_sentiment
from tools.fundamental_analysis import get_fundamental_data
from tools.risk_calculator import calculate_risk_metrics
from tools.market_overview import get_market_overview


# ============================================================
# Global agent instance (lazy init)
# ============================================================
_advisor = None


def get_advisor():
    global _advisor
    if _advisor is None:
        hf_token = os.environ.get("HF_TOKEN", "")
        model_id = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-72B-Instruct")
        _advisor = create_trade_advisor(hf_token=hf_token, model_id=model_id)
    return _advisor


# ============================================================
# Yardımcı: Tool'ları doğrudan çağır (agent bypass — hız modu)
# ============================================================
def _safe_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        return {"error": raw}


def run_direct_analysis(ticker: str, portfolio_value: float, risk_tolerance: str):
    """Agent'ları atla, tool'ları doğrudan çağır — canlı progress için."""
    ticker = ticker.strip().upper()
    results = {}

    # 1) Price
    yield "⏳ **[1/6]** Fiyat verisi çekiliyor…\n"
    raw = get_price_history(ticker, period="1mo", interval="1d")
    results["price"] = _safe_json(raw)
    current_price = results["price"].get("current_price", "?")
    change_pct = results["price"].get("price_change_pct", 0)
    yield f"✅ **[1/6]** Fiyat: **${current_price}** ({change_pct:+.2f}% son 1 ay)\n\n"

    # 2) Technical
    yield "⏳ **[2/6]** Teknik indikatörler hesaplanıyor…\n"
    raw = get_technical_indicators(ticker, period="3mo")
    results["technical"] = _safe_json(raw)
    tech_signal = results["technical"].get("overall_signal", "N/A")
    rsi_val = results["technical"].get("indicators", {}).get("rsi", {}).get("value", "?")
    macd_sig = results["technical"].get("indicators", {}).get("macd", {}).get("signal", "?")
    yield f"✅ **[2/6]** Teknik Sinyal: **{tech_signal}** (RSI: {rsi_val}, MACD: {macd_sig})\n\n"

    # 3) Sentiment
    yield "⏳ **[3/6]** Haberler taranıyor + sentiment analizi…\n"
    raw = get_news_sentiment(ticker, company_name="")
    results["sentiment"] = _safe_json(raw)
    sent = results["sentiment"].get("overall_sentiment", "N/A")
    n_articles = results["sentiment"].get("news_count", 0)
    yield f"✅ **[3/6]** Sentiment: **{sent}** ({n_articles} haber analiz edildi)\n\n"

    # 4) Fundamental
    yield "⏳ **[4/6]** Temel veriler çekiliyor…\n"
    raw = get_fundamental_data(ticker)
    results["fundamental"] = _safe_json(raw)
    pe = results["fundamental"].get("valuation", {}).get("pe_forward", "N/A")
    rec = results["fundamental"].get("analyst_targets", {}).get("recommendation", "N/A")
    name = results["fundamental"].get("profile", {}).get("name", ticker)
    mcap = results["fundamental"].get("profile", {}).get("market_cap_formatted", "N/A")
    yield f"✅ **[4/6]** Temel: **{name}** — P/E: {pe}, Analist: {rec}, MCap: {mcap}\n\n"

    # 5) Market Overview
    yield "⏳ **[5/6]** Piyasa genel durumu kontrol ediliyor…\n"
    raw = get_market_overview()
    results["market"] = _safe_json(raw)
    regime = results["market"].get("market_regime", "N/A")
    vix = results["market"].get("vix", {}).get("value", "?")
    vix_reg = results["market"].get("vix", {}).get("regime", "?")
    yield f"✅ **[5/6]** Piyasa: **{regime}** (VIX: {vix} — {vix_reg})\n\n"

    # 6) Risk
    yield "⏳ **[6/6]** Risk metrikleri hesaplanıyor…\n"
    raw = calculate_risk_metrics(ticker, portfolio_value=portfolio_value, risk_tolerance=risk_tolerance.lower())
    results["risk"] = _safe_json(raw)
    vol = results["risk"].get("volatility", {}).get("regime", "N/A")
    sl = results["risk"].get("trade_levels", {}).get("stop_loss", "?")
    tp = results["risk"].get("trade_levels", {}).get("take_profit", "?")
    shares = results["risk"].get("position_sizing", {}).get("suggested_shares", "?")
    rr = results["risk"].get("trade_levels", {}).get("risk_reward_ratio", "?")
    yield f"✅ **[6/6]** Risk: Vol **{vol}** | SL: ${sl} | TP: ${tp} | {shares} hisse | R/R: {rr}\n\n"

    # --- Agent'a gönder (final karar) ---
    yield "🎯 **Fund Manager** tüm verileri sentezliyor…\n\n"

    try:
        advisor = get_advisor()
        query = (
            f"I already have all the data collected. Here are the raw analysis results for {ticker}:\n\n"
            f"PRICE DATA: current_price=${current_price}, 1mo_change={change_pct}%\n\n"
            f"TECHNICAL: overall_signal={tech_signal}, RSI={rsi_val}, MACD={macd_sig}, "
            f"Bollinger={results['technical'].get('indicators',{}).get('bollinger_bands',{}).get('signal','N/A')}, "
            f"SMA={results['technical'].get('indicators',{}).get('sma',{}).get('signal','N/A')}, "
            f"Stochastic={results['technical'].get('indicators',{}).get('stochastic',{}).get('signal','N/A')}\n\n"
            f"SENTIMENT: overall={sent}, score={results['sentiment'].get('avg_sentiment_score','N/A')}, "
            f"articles={n_articles}\n\n"
            f"FUNDAMENTAL: company={name}, forward_PE={pe}, recommendation={rec}, "
            f"market_cap={mcap}, "
            f"revenue_growth={results['fundamental'].get('financials',{}).get('revenue_growth','N/A')}, "
            f"profit_margin={results['fundamental'].get('financials',{}).get('profit_margin','N/A')}, "
            f"target_upside={results['fundamental'].get('analyst_targets',{}).get('upside_pct','N/A')}\n\n"
            f"MARKET: regime={regime}, VIX={vix} ({vix_reg})\n\n"
            f"RISK: volatility_regime={vol}, stop_loss=${sl}, take_profit=${tp}, "
            f"suggested_shares={shares}, risk_reward={rr}, "
            f"max_drawdown={results['risk'].get('risk_metrics',{}).get('max_drawdown_6mo','N/A')}, "
            f"sharpe={results['risk'].get('risk_metrics',{}).get('sharpe_ratio_approx','N/A')}\n\n"
            f"Portfolio: ${portfolio_value:,.0f}, Risk tolerance: {risk_tolerance}\n\n"
            f"Based on ALL this data, produce your final TRADE ADVISOR REPORT. "
            f"Do NOT call any agents or tools — just synthesize the data above into your standard report format."
        )
        final = advisor.run(query)
        yield f"\n---\n\n{final}"
    except Exception as e:
        yield f"\n❌ Fund Manager hatası: {e}\n\n"
        yield _build_fallback_report(ticker, name, results, current_price, portfolio_value, risk_tolerance)


def _build_fallback_report(ticker, name, results, price, pv, tol):
    """LLM çöktüğünde tool verilerinden doğrudan rapor üret."""
    tech = results.get("technical", {})
    sent = results.get("sentiment", {})
    fund = results.get("fundamental", {})
    risk = results.get("risk", {})
    mkt = results.get("market", {})

    sig = tech.get("overall_signal", "HOLD")
    if sig in ("STRONG_BUY", "BUY"):
        signal, emoji = "BUY", "🟢"
    elif sig in ("STRONG_SELL", "SELL"):
        signal, emoji = "SELL", "🔴"
    else:
        signal, emoji = "HOLD", "🟡"

    tl = risk.get("trade_levels", {})
    ps = risk.get("position_sizing", {})

    return (
        f"\n═══════════════════════════════════════\n"
        f"{emoji} **TRADE ADVISOR REPORT (Fallback)**\n"
        f"═══════════════════════════════════════\n\n"
        f"**Ticker:** {ticker} | **Şirket:** {name}\n"
        f"**Tarih:** {datetime.date.today()}\n\n"
        f"### {emoji} Sinyal: **{signal}**\n\n"
        f"| Analiz | Sonuç |\n|---|---|\n"
        f"| 📈 Teknik | {tech.get('overall_signal','N/A')} (RSI: {tech.get('indicators',{}).get('rsi',{}).get('value','?')}) |\n"
        f"| 📰 Sentiment | {sent.get('overall_sentiment','N/A')} (skor: {sent.get('avg_sentiment_score','?')}) |\n"
        f"| 📊 Temel | P/E: {fund.get('valuation',{}).get('pe_forward','?')}, Rec: {fund.get('analyst_targets',{}).get('recommendation','?')} |\n"
        f"| ⚠️ Risk | Vol: {risk.get('volatility',{}).get('regime','?')}, VIX: {mkt.get('vix',{}).get('value','?')} |\n\n"
        f"### 💰 Trade Parametreleri\n"
        f"- **Entry:** ${tl.get('entry_price','?')}\n"
        f"- **Stop-Loss:** ${tl.get('stop_loss','?')} ({tl.get('stop_loss_pct','?')})\n"
        f"- **Take-Profit:** ${tl.get('take_profit','?')} ({tl.get('take_profit_pct','?')})\n"
        f"- **Pozisyon:** {ps.get('suggested_shares','?')} hisse (${ps.get('position_value_usd','?')})\n"
        f"- **R/R:** {tl.get('risk_reward_ratio','?')}\n\n"
        f"⚠️ *Bu AI destekli analiz, yatırım tavsiyesi değildir.*\n"
    )


# ============================================================
# Hızlı Analiz (streaming)
# ============================================================
def quick_analysis_streaming(ticker: str, portfolio_value: float, risk_tolerance: str):
    if not ticker or not ticker.strip():
        yield "❌ Lütfen bir ticker sembolü girin (örn: AAPL, NVDA, BTC-USD)"
        return

    accumulated = ""
    for chunk in run_direct_analysis(ticker, portfolio_value, risk_tolerance):
        accumulated += chunk
        yield accumulated


# ============================================================
# Çoklu Ticker Karşılaştırma
# ============================================================
def compare_tickers(tickers_str: str, portfolio_value: float, risk_tolerance: str):
    if not tickers_str or not tickers_str.strip():
        yield "❌ En az 2 ticker girin (virgülle ayırın, örn: AAPL, NVDA, MSFT)"
        return

    tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]
    if len(tickers) < 2:
        yield "❌ En az 2 ticker gerekli."
        return
    if len(tickers) > 5:
        yield "❌ Maksimum 5 ticker karşılaştırılabilir."
        return

    results = {}
    accumulated = f"## 🔄 {len(tickers)} Ticker Karşılaştırılıyor: {', '.join(tickers)}\n\n"
    yield accumulated

    for i, ticker in enumerate(tickers):
        accumulated += f"### {'─'*40}\n### 📊 {ticker} ({i+1}/{len(tickers)})\n\n"
        yield accumulated

        try:
            price_data = _safe_json(get_price_history(ticker, period="1mo"))
            tech_data = _safe_json(get_technical_indicators(ticker, period="3mo"))
            fund_data = _safe_json(get_fundamental_data(ticker))
            risk_data = _safe_json(calculate_risk_metrics(ticker, portfolio_value, risk_tolerance.lower()))

            current_price = price_data.get("current_price", "?")
            change_pct = price_data.get("price_change_pct", 0)
            tech_signal = tech_data.get("overall_signal", "N/A")
            pe = fund_data.get("valuation", {}).get("pe_forward", "N/A")
            rec = fund_data.get("analyst_targets", {}).get("recommendation", "N/A")
            upside = fund_data.get("analyst_targets", {}).get("upside_pct", "N/A")
            rr = risk_data.get("trade_levels", {}).get("risk_reward_ratio", "N/A")
            vol = risk_data.get("volatility", {}).get("regime", "N/A")
            rsi = tech_data.get("indicators", {}).get("rsi", {}).get("value", "?")
            name = fund_data.get("profile", {}).get("name", ticker)
            sharpe = risk_data.get("risk_metrics", {}).get("sharpe_ratio_approx", "?")

            results[ticker] = {
                "name": name, "price": current_price, "change": change_pct,
                "tech": tech_signal, "pe": pe, "rec": rec, "upside": upside,
                "rr": rr, "vol": vol, "rsi": rsi, "sharpe": sharpe,
            }

            accumulated += (
                f"- **{name}** — ${current_price} ({change_pct:+.2f}%)\n"
                f"- Teknik: **{tech_signal}** | RSI: {rsi}\n"
                f"- P/E: {pe} | Analist: {rec} | Upside: {upside}\n"
                f"- Volatilite: {vol} | R/R: {rr} | Sharpe: {sharpe}\n\n"
            )
            yield accumulated

        except Exception as e:
            accumulated += f"- ❌ Hata: {e}\n\n"
            yield accumulated

    # --- Karşılaştırma tablosu ---
    accumulated += f"\n## 📋 Karşılaştırma Tablosu\n\n"
    accumulated += "| Metrik | " + " | ".join(results.keys()) + " |\n"
    accumulated += "|---|" + "|".join(["---"] * len(results)) + "|\n"

    rows = [
        ("💰 Fiyat", lambda r: f"${r['price']}"),
        ("📈 1 Ay Δ", lambda r: f"{r['change']:+.2f}%"),
        ("🔧 Teknik", lambda r: f"**{r['tech']}**"),
        ("📊 RSI", lambda r: str(r['rsi'])),
        ("💵 Forward P/E", lambda r: str(r['pe'])),
        ("🎯 Analist", lambda r: str(r['rec'])),
        ("📈 Upside", lambda r: str(r['upside'])),
        ("⚡ Volatilite", lambda r: str(r['vol'])),
        ("⚖️ R/R", lambda r: str(r['rr'])),
        ("📉 Sharpe", lambda r: str(r['sharpe'])),
    ]

    for label, fn in rows:
        vals = " | ".join(fn(r) for r in results.values())
        accumulated += f"| {label} | {vals} |\n"

    # Basit skor
    accumulated += "\n### 🏆 Sonuç\n\n"
    def _score(r):
        s = 0
        t = str(r['tech'])
        if 'STRONG_BUY' in t: s += 3
        elif 'BUY' in t: s += 2
        elif 'HOLD' in t: s += 1
        elif 'SELL' in t: s -= 1
        elif 'STRONG_SELL' in t: s -= 2
        try:
            if float(r['rsi']) < 40: s += 1
            if float(r['rsi']) > 70: s -= 1
        except: pass
        try:
            if float(r['sharpe']) > 1: s += 1
        except: pass
        rec = str(r.get('rec', '')).lower()
        if rec in ('buy', 'strongbuy', 'strong_buy'): s += 2
        return s

    scored = sorted(results.items(), key=lambda x: _score(x[1]), reverse=True)
    for rank, (t, r) in enumerate(scored, 1):
        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"{rank}.")
        accumulated += f"{medal} **{t}** ({r['name']}) — Teknik: {r['tech']}, Analist: {r['rec']}, R/R: {r['rr']}\n\n"

    accumulated += "\n⚠️ *Bu sıralama basit bir skor sistemidir, kesin yatırım tavsiyesi değildir.*\n"
    yield accumulated


# ============================================================
# Telegram Alert
# ============================================================
def send_telegram_alert(bot_token: str, chat_id: str, ticker: str,
                        portfolio_value: float, risk_tolerance: str):
    if not bot_token or not chat_id:
        yield "❌ Telegram Bot Token ve Chat ID gerekli.\n\n📖 **Nasıl alınır:**\n1. Telegram'da @BotFather'a `/newbot` yazın → token alın\n2. Bota bir mesaj gönderin, sonra `https://api.telegram.org/bot<TOKEN>/getUpdates` adresinden chat_id'yi bulun"
        return

    ticker = ticker.strip().upper()
    if not ticker:
        yield "❌ Ticker gerekli."
        return

    accumulated = f"⏳ {ticker} analizi yapılıyor…\n\n"
    yield accumulated

    try:
        price_data = _safe_json(get_price_history(ticker, period="1mo"))
        tech_data = _safe_json(get_technical_indicators(ticker, period="3mo"))
        risk_data = _safe_json(calculate_risk_metrics(ticker, portfolio_value, risk_tolerance.lower()))

        current_price = price_data.get("current_price", "?")
        tech_signal = tech_data.get("overall_signal", "N/A")
        rsi = tech_data.get("indicators", {}).get("rsi", {}).get("value", "?")
        macd = tech_data.get("indicators", {}).get("macd", {}).get("signal", "?")
        tl = risk_data.get("trade_levels", {})

        if tech_signal in ("STRONG_BUY", "BUY"):
            emoji = "🟢"
        elif tech_signal in ("STRONG_SELL", "SELL"):
            emoji = "🔴"
        else:
            emoji = "🟡"

        msg = (
            f"{emoji} *Trade Bot Alert — {ticker}*\n\n"
            f"💰 Fiyat: ${current_price}\n"
            f"📈 Sinyal: *{tech_signal}*\n"
            f"📊 RSI: {rsi} | MACD: {macd}\n"
            f"🛑 SL: ${tl.get('stop_loss','?')} | 🎯 TP: ${tl.get('take_profit','?')}\n"
            f"⚖️ R/R: {tl.get('risk_reward_ratio','?')}\n\n"
            f"⚠️ _Yatırım tavsiyesi değildir._"
        )

        import requests
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )

        if resp.ok:
            accumulated += f"✅ Telegram'a gönderildi!\n\n**Gönderilen mesaj:**\n\n{msg}"
        else:
            accumulated += f"❌ Telegram hatası: {resp.text}"
        yield accumulated

    except Exception as e:
        yield f"❌ Hata: {e}"


# ============================================================
# Serbest Sohbet
# ============================================================
def chat_analysis(message: str, history: list):
    if not message or not message.strip():
        return "Lütfen bir soru sorun. Örnek: 'AAPL hissesini analiz et, portföyüm $100K'"

    try:
        advisor = get_advisor()
        result = advisor.run(message)
        return str(result)
    except Exception as e:
        return f"❌ Hata: {str(e)}"


# ============================================================
# Gradio Interface
# ============================================================
DESCRIPTION = """
# 🤖 Agentic Trade Bot Advisor

**5 AI agent** koordineli çalışarak kapsamlı trade analizi üretir.

| Agent | Görev |
|---|---|
| 📈 **Technical Analyst** | RSI, MACD, Bollinger Bands, SMA, ATR, Stochastic |
| 📰 **Sentiment Analyst** | Haber analizi + piyasa duyarlılığı |
| 📊 **Fundamental Analyst** | P/E, PEG, marjlar, büyüme, bilanço |
| ⚠️ **Risk Manager** | VaR, position sizing, stop-loss, piyasa rejimi |
| 🎯 **Fund Manager** | Tüm raporları sentezler, final trade kararı verir |

> ⚠️ **Disclaimer**: Bu AI destekli bir analiz aracıdır. Yatırım tavsiyesi değildir.
"""

# ============================================================
# Kullanım Rehberi (Tab 5 içeriği)
# ============================================================
USAGE_GUIDE = """
# 📖 Kullanım Rehberi

## 🖥️ Bu Sistemi Nasıl Kullanırım?

### Yöntem 1: Web Arayüzü (Şu an buradasın)
En kolay yol. Tarayıcıdan aç, ticker gir, butona bas.

### Yöntem 2: Kendi Bilgisayarında (Terminal / CLI)

```bash
# 1) Repoyu klonla
git clone https://huggingface.co/spaces/SutskeverFanBoy/trade-bot-advisor
cd trade-bot-advisor

# 2) Bağımlılıkları yükle
pip install -r requirements.txt

# 3) HF Token'ı ayarla (ücretsiz — hf.co/settings/tokens)
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# 4a) Web arayüzünü başlat
python app.py
# → http://localhost:7860 adresini aç

# 4b) VEYA terminal'de direkt analiz
python cli.py NVDA --portfolio 100000 --risk moderate

# 4c) VEYA Python'dan import et
python -c "
from agent import run_analysis
result = run_analysis('Analyze AAPL. Portfolio: 50K, moderate risk.')
print(result)
"
```

### Yöntem 3: Zamanlanmış Otomatik Analiz (Cron Job)

```bash
# Her gün saat 09:30'da NVDA + AAPL + BTC-USD analiz et, Telegram'a gönder
# crontab -e ile aç:
30 9 * * 1-5 cd /path/to/trade-bot-advisor && python cli.py NVDA AAPL BTC-USD --telegram --bot-token "TOKEN" --chat-id "ID"
```

---

## 🔔 Alert Sistemi Nasıl Çalışır?

### Anlık Alert (Tab 3)
1. Telegram Bot Token + Chat ID gir
2. Ticker seç, "Analiz Et & Gönder" butonuna tıkla
3. Sistem analiz yapar ve sonucu Telegram'a gönderir

### Otomatik Alert (CLI ile)
```bash
# Sadece BUY/SELL sinyallerinde bildirim gönder
python cli.py NVDA --telegram --bot-token "123:ABC" --chat-id "-100123" --alert-only
```

### Ne Zaman Alert Gelir?
| Durum | Alert |
|---|---|
| Teknik sinyal **STRONG_BUY** veya **STRONG_SELL** | ✅ Gönderilir |
| Sinyal HOLD, piyasa normal | ❌ Gönderilmez (--alert-only modda) |

---

## ❓ Sık Sorulan Sorular

**S: HF_TOKEN nereden alınır?**
→ [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — ücretsiz, "Read" yetkisi yeterli.

**S: Hangi ticker formatlarını destekliyor?**
→ Yahoo Finance formatı: `AAPL`, `BTC-USD`, `GC=F`, `^GSPC`, `SPY`

**S: Analiz ne kadar sürer?**
→ Hızlı Analiz: ~15-30 saniye | Serbest Sohbet: ~60-120 saniye

**S: Veriler gerçek zamanlı mı?**
→ Hayır, ~15 dakika gecikmeli. Gün içi trading için uygun değil.

**S: Ücretsiz mi?**
→ Evet. HF Inference API ücretsiz tier kullanılır.
"""


with gr.Blocks(
    title="🤖 Trade Bot Advisor",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray"),
) as demo:

    gr.Markdown(DESCRIPTION)

    with gr.Tabs():
        # TAB 1: Hızlı Analiz
        with gr.Tab("⚡ Hızlı Analiz"):
            with gr.Row():
                with gr.Column(scale=1):
                    ticker_input = gr.Textbox(label="📌 Ticker Sembolü", placeholder="AAPL, NVDA, BTC-USD…", value="NVDA")
                    portfolio_input = gr.Number(label="💰 Portföy Değeri ($)", value=100000, minimum=1000)
                    risk_input = gr.Radio(label="⚖️ Risk Toleransı", choices=["Conservative", "Moderate", "Aggressive"], value="Moderate")
                    analyze_btn = gr.Button("🔍 Analiz Başlat", variant="primary", size="lg")
                    gr.Markdown("### 📝 Desteklenen: AAPL, NVDA, BTC-USD, SPY, GC=F…")
                with gr.Column(scale=2):
                    output = gr.Markdown(value="*Analiz başlatmak için butona tıklayın…*")
            analyze_btn.click(fn=quick_analysis_streaming, inputs=[ticker_input, portfolio_input, risk_input], outputs=output)

        # TAB 2: Karşılaştırma
        with gr.Tab("🔄 Karşılaştırma"):
            gr.Markdown("### Birden fazla ticker'ı yan yana karşılaştırın (2–5 ticker, virgülle)")
            with gr.Row():
                with gr.Column(scale=1):
                    compare_input = gr.Textbox(label="📌 Ticker'lar", placeholder="AAPL, NVDA, MSFT", value="AAPL, NVDA, MSFT")
                    compare_portfolio = gr.Number(label="💰 Portföy ($)", value=100000, minimum=1000)
                    compare_risk = gr.Radio(label="⚖️ Risk", choices=["Conservative", "Moderate", "Aggressive"], value="Moderate")
                    compare_btn = gr.Button("🔄 Karşılaştır", variant="primary", size="lg")
                with gr.Column(scale=2):
                    compare_output = gr.Markdown(value="*Karşılaştırma başlatmak için butona tıklayın…*")
            compare_btn.click(fn=compare_tickers, inputs=[compare_input, compare_portfolio, compare_risk], outputs=compare_output)

        # TAB 3: Telegram Alert
        with gr.Tab("🔔 Telegram Alert"):
            gr.Markdown("### Analiz sonucunu Telegram'a gönder\n**Kurulum:** @BotFather → /newbot → Token al, bota mesaj gönder, getUpdates'den chat_id bul")
            with gr.Row():
                with gr.Column():
                    tg_token = gr.Textbox(label="🤖 Bot Token", type="password")
                    tg_chat = gr.Textbox(label="💬 Chat ID")
                    tg_ticker = gr.Textbox(label="📌 Ticker", value="NVDA")
                    tg_portfolio = gr.Number(label="💰 Portföy ($)", value=100000, minimum=1000)
                    tg_risk = gr.Radio(label="⚖️ Risk", choices=["Conservative", "Moderate", "Aggressive"], value="Moderate")
                    tg_btn = gr.Button("📤 Analiz Et & Gönder", variant="primary")
                with gr.Column():
                    tg_output = gr.Markdown(value="*Bilgileri doldurun…*")
            tg_btn.click(fn=send_telegram_alert, inputs=[tg_token, tg_chat, tg_ticker, tg_portfolio, tg_risk], outputs=tg_output)

        # TAB 4: Serbest Sohbet
        with gr.Tab("💬 Serbest Sohbet"):
            gr.Markdown("Doğal dilde soru sorun. Agent'lar otomatik olarak gerekli araçları kullanacak.")
            chatbot = gr.ChatInterface(fn=chat_analysis, type="messages", examples=[
                "Analyze AAPL for a swing trade. Portfolio $50K, moderate risk.",
                "What's the overall market sentiment right now?",
                "Compare NVDA and AMD - which is a better buy?",
            ])

        # TAB 5: Rehber
        with gr.Tab("📖 Rehber"):
            gr.Markdown(USAGE_GUIDE)

    gr.Markdown("---\n**Built with** 🤗 smolagents + yfinance + pandas-ta | **⚠️ Not financial advice**")


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
