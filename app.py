"""
🤖 Trade Bot Advisor - Gradio UI (Faz 1 + BIST/TR)
=====================================================
BIST hisseleri, altın/gümüş TRY, KAP entegrasyonu, günlük dashboard.
"""

import os, json, time, datetime
import gradio as gr
from agent import create_trade_advisor
from tools.price_history import get_price_history
from tools.technical_indicators import get_technical_indicators
from tools.news_sentiment import get_news_sentiment
from tools.fundamental_analysis import get_fundamental_data
from tools.risk_calculator import calculate_risk_metrics
from tools.market_overview import get_market_overview
from tools.bist_scanner import get_bist_scanner
from tools.kap_disclosures import get_kap_disclosures
from tools.daily_dashboard import get_daily_dashboard

_advisor = None
def get_advisor():
    global _advisor
    if _advisor is None:
        _advisor = create_trade_advisor(hf_token=os.environ.get("HF_TOKEN",""), model_id=os.environ.get("MODEL_ID","Qwen/Qwen2.5-72B-Instruct"))
    return _advisor

def _safe_json(raw):
    try: return json.loads(raw)
    except: return {"error": raw}

# ============================================================
# GÜNLÜK DASHBOARD (yeni)
# ============================================================
def run_daily_dashboard(market, portfolio_value):
    accumulated = f"## 📊 Günlük Dashboard Oluşturuluyor…\n**Piyasa:** {market} | **Portföy:** ${portfolio_value:,.0f}\n\n"
    yield accumulated

    accumulated += "⏳ BIST + US hisseleri taranıyor, emtia verileri çekiliyor…\n\n"
    yield accumulated

    raw = get_daily_dashboard(market=market.lower(), portfolio_value=portfolio_value)
    d = _safe_json(raw)

    if "error" in d:
        yield accumulated + f"❌ Hata: {d['error']}"
        return

    # Piyasa özeti
    ms = d.get("market_summary", {})
    accumulated += "## 🌍 Piyasa Durumu\n\n"
    if "bist100" in ms:
        b = ms["bist100"]
        accumulated += f"🇹🇷 **BIST 100:** {b['value']:,.2f} ({b['change_pct']:+.2f}%)\n\n"
    if "sp500" in ms:
        s = ms["sp500"]
        accumulated += f"🇺🇸 **S&P 500:** {s['value']:,.2f} ({s['change_pct']:+.2f}%)\n\n"
    if "vix" in ms:
        accumulated += f"😱 **VIX:** {ms['vix']}\n\n"

    # Emtia
    cm = d.get("commodities_try", {})
    if cm:
        accumulated += "## 🪙 Emtia & Döviz\n\n"
        accumulated += f"| Varlık | Fiyat |\n|---|---|\n"
        accumulated += f"| 🥇 Altın | **₺{cm.get('gold_try_gram','?')}/gram** (${cm.get('gold_usd_oz','?')}/oz) |\n"
        accumulated += f"| 🥈 Gümüş | **₺{cm.get('silver_try_gram','?')}/gram** |\n"
        accumulated += f"| 💵 USD/TRY | **{cm.get('usd_try','?')}** |\n\n"

    # Alertler
    alerts = d.get("alerts", [])
    if alerts:
        accumulated += "## 🚨 Uyarılar\n\n"
        for a in alerts:
            accumulated += f"- {a}\n"
        accumulated += "\n"
    yield accumulated

    # AL sinyalleri
    buys = d.get("buy_recommendations", [])
    accumulated += f"## 📈 AL Sinyalleri ({len(buys)} hisse)\n\n"
    if buys:
        accumulated += "| Hisse | Fiyat | Değişim | RSI | Sinyal |\n|---|---|---|---|---|\n"
        for b in buys:
            accumulated += f"| **{b['ticker']}** ({b['name']}) | {b['price']} | {b['change_pct']:+.2f}% | {b['rsi']} | {b['signal']} |\n"
    else:
        accumulated += "*Bugün AL sinyali yok.*\n"
    accumulated += "\n"
    yield accumulated

    # SAT sinyalleri
    sells = d.get("sell_recommendations", [])
    accumulated += f"## 📉 SAT Sinyalleri ({len(sells)} hisse)\n\n"
    if sells:
        accumulated += "| Hisse | Fiyat | Değişim | RSI | Sinyal |\n|---|---|---|---|---|\n"
        for s in sells:
            accumulated += f"| **{s['ticker']}** ({s['name']}) | {s['price']} | {s['change_pct']:+.2f}% | {s['rsi']} | {s['signal']} |\n"
    else:
        accumulated += "*Bugün SAT sinyali yok.*\n"
    accumulated += "\n"
    yield accumulated

    # Top gainers/losers
    accumulated += "## 🏆 Günün Yıldızları\n\n"
    accumulated += "| # | Hisse | Değişim |\n|---|---|---|\n"
    for i, g in enumerate(d.get("top_gainers", [])[:5], 1):
        accumulated += f"| 🟢 {i} | **{g['ticker']}** ({g['name']}) | **{g['change_pct']:+.2f}%** |\n"
    accumulated += "\n## 📉 Günün Düşenleri\n\n"
    accumulated += "| # | Hisse | Değişim |\n|---|---|---|\n"
    for i, l in enumerate(d.get("top_losers", [])[:5], 1):
        accumulated += f"| 🔴 {i} | **{l['ticker']}** ({l['name']}) | **{l['change_pct']:+.2f}%** |\n"

    accumulated += f"\n---\n📋 **Özet:** {d.get('summary','')}\n\n⚠️ *Yatırım tavsiyesi değildir.*"
    yield accumulated


# ============================================================
# Hızlı analiz (mevcut — streaming)
# ============================================================
def run_direct_analysis(ticker, pv, rt):
    ticker = ticker.strip().upper()
    results = {}
    yield "⏳ **[1/6]** Fiyat verisi çekiliyor…\n"
    results["price"] = _safe_json(get_price_history(ticker, period="1mo", interval="1d"))
    cp = results["price"].get("current_price", "?"); chg = results["price"].get("price_change_pct", 0)
    yield f"✅ **[1/6]** Fiyat: **${cp}** ({chg:+.2f}%)\n\n"
    yield "⏳ **[2/6]** Teknik indikatörler…\n"
    results["technical"] = _safe_json(get_technical_indicators(ticker, period="3mo"))
    ts = results["technical"].get("overall_signal","N/A"); rsi = results["technical"].get("indicators",{}).get("rsi",{}).get("value","?"); macd = results["technical"].get("indicators",{}).get("macd",{}).get("signal","?")
    yield f"✅ **[2/6]** Teknik: **{ts}** (RSI: {rsi}, MACD: {macd})\n\n"
    yield "⏳ **[3/6]** Haberler + sentiment…\n"
    results["sentiment"] = _safe_json(get_news_sentiment(ticker, company_name=""))
    sent = results["sentiment"].get("overall_sentiment","N/A"); na = results["sentiment"].get("news_count",0)
    yield f"✅ **[3/6]** Sentiment: **{sent}** ({na} haber)\n\n"
    yield "⏳ **[4/6]** Temel veriler…\n"
    results["fundamental"] = _safe_json(get_fundamental_data(ticker))
    pe = results["fundamental"].get("valuation",{}).get("pe_forward","N/A"); rec = results["fundamental"].get("analyst_targets",{}).get("recommendation","N/A")
    name = results["fundamental"].get("profile",{}).get("name",ticker); mcap = results["fundamental"].get("profile",{}).get("market_cap_formatted","N/A")
    yield f"✅ **[4/6]** **{name}** — P/E: {pe}, Analist: {rec}, MCap: {mcap}\n\n"
    yield "⏳ **[5/6]** Piyasa durumu…\n"
    results["market"] = _safe_json(get_market_overview())
    regime = results["market"].get("market_regime","N/A"); vix = results["market"].get("vix",{}).get("value","?"); vr = results["market"].get("vix",{}).get("regime","?")
    yield f"✅ **[5/6]** Piyasa: **{regime}** (VIX: {vix} — {vr})\n\n"
    yield "⏳ **[6/6]** Risk metrikleri…\n"
    results["risk"] = _safe_json(calculate_risk_metrics(ticker, portfolio_value=pv, risk_tolerance=rt.lower()))
    vol = results["risk"].get("volatility",{}).get("regime","N/A"); sl = results["risk"].get("trade_levels",{}).get("stop_loss","?"); tp = results["risk"].get("trade_levels",{}).get("take_profit","?")
    shares = results["risk"].get("position_sizing",{}).get("suggested_shares","?"); rr = results["risk"].get("trade_levels",{}).get("risk_reward_ratio","?")
    yield f"✅ **[6/6]** Risk: Vol **{vol}** | SL: ${sl} | TP: ${tp} | {shares} hisse | R/R: {rr}\n\n"
    yield "🎯 **Fund Manager** sentezliyor…\n\n"
    try:
        advisor = get_advisor()
        q = f"Data for {ticker}: price=${cp} chg={chg}%, tech={ts} RSI={rsi} MACD={macd}, sentiment={sent}, fundamental: {name} PE={pe} rec={rec} mcap={mcap}, market={regime} VIX={vix}, risk: vol={vol} SL=${sl} TP=${tp} shares={shares} RR={rr}. Portfolio ${pv:,.0f}, risk={rt}. Produce TRADE ADVISOR REPORT. Do NOT call tools."
        yield f"\n---\n\n{advisor.run(q)}"
    except Exception as e:
        sig = ts; emoji = "🟢" if "BUY" in sig else ("🔴" if "SELL" in sig else "🟡")
        signal = "BUY" if "BUY" in sig else ("SELL" if "SELL" in sig else "HOLD")
        tl = results["risk"].get("trade_levels",{}); ps = results["risk"].get("position_sizing",{})
        yield f"\n{emoji} **{signal}** — {name} ({ticker})\n\n| | |\n|---|---|\n| Entry | ${tl.get('entry_price','?')} |\n| SL | ${sl} |\n| TP | ${tp} |\n| Hisse | {shares} |\n| R/R | {rr} |\n\n⚠️ *Yatırım tavsiyesi değildir.*"


def quick_analysis_streaming(ticker, pv, rt):
    if not ticker or not ticker.strip(): yield "❌ Ticker girin"; return
    a = ""
    for c in run_direct_analysis(ticker, pv, rt): a += c; yield a


def compare_tickers(ts, pv, rt):
    if not ts or not ts.strip(): yield "❌ En az 2 ticker girin"; return
    tickers = [t.strip().upper() for t in ts.split(",") if t.strip()]
    if len(tickers) < 2: yield "❌ En az 2 ticker"; return
    if len(tickers) > 5: yield "❌ Max 5 ticker"; return
    results = {}; a = f"## 🔄 {', '.join(tickers)} Karşılaştırılıyor\n\n"; yield a
    for i, t in enumerate(tickers):
        a += f"### 📊 {t} ({i+1}/{len(tickers)})\n"; yield a
        try:
            pd = _safe_json(get_price_history(t,"1mo")); td = _safe_json(get_technical_indicators(t,"3mo"))
            fd = _safe_json(get_fundamental_data(t)); rd = _safe_json(calculate_risk_metrics(t,pv,rt.lower()))
            r = {"name":fd.get("profile",{}).get("name",t),"price":pd.get("current_price","?"),"change":pd.get("price_change_pct",0),"tech":td.get("overall_signal","?"),"pe":fd.get("valuation",{}).get("pe_forward","?"),"rec":fd.get("analyst_targets",{}).get("recommendation","?"),"rsi":td.get("indicators",{}).get("rsi",{}).get("value","?"),"rr":rd.get("trade_levels",{}).get("risk_reward_ratio","?"),"vol":rd.get("volatility",{}).get("regime","?"),"sharpe":rd.get("risk_metrics",{}).get("sharpe_ratio_approx","?")}
            results[t] = r
            a += f"- **{r['name']}** ${r['price']} ({r['change']:+.2f}%) Tech:{r['tech']} RSI:{r['rsi']}\n\n"; yield a
        except Exception as e: a += f"❌ {e}\n\n"; yield a
    if results:
        a += "\n## 📋 Tablo\n\n| Metrik | "+" | ".join(results.keys())+" |\n|---|"+("|".join(["---"]*len(results)))+"|\n"
        for l,f in [("Fiyat",lambda r:f"${r['price']}"),("Teknik",lambda r:f"**{r['tech']}**"),("RSI",lambda r:str(r['rsi'])),("P/E",lambda r:str(r['pe'])),("R/R",lambda r:str(r['rr']))]:
            a += f"| {l} | "+" | ".join(f(r) for r in results.values())+" |\n"
    a += "\n⚠️ *Yatırım tavsiyesi değildir.*\n"; yield a


def send_telegram_alert(bt, ci, t, pv, rt):
    if not bt or not ci: yield "❌ Bot Token + Chat ID gerekli"; return
    t = t.strip().upper()
    if not t: yield "❌ Ticker gerekli"; return
    yield f"⏳ {t} analiz ediliyor…\n"
    try:
        td = _safe_json(get_technical_indicators(t,"3mo")); rd = _safe_json(calculate_risk_metrics(t,pv,rt.lower()))
        ts = td.get("overall_signal","?"); rsi = td.get("indicators",{}).get("rsi",{}).get("value","?")
        tl = rd.get("trade_levels",{}); e = "🟢" if "BUY" in ts else ("🔴" if "SELL" in ts else "🟡")
        msg = f"{e} *{t}* — *{ts}*\nRSI: {rsi}\nSL: ${tl.get('stop_loss','?')} | TP: ${tl.get('take_profit','?')}\nR/R: {tl.get('risk_reward_ratio','?')}"
        import requests
        r = requests.post(f"https://api.telegram.org/bot{bt}/sendMessage",json={"chat_id":ci,"text":msg,"parse_mode":"Markdown"},timeout=10)
        yield f"✅ Gönderildi!\n\n{msg}" if r.ok else f"❌ {r.text}"
    except Exception as ex: yield f"❌ {ex}"


def chat_analysis(message, history):
    if not message or not message.strip(): return "Bir soru sorun."
    try: return str(get_advisor().run(message))
    except Exception as e: return f"❌ {e}"


# ============================================================
# GRADIO UI
# ============================================================
DESC = """
# 🤖 Agentic Trade Bot Advisor

**6 AI agent** — BIST + NASDAQ + Kripto + Altın/Gümüş + KAP

| Agent | Görev |
|---|---|
| 📈 Technical | RSI, MACD, Bollinger, SMA, ATR, Stochastic |
| 📰 Sentiment | Haber + KAP bildirimleri |
| 📊 Fundamental | P/E, PEG, marjlar, bilanço |
| 🇹🇷 BIST Analyst | BIST30, bankalar, altın ₺, USD/TRY, günlük tarama |
| ⚠️ Risk Manager | VaR, pozisyon, stop-loss, piyasa rejimi |
| 🎯 Fund Manager | Tüm raporları sentezler, final karar |

> ⚠️ Yatırım tavsiyesi değildir.
"""

with gr.Blocks(title="🤖 Trade Bot Advisor") as demo:
    gr.Markdown(DESC)
    with gr.Tabs():

        # TAB 1: Dashboard
        with gr.Tab("📊 Günlük Dashboard"):
            gr.Markdown("### Günlük AL/SAT taraması — BIST + US + Emtia")
            with gr.Row():
                with gr.Column(scale=1):
                    dash_market = gr.Radio(label="🌍 Piyasa", choices=["BIST", "US", "Both"], value="Both")
                    dash_portfolio = gr.Number(label="💰 Portföy ($)", value=100000, minimum=1000)
                    dash_btn = gr.Button("📊 Dashboard Oluştur", variant="primary", size="lg")
                with gr.Column(scale=2):
                    dash_output = gr.Markdown(value="*Dashboard oluşturmak için butona tıklayın…*")
            dash_btn.click(fn=run_daily_dashboard, inputs=[dash_market, dash_portfolio], outputs=dash_output)

        # TAB 2: Hızlı Analiz
        with gr.Tab("⚡ Hızlı Analiz"):
            gr.Markdown("### Tek hisse detaylı analiz\n**BIST:** THYAO.IS, GARAN.IS | **US:** AAPL, NVDA | **Kripto:** BTC-USD | **Emtia:** GC=F")
            with gr.Row():
                with gr.Column(scale=1):
                    ticker_input = gr.Textbox(label="📌 Ticker", placeholder="THYAO.IS, AAPL, BTC-USD…", value="THYAO.IS")
                    portfolio_input = gr.Number(label="💰 Portföy ($)", value=100000, minimum=1000)
                    risk_input = gr.Radio(label="⚖️ Risk", choices=["Conservative","Moderate","Aggressive"], value="Moderate")
                    analyze_btn = gr.Button("🔍 Analiz", variant="primary", size="lg")
                with gr.Column(scale=2):
                    output = gr.Markdown(value="*Butona tıklayın…*")
            analyze_btn.click(fn=quick_analysis_streaming, inputs=[ticker_input, portfolio_input, risk_input], outputs=output)

        # TAB 3: Karşılaştırma
        with gr.Tab("🔄 Karşılaştırma"):
            gr.Markdown("### 2–5 ticker karşılaştırın\n**Örnek:** THYAO.IS, GARAN.IS, AKBNK.IS veya AAPL, NVDA, MSFT")
            with gr.Row():
                with gr.Column(scale=1):
                    cmp_in = gr.Textbox(label="📌 Ticker'lar", value="THYAO.IS, GARAN.IS, AKBNK.IS")
                    cmp_pv = gr.Number(label="💰 Portföy ($)", value=100000, minimum=1000)
                    cmp_rt = gr.Radio(label="⚖️ Risk", choices=["Conservative","Moderate","Aggressive"], value="Moderate")
                    cmp_btn = gr.Button("🔄 Karşılaştır", variant="primary", size="lg")
                with gr.Column(scale=2):
                    cmp_out = gr.Markdown(value="*Butona tıklayın…*")
            cmp_btn.click(fn=compare_tickers, inputs=[cmp_in, cmp_pv, cmp_rt], outputs=cmp_out)

        # TAB 4: Telegram
        with gr.Tab("🔔 Telegram"):
            gr.Markdown("### Analiz → Telegram\n@BotFather → /newbot → Token, bota mesaj → getUpdates → chat_id")
            with gr.Row():
                with gr.Column():
                    tg_t = gr.Textbox(label="🤖 Token", type="password")
                    tg_c = gr.Textbox(label="💬 Chat ID")
                    tg_tk = gr.Textbox(label="📌 Ticker", value="THYAO.IS")
                    tg_pv = gr.Number(label="💰 Portföy ($)", value=100000)
                    tg_rt = gr.Radio(label="⚖️ Risk", choices=["Conservative","Moderate","Aggressive"], value="Moderate")
                    tg_btn = gr.Button("📤 Gönder", variant="primary")
                with gr.Column():
                    tg_out = gr.Markdown(value="*Bilgileri doldurun…*")
            tg_btn.click(fn=send_telegram_alert, inputs=[tg_t, tg_c, tg_tk, tg_pv, tg_rt], outputs=tg_out)

        # TAB 5: Sohbet
        with gr.Tab("💬 Sohbet"):
            gr.Markdown("Doğal dilde soru sorun — Türkçe veya İngilizce.")
            gr.ChatInterface(fn=chat_analysis, examples=[
                "THYAO hissesini analiz et, portföyüm 500K TL, orta risk",
                "Bugün BIST'te hangi hisseleri almalıyım?",
                "Altın fiyatı ne kadar? Almalı mıyım?",
                "Compare NVDA and AMD",
                "GARAN.IS için KAP'ta son bildirimler neler?",
            ])

        # TAB 6: Rehber
        with gr.Tab("📖 Rehber"):
            gr.Markdown("""
# 📖 Kullanım Rehberi

## Desteklenen Varlıklar

| Tür | Format | Örnekler |
|---|---|---|
| 🇹🇷 BIST Hisseleri | `TICKER.IS` | THYAO.IS, GARAN.IS, AKBNK.IS, ASELS.IS |
| 🇺🇸 US/NASDAQ | `TICKER` | AAPL, NVDA, MSFT, TSLA, GOOGL |
| ₿ Kripto | `TICKER-USD` | BTC-USD, ETH-USD, SOL-USD |
| 🪙 Emtia | Futures | GC=F (Altın), SI=F (Gümüş), CL=F (Petrol) |
| 📦 ETF | `TICKER` | SPY, QQQ, IWM |
| 💱 Döviz | `XXX=X` | USDTRY=X, EURTRY=X |

## Dashboard Nasıl Çalışır?
1. "📊 Günlük Dashboard" tab'ına git
2. Piyasa seç (BIST / US / Both)
3. Sistem ~35 hisseyi tarar + altın/gümüş/döviz ekler
4. AL sinyali: RSI < 40 + SMA20 > SMA50 (yükselen trend + oversold)
5. SAT sinyali: RSI > 65 + SMA20 < SMA50 (düşen trend + overbought)

## CLI Kullanımı
```bash
git clone https://huggingface.co/spaces/SutskeverFanBoy/trade-bot-advisor
cd trade-bot-advisor && pip install -r requirements.txt
export HF_TOKEN="hf_xxx"
python cli.py THYAO.IS GARAN.IS --portfolio 500000 --risk moderate
python cli.py NVDA AAPL --quiet --telegram --bot-token "..." --chat-id "..." --alert-only
```
""")

    gr.Markdown("---\n🤗 smolagents + yfinance + pandas-ta + KAP | ⚠️ Yatırım tavsiyesi değildir")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray"))
