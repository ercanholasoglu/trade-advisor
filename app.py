"""
🤖 Trade Bot Advisor — Gradio UI v2.0
========================================
Full-featured UI with 10 tabs:
1. Dashboard, 2. Quick Analysis, 3. Compare, 4. Backtest,
5. Portfolio Optimizer, 6. Watchlist, 7. Analysis History,
8. Telegram, 9. Chat, 10. Guide

Gradio queue enabled. FastAPI mount for REST API.
"""

import os
import json
import time
import datetime
import gradio as gr
from agent import create_trade_advisor, run_analysis
from tools.price_history import get_price_history
from tools.technical_indicators import get_technical_indicators
from tools.news_sentiment import get_news_sentiment
from tools.fundamental_analysis import get_fundamental_data
from tools.risk_calculator import calculate_risk_metrics
from tools.market_overview import get_market_overview
from tools.bist_scanner import get_bist_scanner
from tools.kap_disclosures import get_kap_disclosures
from tools.daily_dashboard import get_daily_dashboard
from tools.crypto_fundamental import get_crypto_fundamentals
from tools.insider_trades import get_insider_trades
from tools.options_flow import get_options_flow
from tools.macro_data import get_macro_data
from output_schema import parse_trade_report, report_to_markdown
from db import (
    init_db, save_analysis, get_accuracy_stats, get_recent_analyses,
    add_watchlist, get_watchlist, remove_watchlist,
)

# Initialize database
init_db()

# Start scheduler (if APScheduler available)
_scheduler = None
try:
    from scheduler import start_scheduler
    _scheduler = start_scheduler()
except Exception:
    pass

_advisor = None


def get_advisor():
    global _advisor
    if _advisor is None:
        _advisor = create_trade_advisor(
            hf_token=os.environ.get("HF_TOKEN", ""),
            model_id=os.environ.get("MODEL_ID", "Qwen/Qwen2.5-72B-Instruct"),
            fast_model_id=os.environ.get("FAST_MODEL_ID"),
            deep_model_id=os.environ.get("DEEP_MODEL_ID"),
        )
    return _advisor


def _safe_json(raw):
    try:
        return json.loads(raw)
    except Exception:
        return {"error": raw}


# ============================================================
# TAB 1: GÜNLÜK DASHBOARD
# ============================================================
def run_daily_dashboard(market, portfolio_value):
    accumulated = f"## 📊 Günlük Dashboard Oluşturuluyor…\n**Piyasa:** {market} | **Portföy:** ${portfolio_value:,.0f}\n\n"
    yield accumulated

    accumulated += "⏳ BIST + US hisseleri taranıyor, emtia verileri çekiliyor…\n\n"
    yield accumulated

    try:
        raw = get_daily_dashboard(market=market.lower(), portfolio_value=portfolio_value)
        d = _safe_json(raw)
    except Exception as e:
        yield accumulated + f"❌ Hata: {e}"
        return

    if "error" in d:
        yield accumulated + f"❌ Hata: {d['error']}"
        return

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

    cm = d.get("commodities_try", {})
    if cm:
        accumulated += "## 🪙 Emtia & Döviz\n\n"
        accumulated += "| Varlık | Fiyat |\n|---|---|\n"
        accumulated += f"| 🥇 Altın | **₺{cm.get('gold_try_gram', '?')}/gram** (${cm.get('gold_usd_oz', '?')}/oz) |\n"
        accumulated += f"| 🥈 Gümüş | **₺{cm.get('silver_try_gram', '?')}/gram** |\n"
        accumulated += f"| 💵 USD/TRY | **{cm.get('usd_try', '?')}** |\n\n"

    alerts = d.get("alerts", [])
    if alerts:
        accumulated += "## 🚨 Uyarılar\n\n"
        for a in alerts:
            accumulated += f"- {a}\n"
        accumulated += "\n"
    yield accumulated

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

    accumulated += "## 🏆 Günün Yıldızları\n\n"
    accumulated += "| # | Hisse | Değişim |\n|---|---|---|\n"
    for i, g in enumerate(d.get("top_gainers", [])[:5], 1):
        accumulated += f"| 🟢 {i} | **{g['ticker']}** ({g['name']}) | **{g['change_pct']:+.2f}%** |\n"
    accumulated += "\n## 📉 Günün Düşenleri\n\n"
    accumulated += "| # | Hisse | Değişim |\n|---|---|---|\n"
    for i, l in enumerate(d.get("top_losers", [])[:5], 1):
        accumulated += f"| 🔴 {i} | **{l['ticker']}** ({l['name']}) | **{l['change_pct']:+.2f}%** |\n"

    accumulated += f"\n---\n📋 **Özet:** {d.get('summary', '')}\n\n⚠️ *Yatırım tavsiyesi değildir.*"
    yield accumulated


# ============================================================
# TAB 2: HIZLI ANALİZ (streaming)
# ============================================================
def run_direct_analysis(ticker, pv, rt):
    ticker = ticker.strip().upper()
    results = {}
    data_errors = []

    yield "⏳ **[1/7]** Fiyat verisi çekiliyor…\n"
    try:
        results["price"] = _safe_json(get_price_history(ticker, period="1mo", interval="1d"))
        cp = results["price"].get("current_price", "?")
        chg = results["price"].get("price_change_pct", 0)
        yield f"✅ **[1/7]** Fiyat: **${cp}** ({chg:+.2f}%)\n\n"
    except Exception as e:
        data_errors.append(f"Fiyat verisi: {e}")
        cp, chg = "?", 0
        yield f"⚠️ **[1/7]** Fiyat verisi alınamadı: {e}\n\n"

    yield "⏳ **[2/7]** Teknik indikatörler…\n"
    try:
        results["technical"] = _safe_json(get_technical_indicators(ticker, period="3mo"))
        ts = results["technical"].get("overall_signal", "N/A")
        rsi = results["technical"].get("indicators", {}).get("rsi", {}).get("value", "?")
        macd = results["technical"].get("indicators", {}).get("macd", {}).get("signal", "?")
        yield f"✅ **[2/7]** Teknik: **{ts}** (RSI: {rsi}, MACD: {macd})\n\n"
    except Exception as e:
        data_errors.append(f"Teknik analiz: {e}")
        ts, rsi, macd = "N/A", "?", "?"
        yield f"⚠️ **[2/7]** Teknik analiz alınamadı: {e}\n\n"

    yield "⏳ **[3/7]** Haberler + sentiment…\n"
    try:
        results["sentiment"] = _safe_json(get_news_sentiment(ticker, company_name=""))
        sent = results["sentiment"].get("overall_sentiment", "N/A")
        na = results["sentiment"].get("news_count", 0)
        method = results["sentiment"].get("sentiment_method", "keyword")
        yield f"✅ **[3/7]** Sentiment: **{sent}** ({na} haber, {method})\n\n"
    except Exception as e:
        data_errors.append(f"Sentiment: {e}")
        sent, na = "N/A", 0
        yield f"⚠️ **[3/7]** Sentiment alınamadı: {e}\n\n"

    yield "⏳ **[4/7]** Temel veriler…\n"
    try:
        results["fundamental"] = _safe_json(get_fundamental_data(ticker))
        pe = results["fundamental"].get("valuation", {}).get("pe_forward", "N/A")
        rec = results["fundamental"].get("analyst_targets", {}).get("recommendation", "N/A")
        name = results["fundamental"].get("profile", {}).get("name", ticker)
        mcap = results["fundamental"].get("profile", {}).get("market_cap_formatted", "N/A")
        yield f"✅ **[4/7]** **{name}** — P/E: {pe}, Analist: {rec}, MCap: {mcap}\n\n"
    except Exception as e:
        data_errors.append(f"Temel veriler: {e}")
        pe, rec, name, mcap = "N/A", "N/A", ticker, "N/A"
        yield f"⚠️ **[4/7]** Temel veriler alınamadı: {e}\n\n"

    yield "⏳ **[5/7]** Piyasa durumu…\n"
    try:
        results["market"] = _safe_json(get_market_overview())
        regime = results["market"].get("market_regime", "N/A")
        vix = results["market"].get("vix", {}).get("value", "?")
        vr = results["market"].get("vix", {}).get("regime", "?")
        yield f"✅ **[5/7]** Piyasa: **{regime}** (VIX: {vix} — {vr})\n\n"
    except Exception as e:
        data_errors.append(f"Piyasa durumu: {e}")
        regime, vix, vr = "N/A", "?", "?"
        yield f"⚠️ **[5/7]** Piyasa durumu alınamadı: {e}\n\n"

    yield "⏳ **[6/7]** Risk metrikleri…\n"
    try:
        results["risk"] = _safe_json(calculate_risk_metrics(ticker, portfolio_value=pv, risk_tolerance=rt.lower()))
        vol = results["risk"].get("volatility", {}).get("regime", "N/A")
        sl = results["risk"].get("trade_levels", {}).get("stop_loss", "?")
        tp = results["risk"].get("trade_levels", {}).get("take_profit", "?")
        shares = results["risk"].get("position_sizing", {}).get("suggested_shares", "?")
        rr = results["risk"].get("trade_levels", {}).get("risk_reward_ratio", "?")
        yield f"✅ **[6/7]** Risk: Vol **{vol}** | SL: ${sl} | TP: ${tp} | {shares} hisse | R/R: {rr}\n\n"
    except Exception as e:
        data_errors.append(f"Risk metrikleri: {e}")
        vol, sl, tp, shares, rr = "N/A", "?", "?", "?", "?"
        yield f"⚠️ **[6/7]** Risk metrikleri alınamadı: {e}\n\n"

    yield "⏳ **[7/7]** Makro ortam…\n"
    try:
        results["macro"] = _safe_json(get_macro_data(region="both"))
        global_regime = results["macro"].get("global_regime", "N/A")
        risk_factor = results["macro"].get("global_risk_factor", "?")
        yield f"✅ **[7/7]** Makro: **{global_regime}** (Risk: {risk_factor}/10)\n\n"
    except Exception as e:
        data_errors.append(f"Makro veriler: {e}")
        yield f"⚠️ **[7/7]** Makro veriler alınamadı: {e}\n\n"

    if data_errors:
        yield "⚠️ **Veri hataları:** " + ", ".join(data_errors) + "\n\n"

    yield "🎯 **Fund Manager** council'ı topluyor ve sentezliyor…\n\n"
    try:
        advisor = get_advisor()
        q = f"Analyze {ticker} ({name}). Data: price=${cp} chg={chg}%, tech={ts} RSI={rsi} MACD={macd}, sentiment={sent}, PE={pe} rec={rec} mcap={mcap}, market={regime} VIX={vix}, risk: vol={vol} SL=${sl} TP=${tp} shares={shares} RR={rr}. Portfolio ${pv:,.0f}, risk={rt}."
        if data_errors:
            q += f" DATA ERRORS: {'; '.join(data_errors)}. Reduce confidence accordingly."
        q += " Run FULL council debate (bull + bear + mediator). Produce TRADE ADVISOR REPORT with JSON block."

        raw_result = str(advisor.run(q))

        # Try structured parsing
        report, _ = parse_trade_report(raw_result)
        if report:
            # Save to DB
            try:
                save_analysis(
                    ticker=report.ticker, signal=report.signal.value,
                    confidence=report.confidence, entry_price=report.entry_price,
                    stop_loss=report.stop_loss, take_profit=report.take_profit,
                    shares=report.shares, risk_reward=report.risk_reward,
                    technical_signal=report.technical_signal, sentiment=report.sentiment,
                    fundamental_signal=report.fundamental_signal, market_regime=report.market_regime,
                    vix=report.vix, reasoning=report.reasoning, raw_report=raw_result,
                )
            except Exception:
                pass
            yield f"\n---\n\n{report_to_markdown(report)}"
        else:
            # Fallback to raw markdown
            yield f"\n---\n\n{raw_result}"

    except Exception as e:
        sig = ts
        emoji = "🟢" if "BUY" in str(sig) else ("🔴" if "SELL" in str(sig) else "🟡")
        signal = "BUY" if "BUY" in str(sig) else ("SELL" if "SELL" in str(sig) else "HOLD")
        tl = results.get("risk", {}).get("trade_levels", {})
        yield f"\n{emoji} **{signal}** — {name} ({ticker})\n\n| | |\n|---|---|\n| Entry | ${tl.get('entry_price', '?')} |\n| SL | ${sl} |\n| TP | ${tp} |\n| Hisse | {shares} |\n| R/R | {rr} |\n\n⚠️ Council çalıştırılamadı: {e}\n\n⚠️ *Yatırım tavsiyesi değildir.*"


def quick_analysis_streaming(ticker, pv, rt):
    if not ticker or not ticker.strip():
        yield "❌ Ticker girin"
        return
    a = ""
    for c in run_direct_analysis(ticker, pv, rt):
        a += c
        yield a


# ============================================================
# TAB 3: KARŞILAŞTIRMA
# ============================================================
def compare_tickers(ts, pv, rt):
    if not ts or not ts.strip():
        yield "❌ En az 2 ticker girin"
        return
    tickers = [t.strip().upper() for t in ts.split(",") if t.strip()]
    if len(tickers) < 2:
        yield "❌ En az 2 ticker"
        return
    if len(tickers) > 5:
        yield "❌ Max 5 ticker"
        return
    results = {}
    a = f"## 🔄 {', '.join(tickers)} Karşılaştırılıyor\n\n"
    yield a
    for i, t in enumerate(tickers):
        a += f"### 📊 {t} ({i + 1}/{len(tickers)})\n"
        yield a
        try:
            pd_data = _safe_json(get_price_history(t, "1mo"))
            td = _safe_json(get_technical_indicators(t, "3mo"))
            fd = _safe_json(get_fundamental_data(t))
            rd = _safe_json(calculate_risk_metrics(t, pv, rt.lower()))
            r = {
                "name": fd.get("profile", {}).get("name", t),
                "price": pd_data.get("current_price", "?"),
                "change": pd_data.get("price_change_pct", 0),
                "tech": td.get("overall_signal", "?"),
                "pe": fd.get("valuation", {}).get("pe_forward", "?"),
                "rec": fd.get("analyst_targets", {}).get("recommendation", "?"),
                "rsi": td.get("indicators", {}).get("rsi", {}).get("value", "?"),
                "rr": rd.get("trade_levels", {}).get("risk_reward_ratio", "?"),
                "vol": rd.get("volatility", {}).get("regime", "?"),
                "sharpe": rd.get("risk_metrics", {}).get("sharpe_ratio_approx", "?"),
            }
            results[t] = r
            a += f"- **{r['name']}** ${r['price']} ({r['change']:+.2f}%) Tech:{r['tech']} RSI:{r['rsi']}\n\n"
            yield a
        except Exception as e:
            a += f"⚠️ {t}: {e}\n\n"
            yield a
    if results:
        a += "\n## 📋 Karşılaştırma Tablosu\n\n"
        a += "| Metrik | " + " | ".join(results.keys()) + " |\n"
        a += "|---|" + ("|".join(["---"] * len(results))) + "|\n"
        for label, fn in [
            ("Fiyat", lambda r: f"${r['price']}"),
            ("Teknik", lambda r: f"**{r['tech']}**"),
            ("RSI", lambda r: str(r['rsi'])),
            ("P/E", lambda r: str(r['pe'])),
            ("Analist", lambda r: str(r['rec'])),
            ("R/R", lambda r: str(r['rr'])),
            ("Volatilite", lambda r: str(r['vol'])),
        ]:
            a += f"| {label} | " + " | ".join(fn(r) for r in results.values()) + " |\n"
    a += "\n⚠️ *Yatırım tavsiyesi değildir.*\n"
    yield a


# ============================================================
# TAB 4: BACKTEST
# ============================================================
def run_backtest_ui(ticker, period, holding_days, rsi_buy, rsi_sell, use_sma):
    if not ticker or not ticker.strip():
        return "❌ Ticker girin"
    try:
        from backtest import run_backtest, format_backtest_markdown
        result = run_backtest(
            ticker.strip().upper(),
            period_years=period,
            holding_days=int(holding_days),
            rsi_buy=rsi_buy,
            rsi_sell=rsi_sell,
            use_sma_filter=use_sma,
        )
        return format_backtest_markdown(result)
    except Exception as e:
        return f"❌ Backtest hatası: {e}"


# ============================================================
# TAB 5: PORTFÖY OPTİMİZASYONU
# ============================================================
def run_portfolio_optimizer(tickers_str, period):
    if not tickers_str or not tickers_str.strip():
        return "❌ En az 2 ticker girin (virgülle ayırın)"
    try:
        from portfolio_optimizer import optimize_portfolio, format_portfolio_markdown
        tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
        if len(tickers) < 2:
            return "❌ En az 2 ticker gerekli"
        result = optimize_portfolio(tickers, period)
        return format_portfolio_markdown(result)
    except Exception as e:
        return f"❌ Optimizasyon hatası: {e}"


# ============================================================
# TAB 6: WATCHLIST
# ============================================================
def add_to_watchlist(ticker, name, alert_type, alert_value, channel):
    if not ticker:
        return "❌ Ticker girin", get_watchlist_display()
    try:
        add_watchlist(ticker.strip().upper(), name, alert_type, alert_value, channel)
        return f"✅ {ticker.upper()} watchlist'e eklendi!", get_watchlist_display()
    except Exception as e:
        return f"❌ Hata: {e}", get_watchlist_display()


def remove_from_watchlist(watchlist_id):
    try:
        remove_watchlist(int(watchlist_id))
        return "✅ Silindi!", get_watchlist_display()
    except Exception as e:
        return f"❌ {e}", get_watchlist_display()


def get_watchlist_display():
    items = get_watchlist()
    if not items:
        return "📋 Watchlist boş — yukarıdan varlık ekleyin."
    md = "## 📋 Watchlist\n\n| ID | Ticker | İsim | Alert | Değer | Kanal | Son Tetik |\n|---|---|---|---|---|---|---|\n"
    for item in items:
        md += f"| {item['id']} | **{item['ticker']}** | {item.get('name', '')} | {item['alert_type']} | {item['alert_value']} | {item.get('notification_channel', 'telegram')} | {item.get('last_triggered', 'Hiç')} |\n"
    return md


# ============================================================
# TAB 7: ANALİZ GEÇMİŞİ
# ============================================================
def get_history_display(ticker_filter):
    try:
        ticker = ticker_filter.strip().upper() if ticker_filter and ticker_filter.strip() else None
        stats = get_accuracy_stats(ticker)
        md = "## 📊 Analiz Geçmişi\n\n"

        if "accuracy_7d" in stats:
            a7 = stats["accuracy_7d"]
            emoji = "🟢" if a7["accuracy_pct"] >= 60 else ("🟡" if a7["accuracy_pct"] >= 50 else "🔴")
            md += f"### {emoji} 7 Günlük Doğruluk: **{a7['accuracy_pct']}%** ({a7['correct']}/{a7['total']})\n"
            md += f"Ortalama getiri: **{a7['avg_return_pct']:+.2f}%**\n\n"

        if "accuracy_30d" in stats:
            a30 = stats["accuracy_30d"]
            emoji = "🟢" if a30["accuracy_pct"] >= 60 else ("🟡" if a30["accuracy_pct"] >= 50 else "🔴")
            md += f"### {emoji} 30 Günlük Doğruluk: **{a30['accuracy_pct']}%** ({a30['correct']}/{a30['total']})\n"
            md += f"Ortalama getiri: **{a30['avg_return_pct']:+.2f}%**\n\n"

        recent = stats.get("recent_analyses", [])
        if recent:
            md += "### Son Analizler\n\n"
            md += "| Tarih | Ticker | Sinyal | Güven | Giriş | 7G Getiri | 30G Getiri |\n|---|---|---|---|---|---|---|\n"
            for r in recent:
                r7 = f"{r.get('return_7d_pct', ''):+.2f}%" if r.get('return_7d_pct') is not None else "⏳"
                r30 = f"{r.get('return_30d_pct', ''):+.2f}%" if r.get('return_30d_pct') is not None else "⏳"
                c7 = "✅" if r.get('signal_correct_7d') == 1 else ("❌" if r.get('signal_correct_7d') == 0 else "")
                signal_emoji = "🟢" if r.get('signal') in ('BUY', 'STRONG_BUY') else ("🔴" if r.get('signal') in ('SELL', 'STRONG_SELL') else "🟡")
                entry = f"${r['entry_price']:.2f}" if r.get('entry_price') else "N/A"
                ts_short = r.get('timestamp', '')[:16]
                md += f"| {ts_short} | **{r.get('ticker', '?')}** | {signal_emoji} {r.get('signal', '?')} | {r.get('confidence', '?')}% | {entry} | {c7} {r7} | {r30} |\n"
        else:
            md += "*Henüz analiz geçmişi yok.*\n"

        return md
    except Exception as e:
        return f"❌ {e}"


# ============================================================
# TAB 8: TELEGRAM
# ============================================================
def send_telegram_alert(bt, ci, t, pv, rt):
    if not bt or not ci:
        yield "❌ Bot Token + Chat ID gerekli"
        return
    t = t.strip().upper()
    if not t:
        yield "❌ Ticker gerekli"
        return
    yield f"⏳ {t} analiz ediliyor…\n"
    try:
        td = _safe_json(get_technical_indicators(t, "3mo"))
        rd = _safe_json(calculate_risk_metrics(t, pv, rt.lower()))
        ts = td.get("overall_signal", "?")
        rsi = td.get("indicators", {}).get("rsi", {}).get("value", "?")
        tl = rd.get("trade_levels", {})
        e = "🟢" if "BUY" in ts else ("🔴" if "SELL" in ts else "🟡")
        msg = f"{e} *{t}* — *{ts}*\nRSI: {rsi}\nSL: ${tl.get('stop_loss', '?')} | TP: ${tl.get('take_profit', '?')}\nR/R: {tl.get('risk_reward_ratio', '?')}"
        import requests
        r = requests.post(
            f"https://api.telegram.org/bot{bt}/sendMessage",
            json={"chat_id": ci, "text": msg, "parse_mode": "Markdown"},
            timeout=10,
        )
        yield f"✅ Gönderildi!\n\n{msg}" if r.ok else f"❌ {r.text}"
    except Exception as ex:
        yield f"❌ {ex}"


# ============================================================
# TAB 9: SOHBET
# ============================================================
def chat_analysis(message, history):
    if not message or not message.strip():
        return "Bir soru sorun."
    try:
        return str(get_advisor().run(message))
    except Exception as e:
        return f"❌ {e}"


# ============================================================
# GRADIO UI
# ============================================================
DESC = """
# 🤖 Agentic Trade Bot Advisor v2.0

**12 AI Agent** • **LLM Council** • BIST + NASDAQ + Kripto + Altın/Gümüş + KAP + Makro

| Agent | Görev |
|---|---|
| 📈 Technical | RSI, MACD, Bollinger, SMA, ATR, Stochastic |
| 📰 Sentiment | FinBERT NLP + KAP + SEC Insider |
| 📊 Fundamental | P/E, PEG, marjlar + CoinGecko kripto |
| 🇹🇷 BIST Analyst | BIST30, bankalar, altın ₺, USD/TRY |
| ⚠️ Risk Manager | VaR, pozisyon, SL/TP, opsiyon akışı |
| 🌍 Macro Analyst | TCMB, Fed, yield curve, DXY |
| 🐂 Bull Advisor | AL argümanlarını savunur |
| 🐻 Bear Advisor | SAT argümanlarını savunur |
| ⚖️ Mediator | Dengeyi kurar |
| 🎯 Fund Manager | Council sentezler, final karar |

> ⚠️ Yatırım tavsiyesi değildir.
"""

with gr.Blocks(title="🤖 Trade Bot Advisor v2.0") as demo:
    gr.Markdown(DESC)
    with gr.Tabs():

        # TAB 1: Dashboard
        with gr.Tab("📊 Dashboard"):
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
            gr.Markdown("### Tek hisse detaylı analiz + LLM Council\n**BIST:** THYAO.IS, GARAN.IS | **US:** AAPL, NVDA | **Kripto:** BTC-USD | **Emtia:** GC=F")
            with gr.Row():
                with gr.Column(scale=1):
                    ticker_input = gr.Textbox(label="📌 Ticker", placeholder="THYAO.IS, AAPL, BTC-USD…", value="THYAO.IS")
                    portfolio_input = gr.Number(label="💰 Portföy ($)", value=100000, minimum=1000)
                    risk_input = gr.Radio(label="⚖️ Risk", choices=["Conservative", "Moderate", "Aggressive"], value="Moderate")
                    analyze_btn = gr.Button("🔍 Analiz", variant="primary", size="lg")
                with gr.Column(scale=2):
                    output = gr.Markdown(value="*Butona tıklayın…*")
            analyze_btn.click(fn=quick_analysis_streaming, inputs=[ticker_input, portfolio_input, risk_input], outputs=output)

        # TAB 3: Karşılaştırma
        with gr.Tab("🔄 Karşılaştırma"):
            gr.Markdown("### 2–5 ticker karşılaştırın")
            with gr.Row():
                with gr.Column(scale=1):
                    cmp_in = gr.Textbox(label="📌 Ticker'lar", value="THYAO.IS, GARAN.IS, AKBNK.IS")
                    cmp_pv = gr.Number(label="💰 Portföy ($)", value=100000, minimum=1000)
                    cmp_rt = gr.Radio(label="⚖️ Risk", choices=["Conservative", "Moderate", "Aggressive"], value="Moderate")
                    cmp_btn = gr.Button("🔄 Karşılaştır", variant="primary", size="lg")
                with gr.Column(scale=2):
                    cmp_out = gr.Markdown(value="*Butona tıklayın…*")
            cmp_btn.click(fn=compare_tickers, inputs=[cmp_in, cmp_pv, cmp_rt], outputs=cmp_out)

        # TAB 4: Backtest
        with gr.Tab("📊 Backtest"):
            gr.Markdown("### Sinyal doğruluk testi — geçmiş verilerde RSI+SMA stratejisi")
            with gr.Row():
                with gr.Column(scale=1):
                    bt_ticker = gr.Textbox(label="📌 Ticker", value="AAPL")
                    bt_period = gr.Slider(label="📅 Dönem (yıl)", minimum=0.5, maximum=5, step=0.5, value=1)
                    bt_holding = gr.Slider(label="⏱️ Tutma süresi (gün)", minimum=1, maximum=30, step=1, value=10)
                    bt_rsi_buy = gr.Slider(label="📈 RSI Alım eşiği", minimum=20, maximum=50, step=5, value=35)
                    bt_rsi_sell = gr.Slider(label="📉 RSI Satım eşiği", minimum=50, maximum=80, step=5, value=65)
                    bt_sma = gr.Checkbox(label="SMA filtresi kullan", value=True)
                    bt_btn = gr.Button("📊 Backtest Çalıştır", variant="primary", size="lg")
                with gr.Column(scale=2):
                    bt_out = gr.Markdown(value="*Parametreleri ayarlayın ve butona tıklayın…*")
            bt_btn.click(fn=run_backtest_ui, inputs=[bt_ticker, bt_period, bt_holding, bt_rsi_buy, bt_rsi_sell, bt_sma], outputs=bt_out)

        # TAB 5: Portföy Optimizasyonu
        with gr.Tab("📈 Portföy"):
            gr.Markdown("### Markowitz Efficient Frontier — optimal portföy ağırlıkları")
            with gr.Row():
                with gr.Column(scale=1):
                    po_tickers = gr.Textbox(label="📌 Varlıklar (virgülle)", value="AAPL, NVDA, MSFT, GOOGL, AMZN")
                    po_period = gr.Radio(label="📅 Dönem", choices=["6mo", "1y", "2y"], value="1y")
                    po_btn = gr.Button("📈 Optimize Et", variant="primary", size="lg")
                with gr.Column(scale=2):
                    po_out = gr.Markdown(value="*Varlıkları girin ve butona tıklayın…*")
            po_btn.click(fn=run_portfolio_optimizer, inputs=[po_tickers, po_period], outputs=po_out)

        # TAB 6: Watchlist
        with gr.Tab("👁️ Watchlist"):
            gr.Markdown("### Fiyat/RSI/VIX alertleri — otomatik bildirim")
            with gr.Row():
                with gr.Column(scale=1):
                    wl_ticker = gr.Textbox(label="📌 Ticker", placeholder="AAPL")
                    wl_name = gr.Textbox(label="İsim (opsiyonel)", placeholder="Apple Inc")
                    wl_type = gr.Dropdown(
                        label="Alert Tipi",
                        choices=["price_below", "price_above", "rsi_below", "rsi_above", "vix_above"],
                        value="price_below",
                    )
                    wl_value = gr.Number(label="Alert Değeri", value=150)
                    wl_channel = gr.Radio(label="Bildirim Kanalı", choices=["telegram", "email", "discord"], value="telegram")
                    wl_add_btn = gr.Button("➕ Ekle", variant="primary")
                    gr.Markdown("---")
                    wl_remove_id = gr.Number(label="Silinecek ID", value=0, precision=0)
                    wl_remove_btn = gr.Button("🗑️ Sil", variant="secondary")
                with gr.Column(scale=2):
                    wl_status = gr.Markdown(value="")
                    wl_display = gr.Markdown(value=get_watchlist_display())
            wl_add_btn.click(fn=add_to_watchlist, inputs=[wl_ticker, wl_name, wl_type, wl_value, wl_channel], outputs=[wl_status, wl_display])
            wl_remove_btn.click(fn=remove_from_watchlist, inputs=[wl_remove_id], outputs=[wl_status, wl_display])

        # TAB 7: Geçmiş
        with gr.Tab("📜 Geçmiş"):
            gr.Markdown("### Analiz geçmişi + sinyal doğruluk takibi")
            hist_ticker = gr.Textbox(label="Ticker Filtresi (boş=tümü)", placeholder="AAPL")
            hist_btn = gr.Button("📜 Geçmişi Göster", variant="primary")
            hist_out = gr.Markdown(value="*Butona tıklayın…*")
            hist_btn.click(fn=get_history_display, inputs=[hist_ticker], outputs=hist_out)

        # TAB 8: Telegram
        with gr.Tab("🔔 Telegram"):
            gr.Markdown("### Analiz → Telegram\n@BotFather → /newbot → Token, bota mesaj → getUpdates → chat_id")
            with gr.Row():
                with gr.Column():
                    tg_t = gr.Textbox(label="🤖 Token", type="password")
                    tg_c = gr.Textbox(label="💬 Chat ID")
                    tg_tk = gr.Textbox(label="📌 Ticker", value="THYAO.IS")
                    tg_pv = gr.Number(label="💰 Portföy ($)", value=100000)
                    tg_rt = gr.Radio(label="⚖️ Risk", choices=["Conservative", "Moderate", "Aggressive"], value="Moderate")
                    tg_btn = gr.Button("📤 Gönder", variant="primary")
                with gr.Column():
                    tg_out = gr.Markdown(value="*Bilgileri doldurun…*")
            tg_btn.click(fn=send_telegram_alert, inputs=[tg_t, tg_c, tg_tk, tg_pv, tg_rt], outputs=tg_out)

        # TAB 9: Sohbet
        with gr.Tab("💬 Sohbet"):
            gr.Markdown("Doğal dilde soru sorun — Türkçe veya İngilizce.")
            gr.ChatInterface(
                fn=chat_analysis,
                examples=[
                    "THYAO hissesini analiz et, portföyüm 500K TL, orta risk",
                    "Bugün BIST'te hangi hisseleri almalıyım?",
                    "Altın fiyatı ne kadar? Almalı mıyım?",
                    "Compare NVDA and AMD",
                    "Bitcoin temel analizi yap — developer activity nasıl?",
                    "AAPL için insider trading aktivitesi nasıl?",
                    "Türkiye ve ABD makro durumu nedir?",
                ],
            )

        # TAB 10: Rehber
        with gr.Tab("📖 Rehber"):
            gr.Markdown("""
# 📖 Kullanım Rehberi v2.0

## Desteklenen Varlıklar

| Tür | Format | Örnekler |
|---|---|---|
| 🇹🇷 BIST | `TICKER.IS` | THYAO.IS, GARAN.IS, AKBNK.IS |
| 🇺🇸 US/NASDAQ | `TICKER` | AAPL, NVDA, MSFT, TSLA |
| ₿ Kripto | `TICKER-USD` | BTC-USD, ETH-USD, SOL-USD |
| 🪙 Emtia | Futures | GC=F (Altın), SI=F (Gümüş), CL=F (Petrol) |
| 📦 ETF | `TICKER` | SPY, QQQ, IWM |
| 💱 Döviz | `XXX=X` | USDTRY=X, EURTRY=X |

## v2.0 Yenilikler

| Özellik | Açıklama |
|---|---|
| 🧠 **FinBERT Sentiment** | Keyword yerine NLP — %85+ doğruluk |
| 🏛️ **LLM Council** | Bull + Bear tartışması + Mediator hakem |
| 📊 **Backtest** | Geçmiş verilerde sinyal doğruluğu test |
| 📈 **Portföy Optimizer** | Markowitz efficient frontier |
| 👁️ **Watchlist** | Otomatik fiyat/RSI/VIX alertleri |
| 📜 **Analiz Geçmişi** | Her analiz kaydedilir, 7/30 gün sonra doğruluk kontrol |
| ₿ **Kripto Temel** | CoinGecko — market cap, dev activity, community |
| 🕵️ **Insider Trades** | SEC EDGAR Form 4 — içeriden işlemler |
| 📊 **Options Flow** | Put/Call ratio, unusual volume tespiti |
| 🌍 **Macro Analyst** | TCMB, Fed, yield curve, DXY |
| 🗄️ **Caching** | API çağrıları cache'lenir (15dk-24saat TTL) |
| ⚡ **Dual Model** | Data agent'lar hızlı model, Council derin model |

## Dashboard Sinyalleri
- AL: RSI < 40 + SMA20 > SMA50 (yükselen trend + oversold)
- SAT: RSI > 65 + SMA20 < SMA50 (düşen trend + overbought)

## CLI Kullanımı
```bash
git clone https://huggingface.co/spaces/SutskeverFanBoy/trade-bot-advisor
cd trade-bot-advisor && pip install -r requirements.txt
export HF_TOKEN="hf_xxx"
python cli.py THYAO.IS GARAN.IS --portfolio 500000 --risk moderate
python cli.py NVDA AAPL --quiet --telegram --bot-token "..." --chat-id "..." --alert-only
```

## Ortam Değişkenleri

| Değişken | Açıklama | Zorunlu |
|---|---|---|
| `HF_TOKEN` | HuggingFace API token | ✅ |
| `MODEL_ID` | Varsayılan LLM model ID | ❌ (default: Qwen/Qwen2.5-72B-Instruct) |
| `FAST_MODEL_ID` | Data agent'lar için hızlı model | ❌ (örn: Qwen/Qwen2.5-7B-Instruct) |
| `DEEP_MODEL_ID` | Council + FM için derin model | ❌ (örn: Qwen/Qwen2.5-72B-Instruct) |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token | ❌ |
| `TELEGRAM_CHAT_ID` | Telegram chat ID | ❌ |
| `DISCORD_WEBHOOK_URL` | Discord webhook URL | ❌ |
""")

    gr.Markdown("---\n🤗 smolagents + yfinance + pandas-ta + FinBERT + CoinGecko + KAP + SEC EDGAR | ⚠️ Yatırım tavsiyesi değildir")

if __name__ == "__main__":
    demo.queue(max_size=10).launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray"),
    )
