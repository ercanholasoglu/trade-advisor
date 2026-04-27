#!/usr/bin/env python3
"""
🤖 Trade Bot Advisor — CLI (Komut Satırı Arayüzü)
====================================================
Terminal'den trade analizi çalıştır, opsiyonel Telegram alert gönder.

Kullanım:
  python cli.py NVDA                                  # Basit analiz
  python cli.py NVDA --portfolio 50000 --risk aggressive  # Özelleştirilmiş
  python cli.py NVDA AAPL BTC-USD                     # Çoklu ticker
  python cli.py NVDA --telegram --bot-token "..." --chat-id "..."  # Alert
  python cli.py NVDA --alert-only --telegram ...       # Sadece güçlü sinyallerde
"""

import argparse
import json
import os
import sys
import datetime

from tools.price_history import get_price_history
from tools.technical_indicators import get_technical_indicators
from tools.news_sentiment import get_news_sentiment
from tools.fundamental_analysis import get_fundamental_data
from tools.risk_calculator import calculate_risk_metrics
from tools.market_overview import get_market_overview


def _safe_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        return {"error": raw}


def analyze_ticker(ticker: str, portfolio: float, risk: str, verbose: bool = True):
    ticker = ticker.strip().upper()
    results = {}

    if verbose:
        print(f"\n{'='*60}")
        print(f"🤖 TRADE BOT ADVISOR — {ticker}")
        print(f"{'='*60}")
        print(f"📅 {datetime.date.today()} | 💰 ${portfolio:,.0f} | ⚖️ {risk}")
        print(f"{'─'*60}")

    if verbose: print("⏳ [1/6] Fiyat verisi çekiliyor…", end=" ", flush=True)
    results["price"] = _safe_json(get_price_history(ticker, period="1mo"))
    price = results["price"].get("current_price", "?")
    chg = results["price"].get("price_change_pct", 0)
    if verbose: print(f"✅ ${price} ({chg:+.2f}%)")

    if verbose: print("⏳ [2/6] Teknik indikatörler…", end=" ", flush=True)
    results["technical"] = _safe_json(get_technical_indicators(ticker, period="3mo"))
    tech = results["technical"].get("overall_signal", "N/A")
    rsi = results["technical"].get("indicators", {}).get("rsi", {}).get("value", "?")
    macd = results["technical"].get("indicators", {}).get("macd", {}).get("signal", "?")
    if verbose: print(f"✅ {tech} (RSI: {rsi}, MACD: {macd})")

    if verbose: print("⏳ [3/6] Haber sentiment…", end=" ", flush=True)
    results["sentiment"] = _safe_json(get_news_sentiment(ticker))
    sent = results["sentiment"].get("overall_sentiment", "N/A")
    n_news = results["sentiment"].get("news_count", 0)
    if verbose: print(f"✅ {sent} ({n_news} haber)")

    if verbose: print("⏳ [4/6] Temel veriler…", end=" ", flush=True)
    results["fundamental"] = _safe_json(get_fundamental_data(ticker))
    pe = results["fundamental"].get("valuation", {}).get("pe_forward", "N/A")
    rec = results["fundamental"].get("analyst_targets", {}).get("recommendation", "N/A")
    name = results["fundamental"].get("profile", {}).get("name", ticker)
    if verbose: print(f"✅ {name} — P/E: {pe}, Analist: {rec}")

    if verbose: print("⏳ [5/6] Piyasa durumu…", end=" ", flush=True)
    results["market"] = _safe_json(get_market_overview())
    regime = results["market"].get("market_regime", "N/A")
    vix = results["market"].get("vix", {}).get("value", "?")
    if verbose: print(f"✅ {regime} (VIX: {vix})")

    if verbose: print("⏳ [6/6] Risk metrikleri…", end=" ", flush=True)
    results["risk"] = _safe_json(calculate_risk_metrics(ticker, portfolio, risk.lower()))
    vol = results["risk"].get("volatility", {}).get("regime", "N/A")
    tl = results["risk"].get("trade_levels", {})
    ps = results["risk"].get("position_sizing", {})
    rr = tl.get("risk_reward_ratio", "?")
    if verbose: print(f"✅ Vol: {vol}, R/R: {rr}")

    if tech in ("STRONG_BUY", "BUY"):
        signal, emoji = "BUY", "🟢"
    elif tech in ("STRONG_SELL", "SELL"):
        signal, emoji = "SELL", "🔴"
    else:
        signal, emoji = "HOLD", "🟡"

    if verbose:
        print(f"\n{'─'*60}")
        print(f"{emoji} SİNYAL: {signal} ({tech})")
        print(f"{'─'*60}")
        print(f"📊 {ticker} ({name})")
        print(f"💰 Fiyat:      ${price}")
        print(f"📈 Teknik:     {tech} (RSI: {rsi}, MACD: {macd})")
        print(f"📰 Sentiment:  {sent} ({n_news} haber)")
        print(f"📊 Temel:      P/E: {pe}, Analist: {rec}")
        print(f"🌍 Piyasa:     {regime} (VIX: {vix})")
        print(f"⚡ Volatilite: {vol}")
        print(f"{'─'*60}")
        print(f"💰 Entry:      ${tl.get('entry_price', '?')}")
        print(f"🛑 Stop-Loss:  ${tl.get('stop_loss', '?')} ({tl.get('stop_loss_pct', '?')})")
        print(f"🎯 Take-Profit:${tl.get('take_profit', '?')} ({tl.get('take_profit_pct', '?')})")
        print(f"📦 Pozisyon:   {ps.get('suggested_shares', '?')} hisse (${ps.get('position_value_usd', '?')})")
        print(f"⚖️ Risk/Reward:{rr}")
        print(f"{'─'*60}")
        recs = results["risk"].get("recommendations", [])
        if recs:
            print("⚠️  UYARILAR:")
            for r in recs: print(f"   {r}")
            print()
        print("⚠️  Bu AI destekli analiz, yatırım tavsiyesi değildir.")
        print(f"{'='*60}\n")

    return {
        "ticker": ticker, "name": name, "signal": signal, "tech_signal": tech,
        "price": price, "rsi": rsi, "macd": macd, "sentiment": sent,
        "pe": pe, "rec": rec, "regime": regime, "vix": vix, "vol": vol,
        "sl": tl.get("stop_loss"), "tp": tl.get("take_profit"),
        "shares": ps.get("suggested_shares"), "rr": rr,
        "recommendations": results["risk"].get("recommendations", []),
    }


def send_telegram(bot_token: str, chat_id: str, result: dict):
    import requests
    r = result
    emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}[r["signal"]]
    msg = (
        f"{emoji} *Trade Bot — {r['ticker']}*\n\n"
        f"💰 Fiyat: ${r['price']}\n📈 Sinyal: *{r['tech_signal']}*\n"
        f"📊 RSI: {r['rsi']} | MACD: {r['macd']}\n📰 Sentiment: {r['sentiment']}\n"
        f"📊 P/E: {r['pe']} | Analist: {r['rec']}\n🌍 Piyasa: {r['regime']} (VIX: {r['vix']})\n\n"
        f"🛑 SL: ${r['sl']} | 🎯 TP: ${r['tp']}\n📦 {r['shares']} hisse | ⚖️ R/R: {r['rr']}\n\n"
        f"⚠️ _Yatırım tavsiyesi değildir._"
    )
    resp = requests.post(f"https://api.telegram.org/bot{bot_token}/sendMessage",
                         json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}, timeout=10)
    if resp.ok: print(f"✅ Telegram: {r['ticker']} → {r['signal']}")
    else: print(f"❌ Telegram hatası: {resp.text}")


def main():
    parser = argparse.ArgumentParser(description="🤖 Trade Bot Advisor — CLI",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tickers", nargs="+", help="Ticker: AAPL, NVDA, BTC-USD")
    parser.add_argument("--portfolio", type=float, default=100000)
    parser.add_argument("--risk", choices=["conservative", "moderate", "aggressive"], default="moderate")
    parser.add_argument("--telegram", action="store_true")
    parser.add_argument("--bot-token", default=os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    parser.add_argument("--chat-id", default=os.environ.get("TELEGRAM_CHAT_ID", ""))
    parser.add_argument("--alert-only", action="store_true", help="Sadece BUY/SELL'de Telegram gönder")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    all_results = []
    for ticker in args.tickers:
        result = analyze_ticker(ticker, args.portfolio, args.risk, verbose=not args.quiet and not args.json)
        if args.quiet:
            r = result
            emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}[r["signal"]]
            print(f"{emoji} {r['ticker']:8s} ${str(r['price']):>10s}  {r['tech_signal']:14s}  RSI:{str(r['rsi']):>6s}  SL:${r['sl']}  TP:${r['tp']}  R/R:{r['rr']}")
        all_results.append(result)
        if args.telegram:
            if args.alert_only and result["signal"] == "HOLD":
                if not args.quiet: print(f"ℹ️  {ticker}: HOLD — alert-only, Telegram yok.\n")
            else:
                send_telegram(args.bot_token, args.chat_id, result)

    if args.json: print(json.dumps(all_results, indent=2, default=str))
    if len(all_results) > 1 and not args.json:
        print(f"\n{'='*60}\n📋 KARŞILAŞTIRMA ÖZETİ\n{'='*60}")
        print(f"{'Ticker':<10} {'Sinyal':<14} {'Fiyat':>10} {'RSI':>6} {'R/R':>8} {'Sentiment':<10}")
        print(f"{'─'*60}")
        for r in all_results:
            emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}[r["signal"]]
            print(f"{emoji} {r['ticker']:<8} {r['tech_signal']:<14} ${str(r['price']):>8} {str(r['rsi']):>6} {str(r['rr']):>8} {r['sentiment']:<10}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
