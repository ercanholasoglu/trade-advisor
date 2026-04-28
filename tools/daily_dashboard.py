"""Tool: Günlük Dashboard — Al/Sat önerileri, performans takibi, piyasa özeti."""

from smolagents import tool


@tool
def get_daily_dashboard(market: str = "both", portfolio_value: float = 100000.0) -> str:
    """
    Generates a comprehensive daily trading dashboard with buy/sell recommendations,
    market overview, top movers, and commodity prices. Covers both BIST and US markets.

    Args:
        market: Which market to scan. Options: 'bist' (Borsa Istanbul only), 'us' (US/NASDAQ only), 'both' (default).
        portfolio_value: Portfolio size in USD for position sizing context. Default 100000.

    Returns:
        JSON string with daily dashboard including buy signals, sell signals,
        top gainers/losers, market regime, commodities, and summary.
    """
    import yfinance as yf
    import json
    import datetime

    TROY_OZ_TO_GRAM = 31.1035
    dashboard = {"date": str(datetime.date.today()), "portfolio_value": portfolio_value, "markets_scanned": market, "buy_recommendations": [], "sell_recommendations": [], "top_gainers": [], "top_losers": [], "market_summary": {}, "commodities_try": {}, "alerts": []}

    def _calc_rsi(closes, period=14):
        if len(closes) < period + 1: return 50.0
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d for d in deltas[-period:] if d > 0]
        losses = [-d for d in deltas[-period:] if d < 0]
        avg_gain = sum(gains) / period if gains else 0
        avg_loss = sum(losses) / period if losses else 0.001
        return round(100 - (100 / (1 + avg_gain / avg_loss)), 1)

    def _scan_stocks(tickers_dict, suffix=""):
        results = []
        for ticker, name in tickers_dict.items():
            try:
                h = yf.Ticker(ticker + suffix).history(period="3mo", interval="1d")
                if h.empty or len(h) < 20: continue
                cur, prev = float(h["Close"].iloc[-1]), float(h["Close"].iloc[-2])
                chg = (cur - prev) / prev * 100
                rsi = _calc_rsi(h["Close"].values)
                sma20 = float(h["Close"].iloc[-20:].mean())
                sma50 = float(h["Close"].iloc[-50:].mean()) if len(h) >= 50 else sma20
                if rsi < 30 and cur > sma20: signal = "STRONG_BUY"
                elif rsi < 40 and sma20 > sma50: signal = "BUY"
                elif rsi > 70 and cur < sma20: signal = "STRONG_SELL"
                elif rsi > 65 and sma20 < sma50: signal = "SELL"
                else: signal = "HOLD"
                results.append({"ticker": ticker, "name": name, "price": round(cur,2), "change_pct": round(chg,2), "rsi": rsi, "sma20": round(sma20,2), "signal": signal, "direction": "🟢" if chg >= 0 else "🔴"})
            except Exception: continue
        return results

    bist_stocks, us_stocks = [], []
    if market in ("bist", "both"):
        BIST30 = {"THYAO":"THY","GARAN":"Garanti","AKBNK":"Akbank","SISE":"Şişecam","EREGL":"Ereğli","BIMAS":"BİM","TUPRS":"Tüpraş","SAHOL":"Sabancı","KCHOL":"Koç","ASELS":"Aselsan","FROTO":"Ford Otosan","PGSUS":"Pegasus","TCELL":"Turkcell","TOASO":"Tofaş","PETKM":"Petkim","YKBNK":"Yapı Kredi","HALKB":"Halkbank","VAKBN":"Vakıfbank","KOZAL":"Koza Altın","ARCLK":"Arçelik","TTKOM":"Türk Telekom","MGROS":"Migros","TAVHL":"TAV","ENKAI":"Enka"}
        bist_stocks = _scan_stocks(BIST30, suffix=".IS")
        try:
            xu = yf.Ticker("XU100.IS").history(period="5d")
            if not xu.empty and len(xu) >= 2:
                v, p = float(xu["Close"].iloc[-1]), float(xu["Close"].iloc[-2])
                dashboard["market_summary"]["bist100"] = {"value": round(v,2), "change_pct": round((v-p)/p*100,2)}
        except: pass

    if market in ("us", "both"):
        US = {"AAPL":"Apple","NVDA":"NVIDIA","MSFT":"Microsoft","GOOGL":"Alphabet","AMZN":"Amazon","META":"Meta","TSLA":"Tesla","AMD":"AMD","NFLX":"Netflix","CRM":"Salesforce","AVGO":"Broadcom","ORCL":"Oracle"}
        us_stocks = _scan_stocks(US, suffix="")
        try:
            sp = yf.Ticker("^GSPC").history(period="5d")
            if not sp.empty and len(sp) >= 2:
                v, p = float(sp["Close"].iloc[-1]), float(sp["Close"].iloc[-2])
                dashboard["market_summary"]["sp500"] = {"value": round(v,2), "change_pct": round((v-p)/p*100,2)}
        except: pass

    all_stocks = bist_stocks + us_stocks
    for s in all_stocks:
        if s["signal"] in ("STRONG_BUY","BUY"): dashboard["buy_recommendations"].append(s)
        elif s["signal"] in ("STRONG_SELL","SELL"): dashboard["sell_recommendations"].append(s)
    dashboard["buy_recommendations"].sort(key=lambda x: x["rsi"])
    dashboard["sell_recommendations"].sort(key=lambda x: x["rsi"], reverse=True)
    all_sorted = sorted(all_stocks, key=lambda x: x["change_pct"], reverse=True)
    dashboard["top_gainers"] = all_sorted[:5]
    dashboard["top_losers"] = all_sorted[-5:]

    try:
        gold = float(yf.Ticker("GC=F").history(period="2d")["Close"].iloc[-1])
        silver = float(yf.Ticker("SI=F").history(period="2d")["Close"].iloc[-1])
        usd = float(yf.Ticker("USDTRY=X").history(period="2d")["Close"].iloc[-1])
        dashboard["commodities_try"] = {"gold_try_gram": round(gold*usd/TROY_OZ_TO_GRAM,2), "silver_try_gram": round(silver*usd/TROY_OZ_TO_GRAM,2), "usd_try": round(usd,4), "gold_usd_oz": round(gold,2)}
    except: pass

    try:
        vix = float(yf.Ticker("^VIX").history(period="2d")["Close"].iloc[-1])
        dashboard["market_summary"]["vix"] = round(vix,2)
        if vix > 30: dashboard["alerts"].append("🚨 VIX > 30 — Panik modu!")
        elif vix > 25: dashboard["alerts"].append("⚠️ VIX yüksek")
    except: pass

    n_buy, n_sell = len(dashboard["buy_recommendations"]), len(dashboard["sell_recommendations"])
    dashboard["summary"] = f"Günlük Tarama: {len(all_stocks)} hisse. 📈 {n_buy} AL, 📉 {n_sell} SAT. Altın: ₺{dashboard['commodities_try'].get('gold_try_gram','?')}/gram, USD/TRY: {dashboard['commodities_try'].get('usd_try','?')}"
    return json.dumps(dashboard, indent=2, ensure_ascii=False)
