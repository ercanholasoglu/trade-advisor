"""Tool: BIST (Borsa İstanbul) hisse taraması + Türk piyasa genel durumu."""

from smolagents import tool


@tool
def get_bist_scanner(scan_type: str = "bist30") -> str:
    """
    Scans BIST (Borsa Istanbul) stocks and returns current prices, daily changes,
    technical signals (RSI), and market overview. Also provides gold, silver prices
    in TRY and USD/TRY exchange rate.

    Args:
        scan_type: Type of scan. Options: 'bist30' (top 30 stocks), 'bist_banks' (banking sector),
                   'bist_tech' (technology), 'commodities_try' (gold/silver/forex in TRY),
                   'indices' (BIST indices). Default 'bist30'.

    Returns:
        JSON string with scanned stocks, prices in TRY, daily changes, RSI signals,
        and sector overview.
    """
    import yfinance as yf
    import json

    try:
        TICKER_GROUPS = {
            "bist30": {
                "THYAO.IS": "THY", "GARAN.IS": "Garanti", "AKBNK.IS": "Akbank",
                "SISE.IS": "Şişecam", "EREGL.IS": "Ereğli", "BIMAS.IS": "BİM",
                "TUPRS.IS": "Tüpraş", "SAHOL.IS": "Sabancı", "KCHOL.IS": "Koç",
                "ASELS.IS": "Aselsan", "FROTO.IS": "Ford Otosan", "PGSUS.IS": "Pegasus",
                "TCELL.IS": "Turkcell", "TOASO.IS": "Tofaş", "PETKM.IS": "Petkim",
                "YKBNK.IS": "Yapı Kredi", "HALKB.IS": "Halkbank", "VAKBN.IS": "Vakıfbank",
                "EKGYO.IS": "Emlak GYO", "ENKAI.IS": "Enka", "TAVHL.IS": "TAV",
                "KOZAL.IS": "Koza Altın", "KOZAA.IS": "Koza Anadolu", "DOHOL.IS": "Doğan",
                "ARCLK.IS": "Arçelik", "TTKOM.IS": "Türk Telekom", "MGROS.IS": "Migros",
                "ISGYO.IS": "İş GYO", "SMRTG.IS": "Smart Güneş", "ODAS.IS": "Odaş",
            },
            "bist_banks": {
                "GARAN.IS": "Garanti", "AKBNK.IS": "Akbank", "YKBNK.IS": "Yapı Kredi",
                "HALKB.IS": "Halkbank", "VAKBN.IS": "Vakıfbank", "ISCTR.IS": "İş Bankası",
                "QNBFB.IS": "QNB Finansbank", "TSKB.IS": "TSKB", "ALBRK.IS": "Albaraka",
                "SKBNK.IS": "Şekerbank",
            },
            "bist_tech": {
                "ASELS.IS": "Aselsan", "LOGO.IS": "Logo Yazılım", "NETAS.IS": "Netaş",
                "ARENA.IS": "Arena", "DGATE.IS": "D-Gate", "INDES.IS": "İndes",
                "ESCOM.IS": "Escort", "KFEIN.IS": "Kafein",
            },
        }

        if scan_type == "commodities_try":
            return _get_commodities_try()
        if scan_type == "indices":
            return _get_bist_indices()

        tickers = TICKER_GROUPS.get(scan_type, TICKER_GROUPS["bist30"])
        stocks, buy_signals, sell_signals = [], [], []

        for ticker, name in tickers.items():
            try:
                hist = yf.Ticker(ticker).history(period="3mo", interval="1d")
                if hist.empty or len(hist) < 15: continue
                current = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2])
                change_pct = (current - prev) / prev * 100
                closes = hist["Close"].values[-15:]
                deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
                gains = [d for d in deltas if d > 0]
                losses = [-d for d in deltas if d < 0]
                avg_gain = sum(gains) / 14 if gains else 0
                avg_loss = sum(losses) / 14 if losses else 0.001
                rsi = 100 - (100 / (1 + avg_gain / avg_loss))
                rsi_signal = "OVERBOUGHT" if rsi > 70 else ("OVERSOLD" if rsi < 30 else ("BULLISH" if rsi > 60 else ("BEARISH" if rsi < 40 else "NEUTRAL")))
                avg_vol = float(hist["Volume"].iloc[-20:].mean()) if len(hist) >= 20 else 1
                vol_ratio = float(hist["Volume"].iloc[-1]) / avg_vol if avg_vol > 0 else 1.0
                entry = {"ticker": ticker.replace(".IS",""), "name": name, "price_try": round(current,2), "change_pct": round(change_pct,2), "rsi": round(rsi,1), "rsi_signal": rsi_signal, "volume_ratio": round(vol_ratio,2), "direction": "🟢" if change_pct >= 0 else "🔴"}
                stocks.append(entry)
                if rsi < 30: buy_signals.append(entry)
                elif rsi > 70: sell_signals.append(entry)
            except Exception: continue

        stocks.sort(key=lambda x: x["change_pct"], reverse=True)
        return json.dumps({"scan_type": scan_type, "total_stocks": len(stocks), "top_gainers": stocks[:5], "top_losers": stocks[-5:], "buy_signals_oversold": buy_signals, "sell_signals_overbought": sell_signals, "all_stocks": stocks}, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _get_commodities_try():
    import yfinance as yf
    import json
    TROY_OZ_TO_GRAM = 31.1035
    try:
        gold_usd = yf.Ticker("GC=F").history(period="2d")["Close"].iloc[-1]
        silver_usd = yf.Ticker("SI=F").history(period="2d")["Close"].iloc[-1]
        usd_try = yf.Ticker("USDTRY=X").history(period="2d")["Close"].iloc[-1]
        eur_try = yf.Ticker("EURTRY=X").history(period="2d")["Close"].iloc[-1]
        gold_try = (gold_usd * usd_try) / TROY_OZ_TO_GRAM
        silver_try = (silver_usd * usd_try) / TROY_OZ_TO_GRAM
        gold_prev = yf.Ticker("GC=F").history(period="5d")["Close"].iloc[-2]
        usd_prev = yf.Ticker("USDTRY=X").history(period="5d")["Close"].iloc[-2]
        gold_try_prev = (gold_prev * usd_prev) / TROY_OZ_TO_GRAM
        gold_chg = (gold_try - gold_try_prev) / gold_try_prev * 100
        return json.dumps({"scan_type": "commodities_try", "currency_rates": {"USD/TRY": round(float(usd_try),4), "EUR/TRY": round(float(eur_try),4)}, "gold": {"usd_per_oz": round(float(gold_usd),2), "try_per_gram": round(float(gold_try),2), "change_pct": round(float(gold_chg),2)}, "silver": {"usd_per_oz": round(float(silver_usd),2), "try_per_gram": round(float(silver_try),2)}}, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


def _get_bist_indices():
    import yfinance as yf
    import json
    indices = {"XU100.IS": "BIST 100", "XU030.IS": "BIST 30", "XBANK.IS": "BIST Banka", "XUTEK.IS": "BIST Teknoloji", "XUSIN.IS": "BIST Sınai", "XGIDA.IS": "BIST Gıda"}
    result_list = []
    for ticker, name in indices.items():
        try:
            h = yf.Ticker(ticker).history(period="5d")
            if h.empty or len(h) < 2: continue
            cur, prev = float(h["Close"].iloc[-1]), float(h["Close"].iloc[-2])
            chg = (cur - prev) / prev * 100
            result_list.append({"index": name, "ticker": ticker.replace(".IS",""), "value": round(cur,2), "change_pct": round(chg,2), "direction": "🟢" if chg >= 0 else "🔴"})
        except Exception: continue
    return json.dumps({"scan_type": "indices", "indices": result_list}, indent=2, ensure_ascii=False)
