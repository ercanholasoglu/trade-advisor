"""
Tool: BIST Full Universe Scanner + Sektör Grupları + Yeni Enstrümanlar
========================================================================
BIST 30 → BIST genişletilmiş evren. 7 sektör grubu.
Yeni: ONS/TRY, XTMTU.IS, Eurobond proxy, VİOP vadeli işlemler.
"""

from smolagents import tool
from tools.cache import cache, TTL_BIST

# ═══════════════════════════════════════════════════════════════
# BIST FULL UNIVERSE — sektör bazlı gruplama
# ═══════════════════════════════════════════════════════════════
BIST_SECTORS = {
    "bist_banks": {
        "GARAN.IS": "Garanti BBVA", "AKBNK.IS": "Akbank", "YKBNK.IS": "Yapı Kredi",
        "HALKB.IS": "Halkbank", "VAKBN.IS": "Vakıfbank", "ISCTR.IS": "İş Bankası",
        "QNBFB.IS": "QNB Finansbank", "TSKB.IS": "TSKB", "ALBRK.IS": "Albaraka",
        "SKBNK.IS": "Şekerbank", "KLNMA.IS": "Kalkınma Bankası",
    },
    "bist_defense": {
        "ASELS.IS": "Aselsan", "TUSAS.IS": "TUSAŞ", "KATMR.IS": "Katmerciler",
        "OTOKAR.IS": "Otokar", "VESTL.IS": "Vestel Savunma",
    },
    "bist_energy": {
        "TUPRS.IS": "Tüpraş", "PETKM.IS": "Petkim", "AYGAZ.IS": "Aygaz",
        "AKSEN.IS": "Aksa Enerji", "ODAS.IS": "Odaş Enerji", "AYDEM.IS": "Aydem Enerji",
        "ENKAI.IS": "Enka İnşaat", "KONTR.IS": "Kontrolmatik", "ZOREN.IS": "Zorlu Enerji",
        "GESAN.IS": "Giresun Enerji",
    },
    "bist_reit": {
        "EKGYO.IS": "Emlak Konut GYO", "ISGYO.IS": "İş GYO", "HLGYO.IS": "Halk GYO",
        "VKGYO.IS": "Vakıf GYO", "TRGYO.IS": "Torunlar GYO", "MRGYO.IS": "Marmara GYO",
        "DGGYO.IS": "Doğuş GYO", "RYGYO.IS": "Ray Sigorta GYO",
    },
    "bist_tech": {
        "ASELS.IS": "Aselsan", "LOGO.IS": "Logo Yazılım", "NETAS.IS": "Netaş",
        "ARENA.IS": "Arena Bilgisayar", "DGATE.IS": "D-Gate", "INDES.IS": "İndeks Bilgisayar",
        "ESCOM.IS": "Escort Teknoloji", "KFEIN.IS": "Kafein Yazılım",
        "PAPIL.IS": "Papilon Savunma", "ARDYZ.IS": "Ard Yapı",
        "SMART.IS": "SmartIks", "MIATK.IS": "MIA Teknoloji",
    },
    "bist_retail": {
        "BIMAS.IS": "BİM", "MGROS.IS": "Migros", "SOKM.IS": "Şok Market",
        "CRFSA.IS": "CarrefourSA", "BIZIM.IS": "Bizim Toptan", "ADESE.IS": "Adese",
        "MAVI.IS": "Mavi Giyim", "DESA.IS": "Desa Deri", "VAKKO.IS": "Vakko",
        "BOYP.IS": "Boyner Perakende",
    },
    "bist_food": {
        "ULKER.IS": "Ülker", "BANVT.IS": "Banvit", "TATGD.IS": "Tat Gıda",
        "KERVT.IS": "Kerevitaş", "ERSU.IS": "Ersu Gıda", "PENGD.IS": "Penguen Gıda",
        "PINSU.IS": "Pınar Su", "PNSUT.IS": "Pınar Süt", "AEFES.IS": "Anadolu Efes",
        "CCOLA.IS": "Coca-Cola İçecek", "TBORG.IS": "Türk Tuborg",
    },
}

# BIST 30 is a curated subset
BIST30 = {
    "THYAO.IS": "THY", "GARAN.IS": "Garanti", "AKBNK.IS": "Akbank",
    "SISE.IS": "Şişecam", "EREGL.IS": "Ereğli", "BIMAS.IS": "BİM",
    "TUPRS.IS": "Tüpraş", "SAHOL.IS": "Sabancı", "KCHOL.IS": "Koç Holding",
    "ASELS.IS": "Aselsan", "FROTO.IS": "Ford Otosan", "PGSUS.IS": "Pegasus",
    "TCELL.IS": "Turkcell", "TOASO.IS": "Tofaş", "PETKM.IS": "Petkim",
    "YKBNK.IS": "Yapı Kredi", "HALKB.IS": "Halkbank", "VAKBN.IS": "Vakıfbank",
    "EKGYO.IS": "Emlak GYO", "ENKAI.IS": "Enka", "TAVHL.IS": "TAV",
    "KOZAL.IS": "Koza Altın", "KOZAA.IS": "Koza Anadolu", "DOHOL.IS": "Doğan",
    "ARCLK.IS": "Arçelik", "TTKOM.IS": "Türk Telekom", "MGROS.IS": "Migros",
    "ISGYO.IS": "İş GYO", "SMRTG.IS": "Smart Güneş", "ODAS.IS": "Odaş",
}

# Full universe = BIST30 + all sector groups merged (deduplicated)
BIST_ALL = {}
BIST_ALL.update(BIST30)
for sector_tickers in BIST_SECTORS.values():
    BIST_ALL.update(sector_tickers)

# Extended BIST indices
BIST_INDICES = {
    "XU100.IS": "BIST 100", "XU030.IS": "BIST 30", "XBANK.IS": "BIST Banka",
    "XUTEK.IS": "BIST Teknoloji", "XUSIN.IS": "BIST Sınai", "XGIDA.IS": "BIST Gıda",
    "XSGRT.IS": "BIST Sigorta", "XELKT.IS": "BIST Elektrik",
    "XKMYA.IS": "BIST Kimya", "XMESY.IS": "BIST Metal Eşya",
    "XTMTU.IS": "BIST Temettü", "XILTM.IS": "BIST İletişim",
    "XUHIZ.IS": "BIST Hizmet", "XULAS.IS": "BIST Ulaştırma",
}

TROY_OZ_TO_GRAM = 31.1035


def _calc_rsi_simple(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d for d in deltas[-period:] if d > 0]
    losses = [-d for d in deltas[-period:] if d < 0]
    avg_gain = sum(gains) / period if gains else 0
    avg_loss = sum(losses) / period if losses else 0.001
    return round(100 - (100 / (1 + avg_gain / avg_loss)), 1)


def _scan_ticker_group(tickers_dict):
    """Scan a dict of {ticker: name} and return list of stock data."""
    import yfinance as yf
    stocks, buy_signals, sell_signals = [], [], []
    for ticker, name in tickers_dict.items():
        try:
            hist = yf.Ticker(ticker).history(period="3mo", interval="1d")
            if hist.empty or len(hist) < 15:
                continue
            current = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2])
            change_pct = (current - prev) / prev * 100
            rsi = _calc_rsi_simple(hist["Close"].values[-15:])
            rsi_signal = ("OVERBOUGHT" if rsi > 70 else
                          ("OVERSOLD" if rsi < 30 else
                           ("BULLISH" if rsi > 60 else
                            ("BEARISH" if rsi < 40 else "NEUTRAL"))))
            avg_vol = float(hist["Volume"].iloc[-20:].mean()) if len(hist) >= 20 else 1
            vol_ratio = float(hist["Volume"].iloc[-1]) / avg_vol if avg_vol > 0 else 1.0
            entry = {
                "ticker": ticker.replace(".IS", ""), "name": name,
                "price_try": round(current, 2), "change_pct": round(change_pct, 2),
                "rsi": rsi, "rsi_signal": rsi_signal,
                "volume_ratio": round(vol_ratio, 2),
                "direction": "🟢" if change_pct >= 0 else "🔴",
            }
            stocks.append(entry)
            if rsi < 30:
                buy_signals.append(entry)
            elif rsi > 70:
                sell_signals.append(entry)
        except Exception:
            continue
    stocks.sort(key=lambda x: x["change_pct"], reverse=True)
    return stocks, buy_signals, sell_signals


@cache(ttl=TTL_BIST)
def _cached_get_bist_scanner(scan_type: str = "bist30") -> str:
    import json

    # Route to sub-functions
    if scan_type == "commodities_try":
        return _get_commodities_try()
    if scan_type == "indices":
        return _get_bist_indices()
    if scan_type == "viop":
        return _get_viop_futures()
    if scan_type == "bonds":
        return _get_bonds_eurobond()

    # Pick the right ticker group
    if scan_type == "bist_all":
        tickers = BIST_ALL
    elif scan_type in BIST_SECTORS:
        tickers = BIST_SECTORS[scan_type]
    elif scan_type == "bist30":
        tickers = BIST30
    else:
        tickers = BIST30

    stocks, buy_signals, sell_signals = _scan_ticker_group(tickers)

    return json.dumps({
        "scan_type": scan_type,
        "total_stocks": len(stocks),
        "top_gainers": stocks[:5],
        "top_losers": stocks[-5:],
        "buy_signals_oversold": buy_signals,
        "sell_signals_overbought": sell_signals,
        "all_stocks": stocks,
    }, indent=2, ensure_ascii=False)


@tool
def get_bist_scanner(scan_type: str = "bist30") -> str:
    """
    Scans BIST (Borsa Istanbul) stocks by sector and returns prices, changes, RSI signals.
    Also provides gold/silver in TRY, BIST indices, VİOP futures, and bond proxies.

    Args:
        scan_type: Type of scan. Options:
            'bist30' — BIST 30 blue chips
            'bist_all' — Full BIST universe (~80 stocks across all sectors)
            'bist_banks' — Banking sector (11 stocks)
            'bist_defense' — Defense/aerospace (5 stocks)
            'bist_energy' — Energy sector (10 stocks)
            'bist_reit' — REITs / GYO (8 stocks)
            'bist_tech' — Technology (12 stocks)
            'bist_retail' — Retail/perakende (10 stocks)
            'bist_food' — Food/gıda (11 stocks)
            'commodities_try' — Gold/silver/forex in TRY (ONS/TRY included)
            'indices' — BIST indices including XTMTU (Temettü)
            'viop' — VİOP futures proxies
            'bonds' — Eurobond / gov bond proxies
            Default 'bist30'.

    Returns:
        JSON string with scanned stocks, prices in TRY, daily changes, RSI signals, sector grouping.
    """
    return _cached_get_bist_scanner(scan_type)


def _get_commodities_try():
    """Gold/silver/platinum in TRY, ONS/TRY, forex rates."""
    import yfinance as yf
    import json

    result = {"scan_type": "commodities_try", "currency_rates": {}, "gold": {}, "silver": {}, "platinum": {}}
    try:
        usd_try = float(yf.Ticker("USDTRY=X").history(period="5d")["Close"].iloc[-1])
        eur_try = float(yf.Ticker("EURTRY=X").history(period="5d")["Close"].iloc[-1])
        usd_prev = float(yf.Ticker("USDTRY=X").history(period="5d")["Close"].iloc[-2])
        result["currency_rates"] = {
            "USD/TRY": round(usd_try, 4),
            "EUR/TRY": round(eur_try, 4),
            "USD/TRY_change_pct": round((usd_try - usd_prev) / usd_prev * 100, 2),
        }
    except Exception:
        usd_try = 0

    if usd_try > 0:
        # Gold
        try:
            gold_usd = float(yf.Ticker("GC=F").history(period="5d")["Close"].iloc[-1])
            gold_prev = float(yf.Ticker("GC=F").history(period="5d")["Close"].iloc[-2])
            gold_try_gram = (gold_usd * usd_try) / TROY_OZ_TO_GRAM
            gold_try_prev = (gold_prev * usd_prev) / TROY_OZ_TO_GRAM
            gold_chg = (gold_try_gram - gold_try_prev) / gold_try_prev * 100
            result["gold"] = {
                "usd_per_oz": round(gold_usd, 2),
                "try_per_gram": round(gold_try_gram, 2),
                "try_per_oz": round(gold_usd * usd_try, 2),  # ONS/TRY
                "change_pct": round(gold_chg, 2),
            }
        except Exception:
            pass

        # Silver
        try:
            silver_usd = float(yf.Ticker("SI=F").history(period="5d")["Close"].iloc[-1])
            silver_try_gram = (silver_usd * usd_try) / TROY_OZ_TO_GRAM
            result["silver"] = {
                "usd_per_oz": round(silver_usd, 2),
                "try_per_gram": round(silver_try_gram, 2),
                "try_per_oz": round(silver_usd * usd_try, 2),
            }
        except Exception:
            pass

        # Platinum
        try:
            plat_usd = float(yf.Ticker("PL=F").history(period="5d")["Close"].iloc[-1])
            result["platinum"] = {
                "usd_per_oz": round(plat_usd, 2),
                "try_per_gram": round((plat_usd * usd_try) / TROY_OZ_TO_GRAM, 2),
            }
        except Exception:
            pass

    return json.dumps(result, indent=2, ensure_ascii=False)


def _get_bist_indices():
    """Extended BIST indices including XTMTU (Temettü)."""
    import yfinance as yf
    import json

    result_list = []
    for ticker, name in BIST_INDICES.items():
        try:
            h = yf.Ticker(ticker).history(period="5d")
            if h.empty or len(h) < 2:
                continue
            cur = float(h["Close"].iloc[-1])
            prev = float(h["Close"].iloc[-2])
            chg = (cur - prev) / prev * 100
            result_list.append({
                "index": name, "ticker": ticker.replace(".IS", ""),
                "value": round(cur, 2), "change_pct": round(chg, 2),
                "direction": "🟢" if chg >= 0 else "🔴",
            })
        except Exception:
            continue
    return json.dumps({"scan_type": "indices", "indices": result_list}, indent=2, ensure_ascii=False)


def _get_viop_futures():
    """VİOP futures proxies via yfinance — BIST30 futures, USD/TRY futures."""
    import yfinance as yf
    import json

    # VİOP doesn't have direct yfinance tickers, but we can proxy:
    # - BIST30 futures ≈ XU030.IS (spot) + implied premium
    # - Gold futures ≈ GC=F vs ONS/TRY spot
    viop = {"scan_type": "viop", "note": "VİOP proxy data via spot indices", "contracts": []}

    proxies = {
        "XU030.IS": "VİOP BIST30 Vadeli (proxy)", "USDTRY=X": "VİOP Dolar Vadeli (proxy)",
        "GC=F": "VİOP Altın Vadeli (proxy)", "XU100.IS": "VİOP BIST100 Vadeli (proxy)",
    }
    for ticker, name in proxies.items():
        try:
            h = yf.Ticker(ticker).history(period="5d")
            if h.empty or len(h) < 2:
                continue
            cur = float(h["Close"].iloc[-1])
            prev = float(h["Close"].iloc[-2])
            chg = (cur - prev) / prev * 100
            viop["contracts"].append({
                "contract": name, "proxy_ticker": ticker,
                "value": round(cur, 2), "change_pct": round(chg, 2),
                "direction": "🟢" if chg >= 0 else "🔴",
            })
        except Exception:
            continue
    return json.dumps(viop, indent=2, ensure_ascii=False)


def _get_bonds_eurobond():
    """Turkey government bond / Eurobond proxies."""
    import yfinance as yf
    import json

    bonds = {"scan_type": "bonds", "instruments": []}
    # Turkey-related bond/CDS proxies
    bond_tickers = {
        "^TNX": "US 10Y Treasury",
        "TUR": "iShares MSCI Turkey ETF",
        "EURTRY=X": "EUR/TRY (Eurobond proxy)",
        "USDTRY=X": "USD/TRY",
    }
    for ticker, name in bond_tickers.items():
        try:
            h = yf.Ticker(ticker).history(period="5d")
            if h.empty or len(h) < 2:
                continue
            cur = float(h["Close"].iloc[-1])
            prev = float(h["Close"].iloc[-2])
            chg = (cur - prev) / prev * 100
            bonds["instruments"].append({
                "name": name, "ticker": ticker,
                "value": round(cur, 4 if "=" in ticker else 2),
                "change_pct": round(chg, 2),
                "direction": "🟢" if chg >= 0 else "🔴",
            })
        except Exception:
            continue
    bonds["note"] = "Direct Turkish gov bond/sukuk data requires TCMB EVDS API key. Showing proxy instruments."
    return json.dumps(bonds, indent=2, ensure_ascii=False)
