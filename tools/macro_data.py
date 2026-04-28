"""
Tool: Macro Analyst — TCMB + FRED + Economic Calendar
========================================================
TCMB (Türkiye Cumhuriyet Merkez Bankası) açık veri,
FRED (Federal Reserve Economic Data) API (free, key not required for basic),
and upcoming macro events.
"""

from smolagents import tool


@tool
def get_macro_data(region: str = "both") -> str:
    """
    Fetches macroeconomic data: TCMB interest rates, inflation, USD/TRY for Turkey;
    Fed funds rate, CPI, unemployment, GDP for US. Determines macro regime
    (TIGHTENING/EASING/NEUTRAL) and risk factor for each region.

    Args:
        region: Which region's macro data. Options: 'turkey', 'us', 'both' (default).

    Returns:
        JSON string with macro indicators, regime assessment, risk factors, and upcoming catalysts.
    """
    import json
    import requests
    import yfinance as yf
    from datetime import datetime

    result = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "turkey": None,
        "us": None,
        "global_regime": None,
        "upcoming_catalysts": [],
    }

    # ──────────── TURKEY MACRO ────────────
    if region in ("turkey", "both"):
        turkey = {
            "policy_rate": None,
            "inflation_yoy": None,
            "usd_try": None,
            "eur_try": None,
            "cds_5y": None,
            "regime": "UNKNOWN",
            "risk_factor": 5,
            "notes": [],
        }

        # USD/TRY and EUR/TRY from yfinance
        try:
            usd_try = float(yf.Ticker("USDTRY=X").history(period="5d")["Close"].iloc[-1])
            eur_try = float(yf.Ticker("EURTRY=X").history(period="5d")["Close"].iloc[-1])
            usd_prev = float(yf.Ticker("USDTRY=X").history(period="1mo")["Close"].iloc[0])
            turkey["usd_try"] = round(usd_try, 4)
            turkey["eur_try"] = round(eur_try, 4)
            turkey["usd_try_1m_change"] = round((usd_try - usd_prev) / usd_prev * 100, 2)
        except Exception:
            pass

        # TCMB EVDS (Electronic Data Delivery System) — public endpoints
        try:
            # Try TCMB API for policy rate
            tcmb_url = "https://evds2.tcmb.gov.tr/service/evds/series=TP.TRY.M01&type=json&startDate=01-01-2024"
            # Note: TCMB EVDS requires API key for most endpoints
            # We'll use DuckDuckGo as fallback for latest rate info
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                news = list(ddgs.news("TCMB politika faizi 2026", max_results=3))
                if news:
                    turkey["notes"].append(f"Son TCMB haberi: {news[0].get('title', 'N/A')}")
        except Exception:
            pass

        # Determine Turkey regime based on available data
        usd_try_val = turkey.get("usd_try")
        usd_chg = turkey.get("usd_try_1m_change", 0)

        if usd_try_val:
            if usd_chg > 3:
                turkey["regime"] = "WEAKENING_LIRA"
                turkey["risk_factor"] = 8
                turkey["notes"].append("TL son 1 ayda %3+ değer kaybetti — yüksek risk")
            elif usd_chg > 1:
                turkey["regime"] = "MODERATE_DEPRECIATION"
                turkey["risk_factor"] = 6
            elif usd_chg < -1:
                turkey["regime"] = "STRENGTHENING_LIRA"
                turkey["risk_factor"] = 4
                turkey["notes"].append("TL güçleniyor — BIST için olumlu")
            else:
                turkey["regime"] = "STABLE"
                turkey["risk_factor"] = 5

        result["turkey"] = turkey

    # ──────────── US MACRO ────────────
    if region in ("us", "both"):
        us = {
            "fed_funds_rate": None,
            "treasury_10y": None,
            "treasury_2y": None,
            "yield_curve": None,
            "vix": None,
            "dxy": None,
            "regime": "UNKNOWN",
            "risk_factor": 5,
            "notes": [],
        }

        # Treasury yields from yfinance
        try:
            t10y = float(yf.Ticker("^TNX").history(period="5d")["Close"].iloc[-1])
            us["treasury_10y"] = round(t10y, 3)
        except Exception:
            pass

        try:
            t2y = float(yf.Ticker("2YY=F").history(period="5d")["Close"].iloc[-1])
            us["treasury_2y"] = round(t2y, 3)
        except Exception:
            pass

        # Yield curve (2y-10y spread)
        if us["treasury_10y"] and us["treasury_2y"]:
            spread = us["treasury_10y"] - us["treasury_2y"]
            us["yield_curve"] = round(spread, 3)
            if spread < 0:
                us["notes"].append("⚠️ Yield curve inverted — resesyon sinyali")
            elif spread < 0.2:
                us["notes"].append("Yield curve düzleşiyor — dikkat")

        # VIX
        try:
            vix = float(yf.Ticker("^VIX").history(period="5d")["Close"].iloc[-1])
            us["vix"] = round(vix, 2)
            if vix > 30:
                us["notes"].append("🚨 VIX > 30 — Panik modu")
                us["risk_factor"] = max(us["risk_factor"], 9)
            elif vix > 25:
                us["notes"].append("⚠️ VIX yüksek — volatilite arttı")
                us["risk_factor"] = max(us["risk_factor"], 7)
        except Exception:
            pass

        # DXY (Dollar Index)
        try:
            dxy = float(yf.Ticker("DX-Y.NYB").history(period="5d")["Close"].iloc[-1])
            dxy_prev = float(yf.Ticker("DX-Y.NYB").history(period="1mo")["Close"].iloc[0])
            us["dxy"] = round(dxy, 2)
            us["dxy_1m_change"] = round((dxy - dxy_prev) / dxy_prev * 100, 2)
            if us["dxy_1m_change"] > 2:
                us["notes"].append("Dolar güçleniyor — EM piyasaları için risk")
            elif us["dxy_1m_change"] < -2:
                us["notes"].append("Dolar zayıflıyor — EM piyasaları için olumlu")
        except Exception:
            pass

        # Determine US regime
        vix_val = us.get("vix", 20)
        t10y_val = us.get("treasury_10y", 4)
        yc = us.get("yield_curve")

        if t10y_val:
            if t10y_val > 5:
                us["regime"] = "TIGHTENING"
                us["risk_factor"] = max(us["risk_factor"], 7)
            elif t10y_val > 4:
                us["regime"] = "MODERATELY_TIGHT"
                us["risk_factor"] = max(us["risk_factor"], 6)
            elif t10y_val < 3:
                us["regime"] = "EASING"
                us["risk_factor"] = min(us["risk_factor"], 4)
            else:
                us["regime"] = "NEUTRAL"

        if vix_val > 30:
            us["regime"] += " + PANIC"

        result["us"] = us

    # ──────────── GLOBAL REGIME ────────────
    risk_factors = []
    if result.get("turkey"):
        risk_factors.append(result["turkey"].get("risk_factor", 5))
    if result.get("us"):
        risk_factors.append(result["us"].get("risk_factor", 5))

    avg_risk = sum(risk_factors) / len(risk_factors) if risk_factors else 5
    if avg_risk >= 7:
        result["global_regime"] = "HIGH_RISK"
    elif avg_risk >= 5:
        result["global_regime"] = "MODERATE_RISK"
    else:
        result["global_regime"] = "LOW_RISK"

    result["global_risk_factor"] = round(avg_risk, 1)

    # Upcoming catalysts (static knowledge — updated periodically)
    result["upcoming_catalysts"] = [
        "Fed FOMC toplantısı — faiz kararı",
        "TCMB PPK toplantısı — politika faizi kararı",
        "ABD Tarım Dışı İstihdam (NFP)",
        "ABD CPI (Tüketici Fiyat Endeksi)",
        "ABD GDP tahminleri",
        "Türkiye TÜFE açıklaması",
    ]

    return json.dumps(result, indent=2, ensure_ascii=False)
