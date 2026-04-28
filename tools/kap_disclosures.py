"""Tool: KAP (Kamuoyu Aydınlatma Platformu) entegrasyonu — v2."""

from smolagents import tool


@tool
def get_kap_disclosures(ticker: str) -> str:
    """
    Searches KAP (Kamuoyu Aydınlatma Platformu - Turkish Public Disclosure Platform)
    for company information, recent disclosures, and financial reports.
    Also fetches recent financial news about the company from web search.
    Uses memberDisclosureQuery API for categorized disclosures with importance scoring.

    Args:
        ticker: BIST stock ticker (e.g. 'THYAO', 'GARAN', 'AKBNK'). No .IS suffix needed.

    Returns:
        JSON string with company info from KAP, categorized disclosures, importance scores, and news.
    """
    import json
    import requests

    ticker = ticker.strip().upper().replace(".IS", "")
    result = {
        "ticker": ticker,
        "kap_company": None,
        "kap_url": None,
        "disclosures": [],
        "disclosure_summary": {},
        "recent_news": [],
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0) Chrome/124.0",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Language": "tr",
        "Origin": "https://www.kap.org.tr",
        "Referer": "https://www.kap.org.tr/tr/bildirim-sorgu",
    }

    # 1. Search for company via KAP combined search
    member_oid = None
    try:
        resp = requests.post(
            "https://www.kap.org.tr/tr/api/search/combined",
            headers=headers,
            json={"keyword": ticker, "lang": "tr"},
            timeout=15,
        )
        if resp.ok:
            for cat in resp.json():
                if cat.get("category") == "companyOrFunds":
                    for r in cat.get("results", []):
                        code = r.get("cmpOrFundCode", "")
                        if ticker.lower() == code or ticker.lower() in code:
                            member_oid = r.get("memberOrFundOid", "")
                            result["kap_company"] = {
                                "name": r.get("searchValue", ""),
                                "code": code,
                                "oid": member_oid,
                            }
                            result["kap_url"] = f"https://www.kap.org.tr/tr/sirket-bilgileri/ozet/{code}"
                            result["kap_disclosures_url"] = f"https://www.kap.org.tr/tr/sirket-bildirimleri/{code}"
                            break
    except Exception as e:
        result["kap_search_error"] = str(e)

    # 2. Try memberDisclosureQuery API for categorized disclosures
    if member_oid:
        try:
            # Try the disclosure query endpoint
            disc_url = f"https://www.kap.org.tr/tr/api/memberDisclosureQuery"
            disc_resp = requests.post(
                disc_url,
                headers=headers,
                json={
                    "memberOid": member_oid,
                    "disclosureType": "ALL",
                    "fromDate": None,
                    "toDate": None,
                },
                timeout=15,
            )
            if disc_resp.ok:
                disc_data = disc_resp.json()
                if isinstance(disc_data, list):
                    for d in disc_data[:20]:  # Last 20 disclosures
                        disc_type = d.get("disclosureType", "DIGER")
                        importance = _get_disclosure_importance(disc_type)
                        result["disclosures"].append({
                            "title": d.get("title", ""),
                            "type": disc_type,
                            "date": d.get("publishDate", ""),
                            "importance": importance,
                            "category": _categorize_disclosure(disc_type),
                        })
        except Exception:
            # memberDisclosureQuery might be blocked — continue with other data
            pass

    # 3. Try company info endpoint
    if member_oid:
        try:
            info_url = f"https://www.kap.org.tr/tr/api/member/{member_oid}/info"
            info_resp = requests.get(info_url, headers=headers, timeout=10)
            if info_resp.ok:
                info_data = info_resp.json()
                if isinstance(info_data, dict):
                    result["kap_company"].update({
                        "city": info_data.get("city", ""),
                        "website": info_data.get("webAddress", ""),
                        "sector": info_data.get("sectorName", ""),
                        "market": info_data.get("marketName", ""),
                    })
        except Exception:
            pass

    # 4. Disclosure summary by category
    if result["disclosures"]:
        categories = {}
        for d in result["disclosures"]:
            cat = d["category"]
            if cat not in categories:
                categories[cat] = {"count": 0, "importance": d["importance"]}
            categories[cat]["count"] += 1
        result["disclosure_summary"] = categories

    # 5. Web search for recent news
    try:
        from duckduckgo_search import DDGS
        company_name = result["kap_company"]["name"] if result["kap_company"] else ""
        with DDGS() as ddgs:
            for n in ddgs.news(
                f"{ticker} {company_name} KAP bildirim hisse borsa".strip(),
                max_results=6,
            ):
                result["recent_news"].append({
                    "title": n.get("title", ""),
                    "source": n.get("source", ""),
                    "date": n.get("date", ""),
                    "url": n.get("url", ""),
                    "snippet": (n.get("body", "") or "")[:200],
                })
    except Exception as e:
        result["news_error"] = str(e)

    # Summary
    n_disc = len(result["disclosures"])
    n_news = len(result["recent_news"])
    high_importance = sum(1 for d in result["disclosures"] if d["importance"] >= 8)

    if result["kap_company"]:
        result["summary"] = (
            f"{result['kap_company']['name']} ({ticker}) — "
            f"KAP: {result.get('kap_url', 'N/A')}. "
            f"{n_disc} bildirim ({high_importance} yüksek önem), "
            f"{n_news} güncel haber."
        )
    else:
        result["summary"] = f"{ticker} KAP'ta bulunamadı. {n_news} haber bulundu."

    return json.dumps(result, indent=2, ensure_ascii=False)


def _get_disclosure_importance(disc_type: str) -> int:
    """
    Rate disclosure importance 1-10.
    Finansal sonuç > Genel kurul > Ortaklık yapısı > Diğer
    """
    importance_map = {
        "FR": 10,    # Finansal rapor / sonuç bildirimi
        "ODA": 9,    # Özel durum açıklaması
        "GK": 8,     # Genel kurul
        "OYD": 8,    # Ortaklık yapısı değişikliği
        "BD": 7,     # Bilgi/doküman
        "IHR": 7,    # İhraç/halka arz
        "DGR": 6,    # Derecelendirme
        "TEMP": 5,   # Temettü
        "SPK": 5,    # SPK kararları
        "KYD": 4,    # Kurumsal yönetim
    }
    return importance_map.get(disc_type.upper(), 3)


def _categorize_disclosure(disc_type: str) -> str:
    """Categorize disclosure type in Turkish."""
    categories = {
        "FR": "📊 Finansal Sonuç",
        "ODA": "⚡ Özel Durum",
        "GK": "🏛️ Genel Kurul",
        "OYD": "👥 Ortaklık Değişikliği",
        "BD": "📄 Bilgi/Doküman",
        "IHR": "💰 İhraç/Halka Arz",
        "DGR": "⭐ Derecelendirme",
        "TEMP": "💵 Temettü",
        "SPK": "⚖️ SPK Kararı",
        "KYD": "🏢 Kurumsal Yönetim",
    }
    return categories.get(disc_type.upper(), "📋 Diğer")
