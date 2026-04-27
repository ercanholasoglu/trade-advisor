"""Tool: KAP (Kamuoyu Aydınlatma Platformu) entegrasyonu."""

from smolagents import tool


@tool
def get_kap_disclosures(ticker: str) -> str:
    """
    Searches KAP (Kamuoyu Aydınlatma Platformu - Turkish Public Disclosure Platform)
    for company information and recent news/disclosures related to a BIST stock.
    Also fetches recent financial news about the company from web search.

    Args:
        ticker: BIST stock ticker (e.g. 'THYAO', 'GARAN', 'AKBNK'). No .IS suffix needed.

    Returns:
        JSON string with company info from KAP, recent disclosures/news, and links.
    """
    import json
    import requests

    ticker = ticker.strip().upper().replace(".IS", "")
    result = {"ticker": ticker, "kap_company": None, "kap_url": None, "recent_news": []}

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0) Chrome/124.0", "Content-Type": "application/json", "Accept": "application/json", "Accept-Language": "tr", "Origin": "https://www.kap.org.tr", "Referer": "https://www.kap.org.tr/tr/bildirim-sorgu"}

    try:
        resp = requests.post("https://www.kap.org.tr/tr/api/search/combined", headers=headers, json={"keyword": ticker, "lang": "tr"}, timeout=15)
        if resp.ok:
            for cat in resp.json():
                if cat.get("category") == "companyOrFunds":
                    for r in cat.get("results", []):
                        code = r.get("cmpOrFundCode", "")
                        if ticker.lower() == code or ticker.lower() in code:
                            result["kap_company"] = {"name": r.get("searchValue",""), "code": code, "oid": r.get("memberOrFundOid","")}
                            result["kap_url"] = f"https://www.kap.org.tr/tr/sirket-bilgileri/ozet/{code}"
                            result["kap_disclosures_url"] = f"https://www.kap.org.tr/tr/sirket-bildirimleri/{code}"
                            break
    except Exception as e:
        result["kap_search_error"] = str(e)

    try:
        from duckduckgo_search import DDGS
        company_name = result["kap_company"]["name"] if result["kap_company"] else ""
        with DDGS() as ddgs:
            for n in ddgs.news(f"{ticker} {company_name} KAP bildirim hisse borsa".strip(), max_results=6):
                result["recent_news"].append({"title": n.get("title",""), "source": n.get("source",""), "date": n.get("date",""), "url": n.get("url",""), "snippet": (n.get("body","") or "")[:200]})
    except Exception as e:
        result["news_error"] = str(e)

    if result["kap_company"]:
        result["summary"] = f"{result['kap_company']['name']} ({ticker}) — KAP: {result.get('kap_url','N/A')}. Bildirimler: {result.get('kap_disclosures_url','N/A')}. {len(result['recent_news'])} güncel haber."
    else:
        result["summary"] = f"{ticker} KAP'ta bulunamadı. {len(result['recent_news'])} haber bulundu."

    return json.dumps(result, indent=2, ensure_ascii=False)
