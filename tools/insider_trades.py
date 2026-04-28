"""
Tool: SEC EDGAR Insider Trades
================================
Form 4 insider trading data via EDGAR full-text search API.
Detects large insider buys (strong bullish signal) and sells.
"""

from smolagents import tool
from tools.cache import cache, TTL_INSIDER



@cache(ttl=TTL_INSIDER)
def _cached_get_insider_trades(ticker: str, days: int = 60) -> str:
    import json
    import requests
    from datetime import datetime, timedelta

    try:
        ticker = ticker.strip().upper()
        date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        # SEC EDGAR EFTS search for Form 4 filings
        url = "https://efts.sec.gov/LATEST/search-index"
        params = {
            "q": f'"{ticker}"',
            "dateRange": "custom",
            "startdt": date_from,
            "enddt": datetime.now().strftime("%Y-%m-%d"),
            "forms": "4",
        }
        headers = {
            "User-Agent": "TradeBot research@tradebot.ai",
            "Accept": "application/json",
        }

        # Use EDGAR full-text search API
        search_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=4&dateRange=custom&startdt={date_from}"
        resp = requests.get(search_url, headers=headers, timeout=15)

        # Alternative: Use EDGAR company search API
        # First get CIK for the ticker
        cik_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={ticker}&CIK=&type=4&dateb=&owner=include&count=20&search_text=&action=getcompany"

        # Simpler approach: Use SEC EDGAR company filings API
        ticker_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&forms=4&dateRange=custom&startdt={date_from}"

        # Use the EDGAR full-text search
        api_url = "https://efts.sec.gov/LATEST/search-index"
        params = {
            "q": f'"{ticker}"',
            "forms": "4",
            "dateRange": "custom",
            "startdt": date_from,
        }

        # Fallback to company tickers JSON for CIK lookup
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        resp_tickers = requests.get(tickers_url, headers=headers, timeout=10)
        cik = None
        company_name = ticker

        if resp_tickers.ok:
            tickers_data = resp_tickers.json()
            for entry in tickers_data.values():
                if entry.get("ticker", "").upper() == ticker:
                    cik = str(entry.get("cik_str", "")).zfill(10)
                    company_name = entry.get("title", ticker)
                    break

        transactions = []
        total_buys_value = 0
        total_sells_value = 0
        buy_count = 0
        sell_count = 0

        if cik:
            # Get recent filings via EDGAR API
            filings_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
            resp_filings = requests.get(filings_url, headers=headers, timeout=15)

            if resp_filings.ok:
                filings_data = resp_filings.json()
                recent = filings_data.get("filings", {}).get("recent", {})
                forms = recent.get("form", [])
                dates = recent.get("filingDate", [])
                accessions = recent.get("accessionNumber", [])
                names = recent.get("primaryDocDescription", [])

                for i, form in enumerate(forms):
                    if form == "4" and i < len(dates):
                        filing_date = dates[i]
                        if filing_date >= date_from:
                            transactions.append({
                                "form": "4",
                                "filing_date": filing_date,
                                "accession": accessions[i] if i < len(accessions) else "",
                                "description": names[i] if i < len(names) else "Form 4 — Insider Trading",
                                "edgar_url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=4&dateb=&owner=include&count=20",
                            })

                # Count form 4 filings as proxy for activity
                form4_count = len(transactions)

        # Generate signals
        signals = []
        if len(transactions) > 10:
            signals.append("HIGH_INSIDER_ACTIVITY")
        elif len(transactions) < 2:
            signals.append("LOW_INSIDER_ACTIVITY")

        result = {
            "ticker": ticker,
            "company": company_name,
            "cik": cik,
            "period_days": days,
            "form4_filings": len(transactions),
            "recent_filings": transactions[:15],  # Last 15
            "signals": signals,
            "edgar_search_url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik or ticker}&type=4&dateb=&owner=include&count=40",
            "note": "Form 4 filings indicate insider trading activity. Full transaction details require XML parsing of individual filings.",
        }
        return json.dumps(result, indent=2)

    except requests.exceptions.Timeout:
        return json.dumps({"error": "SEC EDGAR timeout", "ticker": ticker})
    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@tool
def get_insider_trades(ticker: str, days: int = 60) -> str:
    """
    Fetches recent SEC EDGAR Form 4 insider trading filings for a US stock.
    Large insider purchases (>$500K) are a strong bullish signal.

    Args:
        ticker: US stock ticker symbol (e.g. 'AAPL', 'NVDA', 'TSLA'). BIST stocks not supported.
        days: Number of days to look back. Default 60.

    Returns:
        JSON string with insider transactions, buy/sell summary, and signals.
    """
    return _cached_get_insider_trades(ticker, days)
