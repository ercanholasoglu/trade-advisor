"""Tool: Temel analiz - finansal tablolar, oranlar, earnings."""

from smolagents import tool
from tools.cache import cache, TTL_FUNDAMENTAL



@cache(ttl=TTL_FUNDAMENTAL)
def _cached_get_fundamental_data(ticker: str) -> str:
    import yfinance as yf
    import json

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or "symbol" not in info:
            return json.dumps({"error": f"No fundamental data for {ticker}", "ticker": ticker})

        profile = {
            "name": info.get("longName", "N/A"), "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"), "market_cap": info.get("marketCap", "N/A"),
            "employees": info.get("fullTimeEmployees", "N/A"), "country": info.get("country", "N/A"),
            "description": (info.get("longBusinessSummary", "N/A") or "N/A")[:300] + "...",
        }
        mc = profile["market_cap"]
        if isinstance(mc, (int, float)):
            if mc >= 1e12: profile["market_cap_formatted"] = f"${mc/1e12:.2f}T"
            elif mc >= 1e9: profile["market_cap_formatted"] = f"${mc/1e9:.2f}B"
            elif mc >= 1e6: profile["market_cap_formatted"] = f"${mc/1e6:.2f}M"

        valuation = {
            "current_price": info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
            "pe_trailing": info.get("trailingPE", "N/A"), "pe_forward": info.get("forwardPE", "N/A"),
            "peg_ratio": info.get("pegRatio", "N/A"), "price_to_book": info.get("priceToBook", "N/A"),
            "price_to_sales": info.get("priceToSalesTrailing12Months", "N/A"),
            "ev_to_ebitda": info.get("enterpriseToEbitda", "N/A"),
            "ev_to_revenue": info.get("enterpriseToRevenue", "N/A"),
        }

        financials = {
            "revenue": info.get("totalRevenue", "N/A"), "revenue_growth": info.get("revenueGrowth", "N/A"),
            "gross_margin": info.get("grossMargins", "N/A"), "operating_margin": info.get("operatingMargins", "N/A"),
            "profit_margin": info.get("profitMargins", "N/A"), "roe": info.get("returnOnEquity", "N/A"),
            "roa": info.get("returnOnAssets", "N/A"), "debt_to_equity": info.get("debtToEquity", "N/A"),
            "current_ratio": info.get("currentRatio", "N/A"), "free_cash_flow": info.get("freeCashflow", "N/A"),
            "earnings_growth": info.get("earningsGrowth", "N/A"),
        }
        for k in ["revenue", "free_cash_flow"]:
            v = financials[k]
            if isinstance(v, (int, float)):
                if abs(v) >= 1e9: financials[f"{k}_formatted"] = f"${v/1e9:.2f}B"
                elif abs(v) >= 1e6: financials[f"{k}_formatted"] = f"${v/1e6:.2f}M"
        for k in ["revenue_growth", "gross_margin", "operating_margin", "profit_margin", "roe", "roa", "earnings_growth"]:
            v = financials[k]
            if isinstance(v, (int, float)): financials[k] = f"{v*100:.1f}%"

        dividends = {
            "dividend_yield": info.get("dividendYield", "N/A"),
            "dividend_rate": info.get("dividendRate", "N/A"),
            "payout_ratio": info.get("payoutRatio", "N/A"),
        }
        if isinstance(dividends["dividend_yield"], (int, float)):
            dividends["dividend_yield"] = f"{dividends['dividend_yield']*100:.2f}%"

        target = {
            "target_mean": info.get("targetMeanPrice", "N/A"), "target_high": info.get("targetHighPrice", "N/A"),
            "target_low": info.get("targetLowPrice", "N/A"), "recommendation": info.get("recommendationKey", "N/A"),
            "number_of_analysts": info.get("numberOfAnalystOpinions", "N/A"),
        }
        current = valuation["current_price"]
        target_mean = target["target_mean"]
        if isinstance(current, (int, float)) and isinstance(target_mean, (int, float)) and current > 0:
            target["upside_pct"] = f"{((target_mean - current) / current) * 100:+.1f}%"

        signals = []
        pe = valuation["pe_forward"]
        if isinstance(pe, (int, float)):
            if pe < 15: signals.append("LOW_PE_UNDERVALUED")
            elif pe > 30: signals.append("HIGH_PE_OVERVALUED")
        peg = valuation["peg_ratio"]
        if isinstance(peg, (int, float)):
            if peg < 1: signals.append("PEG_UNDERVALUED")
            elif peg > 2: signals.append("PEG_OVERVALUED")
        rec = target["recommendation"]
        if rec in ["buy", "strongBuy", "strong_buy"]: signals.append("ANALYSTS_BULLISH")
        elif rec in ["sell", "strongSell", "strong_sell"]: signals.append("ANALYSTS_BEARISH")

        result = {
            "ticker": ticker.upper(), "profile": profile, "valuation": valuation,
            "financials": financials, "dividends": dividends, "analyst_targets": target,
            "fundamental_signals": signals,
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@tool
def get_fundamental_data(ticker: str) -> str:
    """
    Fetches fundamental financial data for a stock: valuation ratios, financials,
    earnings, dividends, and company profile. Essential for evaluating intrinsic value.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL', 'MSFT', 'GOOGL'). Crypto tickers not supported.

    Returns:
        JSON string with company profile, valuation metrics (P/E, P/B, etc.),
        financial highlights, and analyst recommendations.
    """
    return _cached_get_fundamental_data(ticker)
