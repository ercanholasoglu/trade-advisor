"""Tool: Genel piyasa durumu - major endeksler, sektörler, VIX."""

from smolagents import tool
from tools.cache import cache, TTL_MARKET



@cache(ttl=TTL_MARKET)
def _cached_get_market_overview() -> str:
    import yfinance as yf
    import json

    try:
        indices = {"S&P 500": "^GSPC", "NASDAQ": "^IXIC", "DOW": "^DJI", "Russell 2000": "^RUT"}
        assets = {"Gold": "GC=F", "Crude Oil": "CL=F", "Bitcoin": "BTC-USD", "10Y Treasury": "^TNX", "US Dollar (DXY)": "DX-Y.NYB"}
        sectors = {"Technology": "XLK", "Healthcare": "XLV", "Financials": "XLF", "Energy": "XLE", "Consumer Disc.": "XLY", "Industrials": "XLI", "Utilities": "XLU", "Real Estate": "XLRE", "Communication": "XLC", "Materials": "XLB", "Consumer Staples": "XLP"}

        def get_change(symbol):
            try:
                t = yf.Ticker(symbol)
                h = t.history(period="5d")
                if h.empty or len(h) < 2: return None, None, None
                current = float(h["Close"].iloc[-1])
                prev = float(h["Close"].iloc[-2])
                change = current - prev
                change_pct = (change / prev) * 100
                return round(current, 2), round(change, 2), round(change_pct, 2)
            except Exception: return None, None, None

        index_data = {}
        for name, symbol in indices.items():
            price, change, change_pct = get_change(symbol)
            if price is not None:
                index_data[name] = {"price": price, "change": change, "change_pct": f"{change_pct:+.2f}%", "direction": "🟢" if change_pct >= 0 else "🔴"}

        vix_price, vix_change, vix_change_pct = get_change("^VIX")
        if vix_price is not None:
            if vix_price < 15: vix_regime = "LOW_FEAR (Complacency)"
            elif vix_price < 20: vix_regime = "NORMAL"
            elif vix_price < 30: vix_regime = "ELEVATED_FEAR"
            else: vix_regime = "HIGH_FEAR (Panic)"
        else: vix_regime = "N/A"

        asset_data = {}
        for name, symbol in assets.items():
            price, change, change_pct = get_change(symbol)
            if price is not None:
                asset_data[name] = {"price": price, "change_pct": f"{change_pct:+.2f}%"}

        sector_data = {}
        for name, symbol in sectors.items():
            price, change, change_pct = get_change(symbol)
            if price is not None:
                sector_data[name] = {"price": price, "change_pct": f"{change_pct:+.2f}%", "direction": "🟢" if change_pct >= 0 else "🔴"}

        sorted_sectors = dict(sorted(sector_data.items(), key=lambda x: float(x[1]["change_pct"].replace("%", "").replace("+", "")), reverse=True))

        positive_indices = sum(1 for d in index_data.values() if d.get("change_pct", "").startswith("+") or d.get("change_pct", "") == "+0.00%")
        total_indices = len(index_data)
        if positive_indices == total_indices: market_regime = "BULLISH"
        elif positive_indices >= total_indices * 0.75: market_regime = "MODERATELY_BULLISH"
        elif positive_indices <= total_indices * 0.25: market_regime = "BEARISH"
        elif positive_indices <= total_indices * 0.5: market_regime = "MODERATELY_BEARISH"
        else: market_regime = "MIXED"
        if vix_price and vix_price > 25: market_regime += " + HIGH_VOLATILITY"

        result = {
            "market_regime": market_regime,
            "vix": {"value": vix_price, "change_pct": f"{vix_change_pct:+.2f}%" if vix_change_pct else "N/A", "regime": vix_regime},
            "indices": index_data, "key_assets": asset_data, "sector_performance": sorted_sectors,
            "top_sectors": list(sorted_sectors.keys())[:3], "bottom_sectors": list(sorted_sectors.keys())[-3:],
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def get_market_overview() -> str:
    """
    Provides a comprehensive overview of current market conditions.
    Includes major indices (S&P 500, NASDAQ, DOW), VIX fear index, sector performance,
    key commodities (Gold, Oil), crypto (Bitcoin), and bond yields (10Y Treasury).

    Returns:
        JSON string with market indices, sector performance, commodities,
        crypto, bond yields, and overall market regime assessment.
    """
    return _cached_get_market_overview()
