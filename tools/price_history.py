"""Tool: Fiyat geçmişi ve OHLCV verileri çekme."""

from smolagents import tool


@tool
def get_price_history(ticker: str, period: str = "1mo", interval: str = "1d") -> str:
    """
    Fetches OHLCV (Open, High, Low, Close, Volume) price history for a given ticker symbol.
    Use this to get historical price data for stocks, ETFs, crypto, or indices.

    Args:
        ticker: Stock/crypto ticker symbol (e.g. 'AAPL', 'BTC-USD', 'GOOGL', 'NVDA')
        period: Time period to fetch. Options: '1d','5d','1mo','3mo','6mo','1y','2y','5y','max'. Default '1mo'.
        interval: Data interval/granularity. Options: '1m','5m','15m','1h','1d','1wk','1mo'. Default '1d'.

    Returns:
        JSON string with OHLCV data including dates, prices, volume, and basic statistics.
    """
    import yfinance as yf
    import json

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period, interval=interval)

        if hist.empty:
            return json.dumps({"error": f"No data found for {ticker}", "ticker": ticker})

        # Son 30 satırı al (çok uzun olmasın)
        hist = hist.tail(30)

        records = []
        for date, row in hist.iterrows():
            records.append({
                "date": str(date.strftime("%Y-%m-%d %H:%M")),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            })

        # Temel istatistikler
        closes = [r["close"] for r in records]
        current_price = closes[-1]
        price_change = current_price - closes[0]
        price_change_pct = (price_change / closes[0]) * 100
        high_price = max(r["high"] for r in records)
        low_price = min(r["low"] for r in records)
        avg_volume = sum(r["volume"] for r in records) // len(records)

        result = {
            "ticker": ticker.upper(),
            "period": period,
            "interval": interval,
            "current_price": current_price,
            "price_change": round(price_change, 2),
            "price_change_pct": round(price_change_pct, 2),
            "period_high": high_price,
            "period_low": low_price,
            "avg_volume": avg_volume,
            "data_points": len(records),
            "ohlcv": records,
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})
