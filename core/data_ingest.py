"""
📡 Data Ingestion — yfinance + Binance REST + sample fallback
================================================================
Unified data interface. Always returns valid data: live > fallback.
Space NEVER crashes — sample data guarantees functionality.
"""

import logging
import datetime
import numpy as np
import pandas as pd

logger = logging.getLogger("trade_bot.data")

# ─────── Sample Data for Demo Mode (ALWAYS available) ───────

def _generate_sample_ohlcv(ticker: str, days: int = 200) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data for demo/fallback."""
    np.random.seed(hash(ticker) % 2**31)
    
    base_prices = {
        "AAPL": 230.0, "NVDA": 135.0, "MSFT": 420.0, "GOOGL": 175.0,
        "TSLA": 285.0, "AMZN": 200.0, "META": 550.0,
        "THYAO.IS": 340.0, "GARAN.IS": 145.0, "AKBNK.IS": 72.0,
        "BTC-USD": 94500.0, "ETH-USD": 3200.0, "SOL-USD": 180.0,
        "GC=F": 3320.0, "SI=F": 30.0,
    }
    base = base_prices.get(ticker.upper(), 100.0)
    
    dates = pd.date_range(end=datetime.date.today(), periods=days, freq='B')
    
    # Random walk with slight upward drift
    returns = np.random.normal(0.0005, 0.018, days)
    prices = base * np.cumprod(1 + returns)
    
    # Generate OHLCV
    highs = prices * (1 + np.abs(np.random.normal(0, 0.01, days)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.01, days)))
    opens = lows + (highs - lows) * np.random.random(days)
    volumes = np.random.randint(1_000_000, 50_000_000, days).astype(float)
    
    df = pd.DataFrame({
        'Open': opens, 'High': highs, 'Low': lows,
        'Close': prices, 'Volume': volumes,
    }, index=dates)
    
    return df


def fetch_ohlcv(ticker: str, period: str = "3mo", interval: str = "1d") -> dict:
    """
    Unified OHLCV fetch. Tries live data first, falls back to sample data.
    NEVER returns None df — always has valid data for the Space to work.
    """
    ticker = ticker.strip().upper()
    
    # Try live data first
    if _is_crypto(ticker):
        result = _fetch_binance(ticker, period, interval)
        if result["df"] is not None and len(result["df"]) >= 26:
            return result
    
    result = _fetch_yfinance(ticker, period, interval)
    if result["df"] is not None and len(result["df"]) >= 26:
        return result
    
    # Fallback to sample data
    logger.warning(f"Live data unavailable for {ticker}, using sample data")
    period_days = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}.get(period, 90)
    df = _generate_sample_ohlcv(ticker, max(period_days, 200))
    return {"df": df, "source": "sample_data", "error": None, "note": "Demo/sample data — live data unavailable"}


def _is_crypto(ticker: str) -> bool:
    crypto_suffixes = ["-USD", "-USDT", "-BTC", "-EUR"]
    crypto_tickers = {"BTC", "ETH", "SOL", "ADA", "DOGE", "XRP", "DOT", "AVAX", "LINK", "BNB"}
    base = ticker.split("-")[0]
    return any(ticker.endswith(s) for s in crypto_suffixes) or base in crypto_tickers


def _fetch_binance(ticker: str, period: str, interval: str) -> dict:
    try:
        import requests
        binance_map = {
            "BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT", "SOL-USD": "SOLUSDT",
            "ADA-USD": "ADAUSDT", "DOGE-USD": "DOGEUSDT", "XRP-USD": "XRPUSDT",
            "DOT-USD": "DOTUSDT", "AVAX-USD": "AVAXUSDT", "LINK-USD": "LINKUSDT",
            "BNB-USD": "BNBUSDT",
        }
        symbol = binance_map.get(ticker, ticker.replace("-USD", "USDT").replace("-", ""))
        bi = {"1d": "1d", "1h": "1h", "4h": "4h", "1wk": "1w"}.get(interval, "1d")
        limit = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730}.get(period, 90)
        
        resp = requests.get("https://api.binance.com/api/v3/klines",
                          params={"symbol": symbol, "interval": bi, "limit": min(limit, 1000)}, timeout=10)
        if not resp.ok:
            return {"df": None, "source": "binance", "error": f"HTTP {resp.status_code}"}
        
        data = resp.json()
        if not data:
            return {"df": None, "source": "binance", "error": "Empty response"}
        
        rows = [{"Date": pd.Timestamp(k[0], unit="ms"), "Open": float(k[1]), "High": float(k[2]),
                 "Low": float(k[3]), "Close": float(k[4]), "Volume": float(k[5])} for k in data]
        df = pd.DataFrame(rows).set_index("Date")
        return {"df": df, "source": "binance", "error": None}
    except Exception as e:
        return {"df": None, "source": "binance", "error": str(e)}


def _fetch_yfinance(ticker: str, period: str, interval: str) -> dict:
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            return {"df": None, "source": "yfinance", "error": f"No data for {ticker}"}
        return {"df": df, "source": "yfinance", "error": None}
    except Exception as e:
        return {"df": None, "source": "yfinance", "error": str(e)}


# ─────── Fallback info for display ───────

TICKER_NAMES = {
    "AAPL": "Apple Inc", "NVDA": "NVIDIA Corp", "MSFT": "Microsoft",
    "GOOGL": "Alphabet", "TSLA": "Tesla", "AMZN": "Amazon", "META": "Meta",
    "THYAO.IS": "Türk Hava Yolları", "GARAN.IS": "Garanti Bankası",
    "AKBNK.IS": "Akbank", "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum",
    "SOL-USD": "Solana", "GC=F": "Gold Futures", "SI=F": "Silver Futures",
}


def get_ticker_name(ticker: str) -> str:
    return TICKER_NAMES.get(ticker.upper(), ticker.upper())
