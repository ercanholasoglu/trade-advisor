"""
📡 Data Ingestion — Binance REST + yfinance fallback
======================================================
Unified data interface. Tries Binance public API for crypto,
yfinance for everything else. No API keys needed.
Returns standardized OHLCV DataFrame.
"""

import json
import logging
import datetime
import numpy as np

logger = logging.getLogger("trade_bot.data")


def fetch_ohlcv(ticker: str, period: str = "3mo", interval: str = "1d") -> dict:
    """
    Unified OHLCV fetch. Routes crypto to Binance, everything else to yfinance.
    Returns dict with 'df' (DataFrame or None), 'source', 'error'.
    """
    ticker = ticker.strip().upper()

    # Route crypto to Binance
    if _is_crypto(ticker):
        result = _fetch_binance(ticker, period, interval)
        if result["df"] is not None:
            return result
        # Binance failed → fallback to yfinance
        logger.warning(f"Binance failed for {ticker}, falling back to yfinance")

    return _fetch_yfinance(ticker, period, interval)


def _is_crypto(ticker: str) -> bool:
    """Check if ticker is a crypto pair."""
    crypto_suffixes = ["-USD", "-USDT", "-BTC", "-EUR"]
    crypto_tickers = {"BTC", "ETH", "SOL", "ADA", "DOGE", "XRP", "DOT", "AVAX",
                      "MATIC", "LINK", "UNI", "ATOM", "LTC", "BNB", "TON", "SUI"}
    base = ticker.split("-")[0]
    return any(ticker.endswith(s) for s in crypto_suffixes) or base in crypto_tickers


def _fetch_binance(ticker: str, period: str, interval: str) -> dict:
    """Fetch from Binance public REST API (no key needed)."""
    import requests

    # Map ticker to Binance symbol
    binance_map = {
        "BTC-USD": "BTCUSDT", "ETH-USD": "ETHUSDT", "SOL-USD": "SOLUSDT",
        "ADA-USD": "ADAUSDT", "DOGE-USD": "DOGEUSDT", "XRP-USD": "XRPUSDT",
        "DOT-USD": "DOTUSDT", "AVAX-USD": "AVAXUSDT", "LINK-USD": "LINKUSDT",
        "BNB-USD": "BNBUSDT", "TON-USD": "TONUSDT", "SUI-USD": "SUIUSDT",
        "LTC-USD": "LTCUSDT", "UNI-USD": "UNIUSDT",
    }
    symbol = binance_map.get(ticker, ticker.replace("-USD", "USDT").replace("-", ""))

    # Map intervals
    interval_map = {"1d": "1d", "1h": "1h", "4h": "4h", "1wk": "1w", "1mo": "1M",
                     "15m": "15m", "5m": "5m", "1m": "1m"}
    bi = interval_map.get(interval, "1d")

    # Map period to limit
    period_limits = {"1d": 1, "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180,
                      "1y": 365, "2y": 730}
    limit = period_limits.get(period, 90)

    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": bi, "limit": min(limit, 1000)}
        resp = requests.get(url, params=params, timeout=10)
        if not resp.ok:
            return {"df": None, "source": "binance", "error": f"Binance HTTP {resp.status_code}"}

        data = resp.json()
        if not data or not isinstance(data, list):
            return {"df": None, "source": "binance", "error": "Empty response"}

        import pandas as pd
        rows = []
        for k in data:
            rows.append({
                "Date": pd.Timestamp(k[0], unit="ms"),
                "Open": float(k[1]), "High": float(k[2]),
                "Low": float(k[3]), "Close": float(k[4]),
                "Volume": float(k[5]),
            })
        df = pd.DataFrame(rows).set_index("Date")
        return {"df": df, "source": "binance", "error": None, "symbol": symbol}

    except Exception as e:
        return {"df": None, "source": "binance", "error": str(e)}


def _fetch_yfinance(ticker: str, period: str, interval: str) -> dict:
    """Fetch from yfinance."""
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty:
            return {"df": None, "source": "yfinance", "error": f"No data for {ticker}"}
        return {"df": df, "source": "yfinance", "error": None}
    except Exception as e:
        return {"df": None, "source": "yfinance", "error": str(e)}


# ─────── Fallback static data for demo mode ───────

FALLBACK_DATA = {
    "AAPL": {"price": 230.50, "change_pct": 1.2, "rsi": 55.3, "signal": "HOLD", "name": "Apple Inc"},
    "NVDA": {"price": 135.80, "change_pct": 2.5, "rsi": 62.1, "signal": "BUY", "name": "NVIDIA"},
    "MSFT": {"price": 420.30, "change_pct": -0.3, "rsi": 48.7, "signal": "HOLD", "name": "Microsoft"},
    "GOOGL": {"price": 175.60, "change_pct": 0.8, "rsi": 51.2, "signal": "HOLD", "name": "Alphabet"},
    "TSLA": {"price": 285.40, "change_pct": -1.5, "rsi": 42.8, "signal": "HOLD", "name": "Tesla"},
    "THYAO.IS": {"price": 340.20, "change_pct": 0.9, "rsi": 58.4, "signal": "HOLD", "name": "THY"},
    "GARAN.IS": {"price": 145.60, "change_pct": 1.1, "rsi": 53.2, "signal": "HOLD", "name": "Garanti"},
    "AKBNK.IS": {"price": 72.30, "change_pct": 0.5, "rsi": 49.8, "signal": "HOLD", "name": "Akbank"},
    "BTC-USD": {"price": 94500.0, "change_pct": 3.1, "rsi": 65.4, "signal": "BUY", "name": "Bitcoin"},
    "ETH-USD": {"price": 3200.0, "change_pct": 2.8, "rsi": 61.2, "signal": "BUY", "name": "Ethereum"},
    "GC=F": {"price": 3320.0, "change_pct": 0.4, "rsi": 54.1, "signal": "HOLD", "name": "Gold"},
}


def get_fallback(ticker: str) -> dict:
    """Return static fallback data for demo mode when APIs are down."""
    ticker = ticker.strip().upper()
    if ticker in FALLBACK_DATA:
        data = FALLBACK_DATA[ticker].copy()
        data["ticker"] = ticker
        data["source"] = "fallback"
        data["note"] = "Statik demo verisi — gerçek zamanlı veri alınamadı"
        return data
    return {
        "ticker": ticker, "price": 100.0, "change_pct": 0.0,
        "rsi": 50.0, "signal": "HOLD", "name": ticker,
        "source": "fallback", "note": "Bilinmeyen ticker — varsayılan veri",
    }
