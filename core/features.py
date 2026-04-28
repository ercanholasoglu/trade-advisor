"""
🔧 Feature Engineering Pipeline
=================================
Takes raw OHLCV DataFrame → returns feature dict with all indicators.
Deterministic, no LLM. Pure math.
"""

import json
import numpy as np
import logging

logger = logging.getLogger("trade_bot.features")


def compute_features(ticker: str, period: str = "3mo") -> dict:
    """
    Full feature extraction from OHLCV data.
    Returns dict with price, indicators, signals — all numeric, no LLM.
    """
    from core.data_ingest import fetch_ohlcv, get_fallback

    result = fetch_ohlcv(ticker, period=period, interval="1d")
    df = result.get("df")

    if df is None or df.empty or len(df) < 26:
        fb = get_fallback(ticker)
        return {
            "ticker": ticker, "source": "fallback",
            "error": result.get("error", "No data"),
            "price": fb["price"], "change_pct": fb["change_pct"],
            "rsi": fb["rsi"], "signal": fb["signal"],
            "features_available": False,
        }

    closes = df["Close"].values
    highs = df["High"].values
    lows = df["Low"].values
    volumes = df["Volume"].values
    current_price = float(closes[-1])
    prev_price = float(closes[-2])
    change_pct = (current_price - prev_price) / prev_price * 100

    features = {
        "ticker": ticker.upper(),
        "source": result.get("source", "unknown"),
        "features_available": True,
        "price": round(current_price, 2),
        "change_pct": round(change_pct, 2),
        "data_points": len(df),
    }

    # ── RSI(14) ──
    rsi = _rsi(closes, 14)
    features["rsi"] = round(rsi, 2)
    features["rsi_signal"] = (
        "OVERBOUGHT" if rsi > 70 else
        "OVERSOLD" if rsi < 30 else
        "BULLISH" if rsi > 60 else
        "BEARISH" if rsi < 40 else "NEUTRAL"
    )

    # ── MACD(12,26,9) ──
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    histogram = macd_line - signal_line
    features["macd"] = round(float(macd_line[-1]), 4)
    features["macd_signal"] = round(float(signal_line[-1]), 4)
    features["macd_histogram"] = round(float(histogram[-1]), 4)
    features["macd_trend"] = "BULLISH" if histogram[-1] > 0 else "BEARISH"
    if len(histogram) >= 2:
        if histogram[-2] < 0 and histogram[-1] > 0:
            features["macd_trend"] = "BULLISH_CROSSOVER"
        elif histogram[-2] > 0 and histogram[-1] < 0:
            features["macd_trend"] = "BEARISH_CROSSOVER"

    # ── SMA(20, 50) ──
    sma20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else None
    sma50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else None
    features["sma20"] = round(sma20, 2) if sma20 else None
    features["sma50"] = round(sma50, 2) if sma50 else None
    if sma20 and sma50:
        if sma20 > sma50 and current_price > sma20:
            features["sma_signal"] = "STRONG_BULLISH"
        elif sma20 > sma50:
            features["sma_signal"] = "BULLISH"
        elif sma20 < sma50 and current_price < sma20:
            features["sma_signal"] = "STRONG_BEARISH"
        else:
            features["sma_signal"] = "BEARISH"
    else:
        features["sma_signal"] = "N/A"

    # ── Bollinger Bands(20, 2) ──
    if len(closes) >= 20:
        bb_mid = float(np.mean(closes[-20:]))
        bb_std = float(np.std(closes[-20:]))
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        features["bb_upper"] = round(bb_upper, 2)
        features["bb_lower"] = round(bb_lower, 2)
        features["bb_width"] = round((bb_upper - bb_lower) / bb_mid * 100, 2)
        features["bb_signal"] = (
            "OVERBOUGHT" if current_price > bb_upper else
            "OVERSOLD" if current_price < bb_lower else
            "BULLISH" if current_price > bb_mid else "BEARISH"
        )
    else:
        features["bb_signal"] = "N/A"

    # ── ATR(14) ──
    if len(closes) >= 15:
        tr = np.maximum(
            highs[-14:] - lows[-14:],
            np.maximum(
                np.abs(highs[-14:] - np.roll(closes[-14:], 1)),
                np.abs(lows[-14:] - np.roll(closes[-14:], 1))
            )
        )[1:]
        atr = float(np.mean(tr))
        features["atr"] = round(atr, 2)
        features["atr_pct"] = round(atr / current_price * 100, 2)
    else:
        features["atr"] = None

    # ── Volume analysis ──
    if len(volumes) >= 20:
        avg_vol = float(np.mean(volumes[-20:]))
        features["volume_ratio"] = round(float(volumes[-1]) / avg_vol, 2) if avg_vol > 0 else 1.0
    else:
        features["volume_ratio"] = 1.0

    # ── Volatility ──
    if len(closes) >= 30:
        returns = np.diff(closes[-30:]) / closes[-30:-1]
        daily_vol = float(np.std(returns))
        features["daily_volatility"] = round(daily_vol * 100, 2)
        features["annualized_volatility"] = round(daily_vol * np.sqrt(252) * 100, 1)
        features["volatility_regime"] = (
            "LOW" if daily_vol * np.sqrt(252) < 0.15 else
            "MODERATE" if daily_vol * np.sqrt(252) < 0.30 else
            "HIGH" if daily_vol * np.sqrt(252) < 0.50 else "EXTREME"
        )
    else:
        features["volatility_regime"] = "N/A"

    # ── Overall signal (5-indicator vote) ──
    signals = [
        features.get("rsi_signal", "N/A"),
        features.get("macd_trend", "N/A"),
        features.get("bb_signal", "N/A"),
        features.get("sma_signal", "N/A"),
    ]
    bullish = sum(1 for s in signals if "BULLISH" in s or s == "OVERSOLD")
    bearish = sum(1 for s in signals if "BEARISH" in s or s == "OVERBOUGHT")
    if bullish >= 3:
        features["signal"] = "STRONG_BUY" if bullish >= 4 else "BUY"
    elif bearish >= 3:
        features["signal"] = "STRONG_SELL" if bearish >= 4 else "SELL"
    else:
        features["signal"] = "HOLD"

    features["bullish_count"] = bullish
    features["bearish_count"] = bearish

    return features


def _rsi(closes, period=14):
    """Wilder RSI."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    return 100 - (100 / (1 + avg_gain / avg_loss))


def _ema(data, period):
    """Exponential moving average."""
    data = np.array(data, dtype=float)
    ema = np.zeros_like(data)
    ema[:period] = np.mean(data[:period])
    multiplier = 2 / (period + 1)
    for i in range(period, len(data)):
        ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema
