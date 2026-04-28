"""Tool: Teknik analiz indikatörleri hesaplama."""

from smolagents import tool


@tool
def get_technical_indicators(ticker: str, period: str = "3mo") -> str:
    """
    Computes key technical indicators (RSI, MACD, Bollinger Bands, SMA, EMA, ATR, Stochastic)
    for a given ticker. Use this to understand momentum, trend direction, and volatility.

    Args:
        ticker: Stock/crypto ticker symbol (e.g. 'AAPL', 'BTC-USD', 'NVDA')
        period: Time period for historical data. Options: '1mo','3mo','6mo','1y'. Default '3mo'.

    Returns:
        JSON string with technical indicator values and their signal interpretations.
    """
    import yfinance as yf
    import pandas_ta as ta
    import json

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval="1d")

        if df.empty or len(df) < 26:
            return json.dumps({"error": f"Insufficient data for {ticker} (need 26+ days)", "ticker": ticker})

        rsi = ta.rsi(df["Close"], length=14)
        rsi_val = round(float(rsi.iloc[-1]), 2) if rsi is not None and not rsi.empty else None
        if rsi_val is not None:
            if rsi_val > 70: rsi_signal = "OVERBOUGHT"
            elif rsi_val < 30: rsi_signal = "OVERSOLD"
            elif rsi_val > 60: rsi_signal = "BULLISH"
            elif rsi_val < 40: rsi_signal = "BEARISH"
            else: rsi_signal = "NEUTRAL"
        else: rsi_signal = "N/A"

        macd_df = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            macd_val = round(float(macd_df.iloc[-1, 0]), 4)
            macd_signal_val = round(float(macd_df.iloc[-1, 2]), 4)
            macd_hist = round(float(macd_df.iloc[-1, 1]), 4)
            macd_signal = "BULLISH" if macd_hist > 0 else "BEARISH"
            if len(macd_df) >= 2:
                prev_hist = float(macd_df.iloc[-2, 1])
                if prev_hist < 0 and macd_hist > 0: macd_signal = "BULLISH_CROSSOVER"
                elif prev_hist > 0 and macd_hist < 0: macd_signal = "BEARISH_CROSSOVER"
        else:
            macd_val = macd_signal_val = macd_hist = None
            macd_signal = "N/A"

        bbands = ta.bbands(df["Close"], length=20, std=2)
        if bbands is not None and not bbands.empty:
            bb_upper = round(float(bbands.iloc[-1, 0]), 2)
            bb_mid = round(float(bbands.iloc[-1, 1]), 2)
            bb_lower = round(float(bbands.iloc[-1, 2]), 2)
            current_price = round(float(df["Close"].iloc[-1]), 2)
            if current_price > bb_upper: bb_signal = "OVERBOUGHT"
            elif current_price < bb_lower: bb_signal = "OVERSOLD"
            elif current_price > bb_mid: bb_signal = "BULLISH"
            else: bb_signal = "BEARISH"
            bb_width = round((bb_upper - bb_lower) / bb_mid * 100, 2)
        else:
            bb_upper = bb_mid = bb_lower = bb_width = None
            bb_signal = "N/A"
            current_price = round(float(df["Close"].iloc[-1]), 2)

        sma_20 = ta.sma(df["Close"], length=20)
        sma_50 = ta.sma(df["Close"], length=50)
        sma_20_val = round(float(sma_20.iloc[-1]), 2) if sma_20 is not None and not sma_20.empty else None
        sma_50_val = round(float(sma_50.iloc[-1]), 2) if sma_50 is not None and not sma_50.empty else None
        if sma_20_val and sma_50_val:
            if sma_20_val > sma_50_val and current_price > sma_20_val: sma_signal = "STRONG_BULLISH"
            elif sma_20_val > sma_50_val: sma_signal = "BULLISH"
            elif sma_20_val < sma_50_val and current_price < sma_20_val: sma_signal = "STRONG_BEARISH"
            else: sma_signal = "BEARISH"
        else: sma_signal = "N/A"

        ema_12 = ta.ema(df["Close"], length=12)
        ema_26 = ta.ema(df["Close"], length=26)
        ema_12_val = round(float(ema_12.iloc[-1]), 2) if ema_12 is not None and not ema_12.empty else None
        ema_26_val = round(float(ema_26.iloc[-1]), 2) if ema_26 is not None and not ema_26.empty else None

        atr = ta.atr(df["High"], df["Low"], df["Close"], length=14)
        atr_val = round(float(atr.iloc[-1]), 2) if atr is not None and not atr.empty else None
        atr_pct = round((atr_val / current_price) * 100, 2) if atr_val else None

        stoch = ta.stoch(df["High"], df["Low"], df["Close"], k=14, d=3, smooth_k=3)
        if stoch is not None and not stoch.empty:
            stoch_k = round(float(stoch.iloc[-1, 0]), 2)
            stoch_d = round(float(stoch.iloc[-1, 1]), 2)
            if stoch_k > 80: stoch_signal = "OVERBOUGHT"
            elif stoch_k < 20: stoch_signal = "OVERSOLD"
            else: stoch_signal = "NEUTRAL"
        else:
            stoch_k = stoch_d = None
            stoch_signal = "N/A"

        signals = [rsi_signal, macd_signal, bb_signal, sma_signal, stoch_signal]
        bullish_count = sum(1 for s in signals if "BULLISH" in s or s == "OVERSOLD")
        bearish_count = sum(1 for s in signals if "BEARISH" in s or s == "OVERBOUGHT")
        if bullish_count >= 4: overall = "STRONG_BUY"
        elif bullish_count >= 3: overall = "BUY"
        elif bearish_count >= 4: overall = "STRONG_SELL"
        elif bearish_count >= 3: overall = "SELL"
        else: overall = "HOLD"

        result = {
            "ticker": ticker.upper(), "current_price": current_price,
            "overall_signal": overall, "bullish_indicators": bullish_count, "bearish_indicators": bearish_count,
            "indicators": {
                "rsi": {"value": rsi_val, "signal": rsi_signal},
                "macd": {"macd": macd_val, "signal_line": macd_signal_val, "histogram": macd_hist, "signal": macd_signal},
                "bollinger_bands": {"upper": bb_upper, "middle": bb_mid, "lower": bb_lower, "width_pct": bb_width, "signal": bb_signal},
                "sma": {"sma_20": sma_20_val, "sma_50": sma_50_val, "signal": sma_signal},
                "ema": {"ema_12": ema_12_val, "ema_26": ema_26_val},
                "atr": {"value": atr_val, "atr_pct": atr_pct},
                "stochastic": {"k": stoch_k, "d": stoch_d, "signal": stoch_signal},
            },
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})
