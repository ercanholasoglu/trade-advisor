"""
Tool: Options Flow Analysis
=============================
Put/Call ratio, unusual volume detection via yfinance options API.
"""

from smolagents import tool
from tools.cache import cache, TTL_OPTIONS



@cache(ttl=TTL_OPTIONS)
def _cached_get_options_flow(ticker: str) -> str:
    import yfinance as yf
    import json
    import numpy as np

    try:
        stock = yf.Ticker(ticker.strip().upper())

        # Get available expiration dates
        expirations = stock.options
        if not expirations:
            return json.dumps({"error": f"No options data for {ticker}", "ticker": ticker})

        # Analyze nearest 3 expiration dates
        analysis_dates = expirations[:3]
        total_call_volume = 0
        total_put_volume = 0
        total_call_oi = 0
        total_put_oi = 0
        unusual_calls = []
        unusual_puts = []
        max_pain_data = []

        for exp_date in analysis_dates:
            try:
                chain = stock.option_chain(exp_date)
                calls = chain.calls
                puts = chain.puts

                if calls.empty and puts.empty:
                    continue

                # Volume
                call_vol = int(calls["volume"].sum()) if "volume" in calls.columns and not calls["volume"].isna().all() else 0
                put_vol = int(puts["volume"].sum()) if "volume" in puts.columns and not puts["volume"].isna().all() else 0
                total_call_volume += call_vol
                total_put_volume += put_vol

                # Open Interest
                call_oi = int(calls["openInterest"].sum()) if "openInterest" in calls.columns and not calls["openInterest"].isna().all() else 0
                put_oi = int(puts["openInterest"].sum()) if "openInterest" in puts.columns and not puts["openInterest"].isna().all() else 0
                total_call_oi += call_oi
                total_put_oi += put_oi

                # Detect unusual volume (3x+ average OI)
                for _, row in calls.iterrows():
                    vol = row.get("volume", 0)
                    oi = row.get("openInterest", 1)
                    if vol and oi and not np.isnan(vol) and not np.isnan(oi) and oi > 0 and vol > oi * 3 and vol > 1000:
                        unusual_calls.append({
                            "expiry": exp_date,
                            "strike": float(row["strike"]),
                            "volume": int(vol),
                            "open_interest": int(oi),
                            "volume_oi_ratio": round(vol / oi, 1),
                            "implied_vol": round(float(row.get("impliedVolatility", 0)) * 100, 1),
                        })

                for _, row in puts.iterrows():
                    vol = row.get("volume", 0)
                    oi = row.get("openInterest", 1)
                    if vol and oi and not np.isnan(vol) and not np.isnan(oi) and oi > 0 and vol > oi * 3 and vol > 1000:
                        unusual_puts.append({
                            "expiry": exp_date,
                            "strike": float(row["strike"]),
                            "volume": int(vol),
                            "open_interest": int(oi),
                            "volume_oi_ratio": round(vol / oi, 1),
                            "implied_vol": round(float(row.get("impliedVolatility", 0)) * 100, 1),
                        })

                # Max pain calculation (simplified)
                strikes = sorted(set(calls["strike"].tolist() + puts["strike"].tolist()))
                if strikes:
                    min_pain = float('inf')
                    max_pain_strike = 0
                    for s in strikes:
                        pain = 0
                        for _, c in calls.iterrows():
                            if s > c["strike"]:
                                pain += (s - c["strike"]) * c.get("openInterest", 0)
                        for _, p in puts.iterrows():
                            if s < p["strike"]:
                                pain += (p["strike"] - s) * p.get("openInterest", 0)
                        if not np.isnan(pain) and pain < min_pain:
                            min_pain = pain
                            max_pain_strike = s

                    max_pain_data.append({
                        "expiry": exp_date,
                        "max_pain": round(max_pain_strike, 2),
                    })

            except Exception:
                continue

        # Calculate ratios
        pcr_volume = round(total_put_volume / total_call_volume, 3) if total_call_volume > 0 else None
        pcr_oi = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else None

        # Get current price for context
        try:
            hist = stock.history(period="1d")
            current_price = float(hist["Close"].iloc[-1]) if not hist.empty else None
        except Exception:
            current_price = None

        # Generate signals
        signals = []
        if pcr_volume is not None:
            if pcr_volume > 1.5:
                signals.append("HIGH_PUT_CALL_RATIO_BEARISH")
            elif pcr_volume > 1.0:
                signals.append("ELEVATED_PUT_CALL_RATIO")
            elif pcr_volume < 0.5:
                signals.append("LOW_PUT_CALL_RATIO_BULLISH")
            elif pcr_volume < 0.7:
                signals.append("MODERATELY_BULLISH_OPTIONS")

        if unusual_calls:
            signals.append(f"UNUSUAL_CALL_VOLUME_{len(unusual_calls)}_STRIKES")
        if unusual_puts:
            signals.append(f"UNUSUAL_PUT_VOLUME_{len(unusual_puts)}_STRIKES")

        # Sentiment from options
        if pcr_volume is not None:
            if pcr_volume > 1.3:
                options_sentiment = "BEARISH"
            elif pcr_volume > 0.9:
                options_sentiment = "NEUTRAL"
            elif pcr_volume > 0.6:
                options_sentiment = "MODERATELY_BULLISH"
            else:
                options_sentiment = "BULLISH"
        else:
            options_sentiment = "N/A"

        result = {
            "ticker": ticker.upper(),
            "current_price": current_price,
            "expiration_dates_analyzed": analysis_dates,
            "put_call_ratio": {
                "volume_based": pcr_volume,
                "open_interest_based": pcr_oi,
            },
            "volume_summary": {
                "total_call_volume": total_call_volume,
                "total_put_volume": total_put_volume,
                "total_call_oi": total_call_oi,
                "total_put_oi": total_put_oi,
            },
            "unusual_activity": {
                "unusual_calls": unusual_calls[:5],  # Top 5
                "unusual_puts": unusual_puts[:5],
                "total_unusual_calls": len(unusual_calls),
                "total_unusual_puts": len(unusual_puts),
            },
            "max_pain": max_pain_data,
            "options_sentiment": options_sentiment,
            "signals": signals,
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


@tool
def get_options_flow(ticker: str) -> str:
    """
    Analyzes options activity for a stock using yfinance options API.
    Calculates Put/Call ratio, detects unusual volume, and identifies sentiment signals.
    High put/call ratio is a bearish signal; high call volume is bullish.

    Args:
        ticker: US stock ticker symbol (e.g. 'AAPL', 'NVDA', 'TSLA'). Must have listed options.

    Returns:
        JSON string with put/call ratio, volume analysis, unusual activity flags, and signals.
    """
    return _cached_get_options_flow(ticker)
