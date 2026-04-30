"""
C. Advanced Reasoning — External context + technical synthesis
================================================================
Goes beyond "RSI is oversold" to answer questions like:
  - "Is there a liquidity trap forming?"
  - "How would a Fed rate decision affect this setup?"
  - "Is this a bear market rally or genuine reversal?"
  - "What macro regime are we in and does it support this trade?"

Combines:
  1. Technical signal data (from signal_engine)
  2. Market regime detection (trending/ranging/volatile)
  3. Macro context awareness (rate environment, risk-on/risk-off)
  4. Pattern recognition (traps, divergences, exhaustion)
  5. Synthesized narrative (template-based, LLM-enhanced if available)
"""

import numpy as np
import datetime
import logging

logger = logging.getLogger("trade_bot.reasoning")


def generate_advanced_reasoning(ticker: str, signal_data: dict = None,
                                 period: str = "3mo") -> dict:
    """
    Build deep contextual reasoning that merges technical + external factors.
    """
    from core.data_ingest import fetch_ohlcv, get_ticker_name
    from core.signal_engine import generate_signal

    if signal_data is None:
        signal_data = generate_signal(ticker, period=period)

    result = fetch_ohlcv(ticker, period="6mo", interval="1d")
    df = result.get("df")
    if df is None or len(df) < 50:
        return {"error": "Insufficient data", "ticker": ticker}

    closes = df["Close"].values.astype(float)
    volumes = df["Volume"].values.astype(float)
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)

    ind = signal_data.get("indicators", {})
    sig = signal_data.get("signal", "HOLD")
    rsi = ind.get("rsi", {}).get("value", 50)
    vol_regime = ind.get("volatility", {}).get("regime", "MODERATE")

    analyses = []

    # ═══════════════════════════════════════
    # 1. MARKET REGIME DETECTION
    # ═══════════════════════════════════════
    regime = _detect_market_regime(closes, volumes)
    analyses.append({
        "category": "Market Regime",
        "icon": "🌍",
        "finding": regime["regime"],
        "detail": regime["explanation"],
        "impact": regime["signal_impact"],
    })

    # ═══════════════════════════════════════
    # 2. LIQUIDITY TRAP DETECTION
    # ═══════════════════════════════════════
    trap = _detect_liquidity_trap(closes, volumes, sig)
    if trap["detected"]:
        analyses.append({
            "category": "Liquidity Trap Warning",
            "icon": "⚠️",
            "finding": trap["type"],
            "detail": trap["explanation"],
            "impact": trap["recommendation"],
        })

    # ═══════════════════════════════════════
    # 3. DIVERGENCE CHECK (RSI vs Price)
    # ═══════════════════════════════════════
    div = _check_divergence(closes, rsi)
    if div["detected"]:
        analyses.append({
            "category": "Divergence",
            "icon": "🔀",
            "finding": div["type"],
            "detail": div["explanation"],
            "impact": div["impact"],
        })

    # ═══════════════════════════════════════
    # 4. MACRO CONTEXT AWARENESS
    # ═══════════════════════════════════════
    macro = _assess_macro_context(ticker, closes, vol_regime)
    analyses.append({
        "category": "Macro Context",
        "icon": "🏛️",
        "finding": macro["environment"],
        "detail": macro["explanation"],
        "impact": macro["trade_impact"],
    })

    # ═══════════════════════════════════════
    # 5. EXHAUSTION / REVERSAL DETECTION
    # ═══════════════════════════════════════
    exh = _detect_exhaustion(closes, volumes, rsi)
    if exh["detected"]:
        analyses.append({
            "category": "Exhaustion Signal",
            "icon": "💀",
            "finding": exh["type"],
            "detail": exh["explanation"],
            "impact": exh["impact"],
        })

    # ═══════════════════════════════════════
    # 6. SYNTHESIS — final contextual assessment
    # ═══════════════════════════════════════
    synthesis = _synthesize(sig, signal_data.get("confidence", 50), analyses, regime)

    return {
        "ticker": ticker.upper(),
        "signal": sig,
        "confidence_original": signal_data.get("confidence", 50),
        "confidence_adjusted": synthesis["adjusted_confidence"],
        "analyses": [a for a in analyses],
        "synthesis": synthesis,
        "timestamp": datetime.datetime.now().isoformat(),
    }


def _detect_market_regime(closes, volumes):
    """Classify market as trending/ranging/volatile."""
    if len(closes) < 50:
        return {"regime": "UNKNOWN", "explanation": "Insufficient data", "signal_impact": "Neutral"}

    # ADX proxy: count consecutive up/down days
    changes = np.diff(closes[-50:])
    up_days = sum(1 for c in changes if c > 0)
    up_pct = up_days / len(changes) * 100

    # Volatility
    ret = np.diff(closes[-30:]) / closes[-30:-1]
    vol = float(np.std(ret) * np.sqrt(252))

    # Range detection: price oscillating between levels
    high_20 = float(np.max(closes[-20:]))
    low_20 = float(np.min(closes[-20:]))
    range_pct = (high_20 - low_20) / low_20 * 100

    if vol > 0.40:
        regime = "VOLATILE"
        explanation = f"High volatility ({vol*100:.0f}% annual). Price swings are large and unpredictable."
        impact = "Trend signals less reliable. Consider wider stops or staying flat."
    elif up_pct > 65:
        regime = "STRONG_UPTREND"
        explanation = f"Strong uptrend: {up_pct:.0f}% of days are up. Momentum is clearly bullish."
        impact = "BUY signals are more reliable. Avoid fighting the trend."
    elif up_pct < 35:
        regime = "STRONG_DOWNTREND"
        explanation = f"Strong downtrend: only {up_pct:.0f}% of days are up. Bears in control."
        impact = "SELL signals are more reliable. Avoid bottom-fishing."
    elif range_pct < 8:
        regime = "RANGING"
        explanation = f"Price stuck in {range_pct:.1f}% range. No clear direction."
        impact = "Mean-reversion signals (RSI, Bollinger) work better than trend signals (SMA)."
    else:
        regime = "TRANSITIONAL"
        explanation = f"Market between regimes ({up_pct:.0f}% up, {range_pct:.1f}% range). Watch for breakout."
        impact = "Wait for confirmation. Current signals have lower conviction."

    return {"regime": regime, "explanation": explanation, "signal_impact": impact}


def _detect_liquidity_trap(closes, volumes, signal):
    """Detect potential bull/bear traps (fake breakouts on low volume)."""
    if len(closes) < 20 or len(volumes) < 20:
        return {"detected": False}

    recent_high = float(np.max(closes[-20:]))
    recent_low = float(np.min(closes[-20:]))
    current = float(closes[-1])
    avg_vol = float(np.mean(volumes[-20:]))
    recent_vol = float(np.mean(volumes[-3:]))

    # Bull trap: price near highs but volume declining
    if signal in ("BUY", "STRONG_BUY"):
        near_high = (current - recent_low) / (recent_high - recent_low) > 0.8 if recent_high > recent_low else False
        low_vol = recent_vol < avg_vol * 0.7

        if near_high and low_vol:
            return {
                "detected": True,
                "type": "POTENTIAL BULL TRAP",
                "explanation": f"Price near 20-day high but volume declining ({recent_vol/avg_vol:.1f}x avg). Breakout may be fake.",
                "recommendation": "Reduce position size. Wait for volume confirmation before full entry.",
            }

    # Bear trap: price near lows but volume declining
    if signal in ("SELL", "STRONG_SELL"):
        near_low = (current - recent_low) / (recent_high - recent_low) < 0.2 if recent_high > recent_low else False
        low_vol = recent_vol < avg_vol * 0.7

        if near_low and low_vol:
            return {
                "detected": True,
                "type": "POTENTIAL BEAR TRAP",
                "explanation": f"Price near 20-day low but volume declining. Breakdown may be fake.",
                "recommendation": "Wait for volume confirmation before shorting.",
            }

    return {"detected": False}


def _check_divergence(closes, current_rsi):
    """Check for RSI-price divergence (last 20 bars)."""
    if len(closes) < 20:
        return {"detected": False}

    price_trend = closes[-1] > closes[-20]  # Price going up?
    # Simple RSI trend: is current RSI lower than 20 bars ago?
    # (We only have current RSI, so we approximate)
    price_higher = closes[-1] > closes[-10]
    rsi_extreme = current_rsi > 65 or current_rsi < 35

    # Bearish divergence: price making new highs but RSI overbought
    if price_higher and current_rsi > 70:
        return {
            "detected": True,
            "type": "BEARISH DIVERGENCE",
            "explanation": f"Price near highs but RSI at {current_rsi:.0f} (overbought). Momentum weakening.",
            "impact": "BUY signals may be premature. Potential reversal ahead.",
        }

    # Bullish divergence: price making new lows but RSI oversold
    if not price_higher and current_rsi < 30:
        return {
            "detected": True,
            "type": "BULLISH DIVERGENCE",
            "explanation": f"Price near lows but RSI at {current_rsi:.0f} (oversold). Selling exhaustion.",
            "impact": "SELL signals may be overdone. Watch for bounce.",
        }

    return {"detected": False}


def _assess_macro_context(ticker, closes, vol_regime):
    """
    Assess macro environment based on asset class + volatility.
    In production: integrate real macro data (Fed, TCMB, yields).
    """
    ticker = ticker.upper()
    is_crypto = "-USD" in ticker and ticker.split("-")[0] in {"BTC", "ETH", "SOL"}
    is_bist = ".IS" in ticker
    is_commodity = ticker in ("GC=F", "SI=F", "CL=F")

    # Price momentum as macro proxy
    if len(closes) >= 60:
        ret_60d = (closes[-1] - closes[-60]) / closes[-60] * 100
    else:
        ret_60d = 0

    if is_crypto:
        if vol_regime in ("HIGH", "EXTREME"):
            env = "RISK-OFF"
            explanation = "Crypto in high volatility. Market likely in risk-off mode. Institutional selling possible."
            impact = "Reduce position sizes. Crypto signals less reliable in risk-off."
        else:
            env = "RISK-ON"
            explanation = "Crypto volatility manageable. Risk appetite appears healthy."
            impact = "Normal signal reliability. Trend-following viable."
    elif is_bist:
        if vol_regime in ("HIGH", "EXTREME"):
            env = "TIGHTENING"
            explanation = "BIST in high volatility. Possibly TCMB policy uncertainty or TRY pressure."
            impact = "Caution on long positions. FX risk adds to equity risk."
        else:
            env = "STABLE"
            explanation = "BIST volatility manageable. Macro conditions appear stable."
            impact = "Normal trading conditions for BIST equities."
    elif is_commodity:
        env = "INFLATION-SENSITIVE"
        explanation = "Commodity prices reflect inflation expectations. Watch for central bank rhetoric."
        impact = "Trend signals matter more than mean-reversion in commodities."
    else:
        if ret_60d > 10 and vol_regime == "LOW":
            env = "GOLDILOCKS"
            explanation = f"Strong returns (+{ret_60d:.0f}% over 60d) with low volatility. Ideal conditions."
            impact = "BUY signals highly reliable. Trend-following works best here."
        elif ret_60d < -10:
            env = "CORRECTION"
            explanation = f"Significant decline ({ret_60d:.0f}% over 60d). Market under pressure."
            impact = "Be cautious with BUY signals. SELL/HOLD more appropriate."
        elif vol_regime in ("HIGH", "EXTREME"):
            env = "UNCERTAIN"
            explanation = "High volatility suggests macro uncertainty. Could be policy, geopolitical, or earnings."
            impact = "All signals carry extra risk. Consider reducing exposure."
        else:
            env = "NORMAL"
            explanation = "Standard market conditions. No unusual macro signals."
            impact = "Normal signal reliability."

    return {"environment": env, "explanation": explanation, "trade_impact": impact}


def _detect_exhaustion(closes, volumes, rsi):
    """Detect buying/selling exhaustion (climactic volume + extreme RSI)."""
    if len(closes) < 10 or len(volumes) < 10:
        return {"detected": False}

    avg_vol = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
    recent_vol = float(volumes[-1])
    vol_spike = recent_vol > avg_vol * 2.5

    # Buying exhaustion: extreme high RSI + volume spike
    if rsi > 75 and vol_spike:
        return {
            "detected": True,
            "type": "BUYING EXHAUSTION",
            "explanation": f"RSI at {rsi:.0f} with volume spike ({recent_vol/avg_vol:.1f}x avg). Buyers may be spent.",
            "impact": "Potential top forming. Avoid new BUY entries. Consider taking profits.",
        }

    # Selling exhaustion: extreme low RSI + volume spike
    if rsi < 25 and vol_spike:
        return {
            "detected": True,
            "type": "SELLING EXHAUSTION",
            "explanation": f"RSI at {rsi:.0f} with volume spike ({recent_vol/avg_vol:.1f}x avg). Capitulation possible.",
            "impact": "Potential bottom forming. Watch for reversal confirmation before BUY.",
        }

    return {"detected": False}


def _synthesize(signal, confidence, analyses, regime):
    """Combine all contextual analyses into final adjusted assessment."""
    # Start with original confidence
    adj = confidence

    # Adjust based on regime alignment
    regime_name = regime.get("regime", "UNKNOWN")
    if signal in ("BUY", "STRONG_BUY"):
        if regime_name == "STRONG_UPTREND": adj += 10
        elif regime_name == "STRONG_DOWNTREND": adj -= 15
        elif regime_name == "VOLATILE": adj -= 10
        elif regime_name == "RANGING": adj -= 5
    elif signal in ("SELL", "STRONG_SELL"):
        if regime_name == "STRONG_DOWNTREND": adj += 10
        elif regime_name == "STRONG_UPTREND": adj -= 15
        elif regime_name == "VOLATILE": adj -= 5

    # Adjust for warnings
    warnings = [a for a in analyses if a["icon"] in ("⚠️", "💀", "🔀")]
    for w in warnings:
        if "TRAP" in w.get("finding", ""):
            adj -= 10
        elif "EXHAUSTION" in w.get("finding", ""):
            adj -= 8
        elif "DIVERGENCE" in w.get("finding", ""):
            adj -= 5

    adj = max(5, min(95, adj))

    # Build narrative
    bullish_factors = [a for a in analyses if "reliable" in a.get("impact", "").lower() or "bullish" in a.get("finding", "").lower()]
    bearish_factors = [a for a in analyses if "caution" in a.get("impact", "").lower() or "bearish" in a.get("finding", "").lower() or "trap" in a.get("finding", "").lower()]

    if adj > confidence:
        verdict = "STRENGTHENED"
        narrative = f"Contextual factors support the {signal} signal. Regime alignment and absence of traps increase conviction."
    elif adj < confidence - 10:
        verdict = "WEAKENED"
        narrative = f"Warning signs detected. {len(warnings)} contextual risk(s) reduce conviction from {confidence}% to {adj}%."
    else:
        verdict = "CONFIRMED"
        narrative = f"Signal holds under contextual analysis. No major contradictions found."

    return {
        "verdict": verdict,
        "adjusted_confidence": adj,
        "original_confidence": confidence,
        "confidence_change": adj - confidence,
        "narrative": narrative,
        "bullish_factors": len(bullish_factors),
        "bearish_factors": len(bearish_factors),
        "warnings_count": len(warnings),
    }
