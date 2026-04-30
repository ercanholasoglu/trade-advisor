"""
📊 Signal Engine — Deterministic rule-based signal generation
================================================================
This is the BRAIN of the system. Signals come from math, not LLM.

Pipeline:
  1. OHLCV data fetch (live or sample fallback)
  2. Technical indicators (RSI, SMA crossover, MACD, Bollinger, ATR)
  3. Volatility filter
  4. Multi-indicator voting → signal
  5. Confidence scoring
  6. Position sizing (ATR-based)

LLM is NEVER involved in signal generation.
"""

import datetime
import numpy as np
import logging

logger = logging.getLogger("trade_bot.signal_engine")


def generate_signal(ticker: str, portfolio_value: float = 100000,
                     risk_tolerance: str = "moderate", period: str = "3mo") -> dict:
    """
    Generate a complete trading signal with structured output.
    100% deterministic — no LLM, no randomness.
    
    Returns:
        {
            "signal": "BUY"|"SELL"|"HOLD"|"STRONG_BUY"|"STRONG_SELL",
            "confidence": 0-100,
            "reason": "human readable explanation",
            "risk": "low"|"medium"|"high"|"extreme",
            "indicators": {...},
            "trade": {...},
            "explainability": {...}
        }
    """
    from core.data_ingest import fetch_ohlcv, get_ticker_name
    
    result = fetch_ohlcv(ticker, period=period, interval="1d")
    df = result.get("df")
    source = result.get("source", "unknown")
    
    if df is None or len(df) < 26:
        return _empty_signal(ticker, "Insufficient data")
    
    closes = df["Close"].values.astype(float)
    highs = df["High"].values.astype(float)
    lows = df["Low"].values.astype(float)
    volumes = df["Volume"].values.astype(float)
    
    current_price = float(closes[-1])
    prev_price = float(closes[-2]) if len(closes) >= 2 else current_price
    change_pct = (current_price - prev_price) / prev_price * 100 if prev_price > 0 else 0
    
    # ═══════════════════════════════════════
    # INDICATOR CALCULATIONS
    # ═══════════════════════════════════════
    
    # 1. RSI(14)
    rsi_val = _rsi(closes, 14)
    rsi_signal = _classify_rsi(rsi_val)
    
    # 2. SMA Crossover (20/50)
    sma20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else None
    sma50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else None
    sma_crossover = _sma_crossover_signal(closes, sma20, sma50, current_price)
    
    # 3. MACD(12,26,9)
    macd_data = _macd(closes)
    
    # 4. Bollinger Bands(20,2)
    bb_data = _bollinger(closes)
    
    # 5. ATR(14) for volatility + position sizing
    atr_val = _atr(highs, lows, closes)
    
    # 6. Volume analysis
    vol_ratio = float(volumes[-1] / np.mean(volumes[-20:])) if len(volumes) >= 20 and np.mean(volumes[-20:]) > 0 else 1.0
    
    # 7. Volatility regime
    volatility = _volatility_regime(closes)
    
    # ═══════════════════════════════════════
    # SIGNAL VOTING (5 indicators)
    # ═══════════════════════════════════════
    
    votes = {
        "RSI": rsi_signal,
        "SMA_Crossover": sma_crossover["signal"],
        "MACD": macd_data["signal"],
        "Bollinger": bb_data["signal"],
        "Volume": "BULLISH" if vol_ratio > 1.5 else ("BEARISH" if vol_ratio < 0.5 else "NEUTRAL"),
    }
    
    bullish = sum(1 for s in votes.values() if "BULLISH" in s or s == "OVERSOLD")
    bearish = sum(1 for s in votes.values() if "BEARISH" in s or s == "OVERBOUGHT")
    
    # Determine signal
    if bullish >= 4:
        signal = "STRONG_BUY"
    elif bullish >= 3:
        signal = "BUY"
    elif bearish >= 4:
        signal = "STRONG_SELL"
    elif bearish >= 3:
        signal = "SELL"
    else:
        signal = "HOLD"
    
    # ═══════════════════════════════════════
    # VOLATILITY FILTER (can downgrade signal)
    # ═══════════════════════════════════════
    
    vol_regime = volatility["regime"]
    if vol_regime == "EXTREME":
        if signal in ("BUY", "STRONG_BUY"):
            signal = "HOLD"  # Don't buy in extreme volatility
        # Sell signals are ok in extreme vol
    
    # ═══════════════════════════════════════
    # CONFIDENCE SCORING
    # ═══════════════════════════════════════
    
    confidence = _compute_confidence(signal, bullish, bearish, rsi_val, vol_ratio, vol_regime)
    
    # ═══════════════════════════════════════
    # RISK CLASSIFICATION
    # ═══════════════════════════════════════
    
    risk = _classify_risk(vol_regime, rsi_val, confidence)
    
    # ═══════════════════════════════════════
    # POSITION SIZING (ATR-based)
    # ═══════════════════════════════════════
    
    trade = _compute_position(current_price, atr_val, portfolio_value, risk_tolerance, signal)
    
    # ═══════════════════════════════════════
    # EXPLAINABILITY — "Why does this trade make sense?"
    # ═══════════════════════════════════════
    
    reasons = _build_reasons(votes, rsi_val, sma_crossover, macd_data, bb_data, vol_regime, vol_ratio, signal)
    explanation = _build_explanation(ticker, signal, reasons, votes, trade, risk)
    
    # ═══════════════════════════════════════
    # STRUCTURED OUTPUT
    # ═══════════════════════════════════════
    
    return {
        "ticker": ticker.upper(),
        "company_name": get_ticker_name(ticker),
        "date": datetime.date.today().isoformat(),
        "data_source": source,
        
        # Core signal
        "signal": signal,
        "confidence": confidence,
        "risk": risk,
        "reason": reasons[0] if reasons else "No clear signal",
        
        # All reasons for explainability
        "reasons": reasons,
        
        # Price data
        "price": round(current_price, 2),
        "change_pct": round(change_pct, 2),
        
        # Indicators detail
        "indicators": {
            "rsi": {"value": round(rsi_val, 2), "signal": rsi_signal},
            "sma_crossover": {
                "sma20": round(sma20, 2) if sma20 else None,
                "sma50": round(sma50, 2) if sma50 else None,
                "signal": sma_crossover["signal"],
                "detail": sma_crossover["detail"],
            },
            "macd": {
                "macd_line": round(macd_data["macd_line"], 4),
                "signal_line": round(macd_data["signal_line"], 4),
                "histogram": round(macd_data["histogram"], 4),
                "signal": macd_data["signal"],
            },
            "bollinger": {
                "upper": round(bb_data["upper"], 2),
                "middle": round(bb_data["middle"], 2),
                "lower": round(bb_data["lower"], 2),
                "width_pct": round(bb_data["width_pct"], 2),
                "signal": bb_data["signal"],
            },
            "atr": round(atr_val, 2) if atr_val else None,
            "volume_ratio": round(vol_ratio, 2),
            "volatility": {
                "daily_pct": round(volatility["daily_pct"], 2),
                "annualized_pct": round(volatility["annualized_pct"], 1),
                "regime": vol_regime,
            },
        },
        
        # Vote tally
        "votes": votes,
        "bullish_count": bullish,
        "bearish_count": bearish,
        
        # Trade parameters
        "trade": trade,
        
        # Explainability
        "explainability": explanation,
    }


# ═══════════════════════════════════════════════════════
# INDICATOR FUNCTIONS
# ═══════════════════════════════════════════════════════

def _rsi(closes, period=14):
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
    return float(100 - (100 / (1 + avg_gain / avg_loss)))


def _rsi_series(closes, period=14):
    """Calculate RSI for all data points (for charts)."""
    if len(closes) < period + 1:
        return np.full(len(closes), 50.0)
    
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    rsi_values = np.full(len(closes), 50.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    if avg_loss > 0:
        rsi_values[period] = 100 - (100 / (1 + avg_gain / avg_loss))
    else:
        rsi_values[period] = 100.0
    
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi_values[i + 1] = 100.0
        else:
            rsi_values[i + 1] = 100 - (100 / (1 + avg_gain / avg_loss))
    
    return rsi_values


def _classify_rsi(rsi):
    if rsi > 70: return "OVERBOUGHT"
    if rsi < 30: return "OVERSOLD"
    if rsi > 60: return "BULLISH"
    if rsi < 40: return "BEARISH"
    return "NEUTRAL"


def _ema(data, period):
    data = np.array(data, dtype=float)
    ema = np.zeros_like(data)
    if len(data) < period:
        return data.copy()
    ema[:period] = np.mean(data[:period])
    multiplier = 2 / (period + 1)
    for i in range(period, len(data)):
        ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema


def _sma_crossover_signal(closes, sma20, sma50, current_price):
    """SMA(20/50) crossover — the core trend-following signal."""
    if sma20 is None or sma50 is None:
        return {"signal": "NEUTRAL", "detail": "Insufficient data for SMA"}
    
    if sma20 > sma50 and current_price > sma20:
        return {"signal": "STRONG_BULLISH", "detail": f"Golden cross: SMA20({sma20:.2f}) > SMA50({sma50:.2f}), price above both"}
    elif sma20 > sma50:
        return {"signal": "BULLISH", "detail": f"SMA20({sma20:.2f}) > SMA50({sma50:.2f}) — uptrend"}
    elif sma20 < sma50 and current_price < sma20:
        return {"signal": "STRONG_BEARISH", "detail": f"Death cross: SMA20({sma20:.2f}) < SMA50({sma50:.2f}), price below both"}
    else:
        return {"signal": "BEARISH", "detail": f"SMA20({sma20:.2f}) < SMA50({sma50:.2f}) — downtrend"}


def _macd(closes):
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_line = ema12 - ema26
    signal_line = _ema(macd_line, 9)
    histogram = macd_line - signal_line
    
    h = float(histogram[-1])
    sig = "BULLISH" if h > 0 else "BEARISH"
    if len(histogram) >= 2:
        if histogram[-2] < 0 and histogram[-1] > 0:
            sig = "BULLISH_CROSSOVER"
        elif histogram[-2] > 0 and histogram[-1] < 0:
            sig = "BEARISH_CROSSOVER"
    
    return {
        "macd_line": float(macd_line[-1]),
        "signal_line": float(signal_line[-1]),
        "histogram": h,
        "signal": sig,
        "macd_series": macd_line,
        "signal_series": signal_line,
        "histogram_series": histogram,
    }


def _bollinger(closes, period=20, std_mult=2):
    if len(closes) < period:
        return {"upper": 0, "middle": 0, "lower": 0, "width_pct": 0, "signal": "NEUTRAL"}
    
    mid = float(np.mean(closes[-period:]))
    std = float(np.std(closes[-period:]))
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width = (upper - lower) / mid * 100 if mid > 0 else 0
    
    price = float(closes[-1])
    if price > upper:
        sig = "OVERBOUGHT"
    elif price < lower:
        sig = "OVERSOLD"
    elif price > mid:
        sig = "BULLISH"
    else:
        sig = "BEARISH"
    
    return {"upper": upper, "middle": mid, "lower": lower, "width_pct": width, "signal": sig}


def _atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return None
    tr = np.maximum(
        highs[-period:] - lows[-period:],
        np.maximum(
            np.abs(highs[-period:] - np.roll(closes[-period:], 1)),
            np.abs(lows[-period:] - np.roll(closes[-period:], 1))
        )
    )[1:]
    return float(np.mean(tr))


def _volatility_regime(closes):
    if len(closes) < 30:
        return {"daily_pct": 0, "annualized_pct": 0, "regime": "UNKNOWN"}
    returns = np.diff(closes[-30:]) / closes[-30:-1]
    daily_vol = float(np.std(returns))
    annual_vol = daily_vol * np.sqrt(252)
    
    if annual_vol < 0.15:
        regime = "LOW"
    elif annual_vol < 0.30:
        regime = "MODERATE"
    elif annual_vol < 0.50:
        regime = "HIGH"
    else:
        regime = "EXTREME"
    
    return {"daily_pct": daily_vol * 100, "annualized_pct": annual_vol * 100, "regime": regime}


# ═══════════════════════════════════════════════════════
# CONFIDENCE, RISK, POSITION SIZING
# ═══════════════════════════════════════════════════════

def _compute_confidence(signal, bullish, bearish, rsi, vol_ratio, vol_regime):
    conf = 50
    agreement = max(bullish, bearish)
    if agreement >= 4: conf += 25
    elif agreement >= 3: conf += 15
    elif agreement <= 1: conf -= 15
    
    # RSI confirmation
    if signal in ("BUY", "STRONG_BUY") and rsi < 35: conf += 10
    elif signal in ("BUY", "STRONG_BUY") and rsi > 65: conf -= 10
    elif signal in ("SELL", "STRONG_SELL") and rsi > 65: conf += 10
    elif signal in ("SELL", "STRONG_SELL") and rsi < 35: conf -= 10
    
    # Volume
    if vol_ratio > 2.0: conf += 5
    elif vol_ratio < 0.5: conf -= 5
    
    # Volatility penalty
    if vol_regime == "EXTREME": conf -= 15
    elif vol_regime == "HIGH": conf -= 5
    elif vol_regime == "LOW": conf += 5
    
    if signal == "HOLD": conf = min(conf, 55)
    
    return max(0, min(100, conf))


def _classify_risk(vol_regime, rsi, confidence):
    risk_score = 0
    if vol_regime == "EXTREME": risk_score += 3
    elif vol_regime == "HIGH": risk_score += 2
    elif vol_regime == "MODERATE": risk_score += 1
    
    if rsi > 75 or rsi < 25: risk_score += 1
    if confidence < 40: risk_score += 1
    
    if risk_score >= 4: return "extreme"
    if risk_score >= 3: return "high"
    if risk_score >= 2: return "medium"
    return "low"


def _compute_position(price, atr, portfolio_value, risk_tolerance, signal):
    if not price or price <= 0 or not atr or atr <= 0:
        return {"entry_price": price, "stop_loss": None, "take_profit": None,
                "shares": 0, "risk_reward": "N/A", "position_value": 0}
    
    risk_pct = {"conservative": 0.01, "moderate": 0.02, "aggressive": 0.04}.get(risk_tolerance, 0.02)
    risk_amount = portfolio_value * risk_pct
    
    sl_mult = {"conservative": 1.5, "moderate": 2.0, "aggressive": 3.0}.get(risk_tolerance, 2.0)
    tp_mult = {"conservative": 2.0, "moderate": 3.0, "aggressive": 4.5}.get(risk_tolerance, 3.0)
    
    if signal in ("SELL", "STRONG_SELL"):
        # For sell signals, invert SL/TP
        sl_distance = atr * sl_mult
        tp_distance = atr * tp_mult
        stop_loss = round(price + sl_distance, 2)
        take_profit = round(price - tp_distance, 2)
    else:
        sl_distance = atr * sl_mult
        tp_distance = atr * tp_mult
        stop_loss = round(price - sl_distance, 2)
        take_profit = round(price + tp_distance, 2)
    
    shares = int(risk_amount / sl_distance) if sl_distance > 0 else 0
    position_value = shares * price
    rr = tp_distance / sl_distance if sl_distance > 0 else 0
    
    max_position = portfolio_value * 0.20
    if position_value > max_position:
        shares = int(max_position / price)
        position_value = shares * price
    
    return {
        "entry_price": round(price, 2),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "shares": shares,
        "risk_reward": f"1:{rr:.1f}",
        "position_value": round(position_value, 2),
        "position_pct": round(position_value / portfolio_value * 100, 1) if portfolio_value > 0 else 0,
        "risk_per_trade": f"{risk_pct*100:.0f}%",
    }


# ═══════════════════════════════════════════════════════
# EXPLAINABILITY ENGINE
# ═══════════════════════════════════════════════════════

def _build_reasons(votes, rsi, sma, macd, bb, vol_regime, vol_ratio, signal):
    """Build human-readable reasons for the signal."""
    reasons = []
    
    # RSI
    if rsi < 30:
        reasons.append(f"RSI oversold at {rsi:.1f} — potential bounce opportunity")
    elif rsi > 70:
        reasons.append(f"RSI overbought at {rsi:.1f} — potential pullback risk")
    elif "BULLISH" in votes["RSI"]:
        reasons.append(f"RSI at {rsi:.1f} shows bullish momentum")
    elif "BEARISH" in votes["RSI"]:
        reasons.append(f"RSI at {rsi:.1f} shows bearish momentum")
    
    # SMA Crossover
    if "STRONG_BULLISH" in sma["signal"]:
        reasons.append(f"Golden cross confirmed — {sma['detail']}")
    elif "BULLISH" in sma["signal"]:
        reasons.append(f"Uptrend active — {sma['detail']}")
    elif "STRONG_BEARISH" in sma["signal"]:
        reasons.append(f"Death cross confirmed — {sma['detail']}")
    elif "BEARISH" in sma["signal"]:
        reasons.append(f"Downtrend active — {sma['detail']}")
    
    # MACD
    if "CROSSOVER" in macd["signal"]:
        direction = "bullish" if "BULLISH" in macd["signal"] else "bearish"
        reasons.append(f"MACD {direction} crossover detected — momentum shift")
    elif "BULLISH" in macd["signal"]:
        reasons.append(f"MACD positive — bullish momentum continues")
    elif "BEARISH" in macd["signal"]:
        reasons.append(f"MACD negative — bearish momentum continues")
    
    # Bollinger
    if bb["signal"] == "OVERSOLD":
        reasons.append(f"Price at lower Bollinger Band — potential mean reversion up")
    elif bb["signal"] == "OVERBOUGHT":
        reasons.append(f"Price at upper Bollinger Band — potential mean reversion down")
    
    # Volume
    if vol_ratio > 2.0:
        reasons.append(f"High volume ({vol_ratio:.1f}x avg) confirms the move")
    elif vol_ratio < 0.5:
        reasons.append(f"Low volume ({vol_ratio:.1f}x avg) — signal may be weak")
    
    # Volatility filter
    if vol_regime == "EXTREME":
        reasons.append(f"⚠️ Extreme volatility — signals less reliable, BUY signals filtered out")
    elif vol_regime == "HIGH":
        reasons.append(f"⚠️ High volatility — wider stops recommended")
    
    if not reasons:
        reasons.append("Mixed signals — no clear directional bias")
    
    return reasons


def _empty_signal(ticker, error_msg):
    """Return an empty signal when data is unavailable."""
    from core.data_ingest import get_ticker_name
    return {
        "ticker": ticker.upper(),
        "company_name": get_ticker_name(ticker),
        "date": datetime.date.today().isoformat(),
        "data_source": "none",
        "signal": "HOLD",
        "confidence": 0,
        "risk": "high",
        "reason": error_msg,
        "reasons": [error_msg],
        "price": 0,
        "change_pct": 0,
        "indicators": {
            "rsi": {"value": 50, "signal": "NEUTRAL"},
            "sma_crossover": {"sma20": None, "sma50": None, "signal": "NEUTRAL", "detail": "No data"},
            "macd": {"macd_line": 0, "signal_line": 0, "histogram": 0, "signal": "NEUTRAL"},
            "bollinger": {"upper": 0, "middle": 0, "lower": 0, "width_pct": 0, "signal": "NEUTRAL"},
            "atr": None,
            "volume_ratio": 1.0,
            "volatility": {"daily_pct": 0, "annualized_pct": 0, "regime": "UNKNOWN"},
        },
        "votes": {},
        "bullish_count": 0,
        "bearish_count": 0,
        "trade": {"entry_price": 0, "stop_loss": None, "take_profit": None,
                  "shares": 0, "risk_reward": "N/A", "position_value": 0,
                  "position_pct": 0, "risk_per_trade": "0%"},
        "explainability": {"thesis": error_msg, "supporting_evidence": "N/A",
                          "risk_assessment": "N/A", "why_this_trade": error_msg,
                          "indicator_breakdown": {}, "key_factors": []},
    }


def _build_explanation(ticker, signal, reasons, votes, trade, risk):
    """Build the 'Why does this trade make sense?' explanation."""
    
    bullish_indicators = [k for k, v in votes.items() if "BULLISH" in v or v == "OVERSOLD"]
    bearish_indicators = [k for k, v in votes.items() if "BEARISH" in v or v == "OVERBOUGHT"]
    
    if signal in ("BUY", "STRONG_BUY"):
        thesis = f"{ticker} shows a buying opportunity"
        supporting = f"{len(bullish_indicators)}/5 indicators are bullish ({', '.join(bullish_indicators)})"
        risk_note = f"Risk is {risk}. Stop-loss at ${trade.get('stop_loss', 'N/A')} limits downside."
    elif signal in ("SELL", "STRONG_SELL"):
        thesis = f"{ticker} shows selling pressure"
        supporting = f"{len(bearish_indicators)}/5 indicators are bearish ({', '.join(bearish_indicators)})"
        risk_note = f"Risk is {risk}. Consider exiting or hedging positions."
    else:
        thesis = f"{ticker} shows no clear direction"
        supporting = f"Indicators are split: {len(bullish_indicators)} bullish, {len(bearish_indicators)} bearish"
        risk_note = "Wait for a clearer signal before entering."
    
    return {
        "thesis": thesis,
        "supporting_evidence": supporting,
        "risk_assessment": risk_note,
        "why_this_trade": f"{thesis}. {supporting}. {risk_note}",
        "indicator_breakdown": {k: v for k, v in votes.items()},
        "key_factors": reasons[:3],
    }
