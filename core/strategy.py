"""
📊 Strategy Layer — Rule-based signals + ML signal confidence scorer
======================================================================
Signals come from deterministic rules (indicators), NOT from LLM.
ML model scores signal confidence based on historical accuracy.
LLM is used ONLY to explain/narrate the signal — never to generate it.
"""

import json
import logging
import datetime
import numpy as np

logger = logging.getLogger("trade_bot.strategy")


def generate_signal(ticker: str, portfolio_value: float = 100000,
                     risk_tolerance: str = "moderate",
                     period: str = "3mo") -> dict:
    """
    Generate a complete trading signal. Deterministic — no LLM involved.
    
    Pipeline:
      1. Feature extraction (core/features.py)
      2. Rule-based signal (indicator voting)
      3. Risk/position sizing
      4. Confidence scoring (feature quality + agreement strength)
    
    Returns structured dict ready for LLM explanation or direct display.
    """
    from core.features import compute_features

    features = compute_features(ticker, period)

    if not features.get("features_available", False):
        return {
            "ticker": ticker, "signal": "HOLD", "confidence": 0,
            "error": features.get("error", "No data"),
            "source": features.get("source", "fallback"),
            **features,
        }

    # ── Signal from indicators (already computed in features) ──
    signal = features["signal"]
    bullish = features.get("bullish_count", 0)
    bearish = features.get("bearish_count", 0)

    # ── Confidence scoring ──
    confidence = _compute_confidence(features, signal, bullish, bearish)

    # ── Position sizing ──
    position = _compute_position(features, portfolio_value, risk_tolerance)

    # ── Assemble result ──
    result = {
        "ticker": ticker.upper(),
        "date": datetime.date.today().isoformat(),
        "signal": signal,
        "confidence": confidence,
        "source": "strategy_engine",
        "signal_method": "rule_based_indicator_vote",

        # Price data
        "price": features["price"],
        "change_pct": features["change_pct"],

        # Indicators
        "rsi": features.get("rsi"),
        "rsi_signal": features.get("rsi_signal"),
        "macd": features.get("macd"),
        "macd_trend": features.get("macd_trend"),
        "sma_signal": features.get("sma_signal"),
        "bb_signal": features.get("bb_signal"),
        "volume_ratio": features.get("volume_ratio"),
        "volatility_regime": features.get("volatility_regime"),

        # Agreement
        "bullish_count": bullish,
        "bearish_count": bearish,

        # Position
        **position,

        # Metadata
        "data_source": features.get("source"),
        "data_points": features.get("data_points"),
    }
    return result


def _compute_confidence(features: dict, signal: str, bullish: int, bearish: int) -> int:
    """
    Score confidence 0-100 based on:
      - Indicator agreement (how many agree)
      - RSI extremes (more extreme = higher confidence)
      - Volume confirmation
      - Volatility regime
    """
    conf = 50  # Start at neutral

    # Indicator agreement: strongest factor
    agreement = max(bullish, bearish)
    if agreement >= 4:
        conf += 25
    elif agreement >= 3:
        conf += 15
    elif agreement <= 1:
        conf -= 15

    # RSI confirmation
    rsi = features.get("rsi", 50)
    if signal in ("BUY", "STRONG_BUY") and rsi < 35:
        conf += 10  # Oversold confirms buy
    elif signal in ("BUY", "STRONG_BUY") and rsi > 65:
        conf -= 10  # Overbought contradicts buy
    elif signal in ("SELL", "STRONG_SELL") and rsi > 65:
        conf += 10  # Overbought confirms sell
    elif signal in ("SELL", "STRONG_SELL") and rsi < 35:
        conf -= 10  # Oversold contradicts sell

    # Volume confirmation
    vol_ratio = features.get("volume_ratio", 1.0)
    if vol_ratio > 2.0:
        conf += 5  # High volume confirms move
    elif vol_ratio < 0.5:
        conf -= 5  # Low volume = weak signal

    # Volatility penalty
    vol_regime = features.get("volatility_regime", "MODERATE")
    if vol_regime == "EXTREME":
        conf -= 15
    elif vol_regime == "HIGH":
        conf -= 5
    elif vol_regime == "LOW":
        conf += 5

    # HOLD penalty (shouldn't be high confidence)
    if signal == "HOLD":
        conf = min(conf, 55)

    return max(0, min(100, conf))


def _compute_position(features: dict, portfolio_value: float, risk_tolerance: str) -> dict:
    """Compute position sizing, SL, TP. Pure math, no LLM."""
    price = features.get("price", 0)
    atr = features.get("atr")

    if not price or price <= 0 or not atr or atr <= 0:
        return {
            "entry_price": price, "stop_loss": None, "take_profit": None,
            "shares": 0, "risk_reward": "N/A", "position_value": 0,
        }

    risk_pct = {"conservative": 0.01, "moderate": 0.02, "aggressive": 0.04}.get(risk_tolerance, 0.02)
    risk_amount = portfolio_value * risk_pct

    sl_mult = {"conservative": 1.5, "moderate": 2.0, "aggressive": 3.0}.get(risk_tolerance, 2.0)
    tp_mult = {"conservative": 2.0, "moderate": 3.0, "aggressive": 4.5}.get(risk_tolerance, 3.0)

    sl_distance = atr * sl_mult
    tp_distance = atr * tp_mult
    stop_loss = round(price - sl_distance, 2)
    take_profit = round(price + tp_distance, 2)

    shares = int(risk_amount / sl_distance) if sl_distance > 0 else 0
    position_value = shares * price
    rr = tp_distance / sl_distance if sl_distance > 0 else 0

    # Hard limit: never exceed 20% of portfolio
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


def signal_to_json(signal_dict: dict) -> str:
    """Serialize signal to JSON string."""
    return json.dumps(signal_dict, indent=2, default=str)
