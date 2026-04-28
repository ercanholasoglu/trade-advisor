"""Tool: Risk hesaplama - VaR, position sizing, stop-loss."""

from smolagents import tool


@tool
def calculate_risk_metrics(ticker: str, portfolio_value: float, risk_tolerance: str = "moderate") -> str:
    """
    Calculates risk metrics and position sizing for a potential trade.
    Computes Value at Risk (VaR), optimal position size, suggested stop-loss
    and take-profit levels based on historical volatility and risk tolerance.

    Args:
        ticker: Stock/crypto ticker symbol (e.g. 'AAPL', 'BTC-USD')
        portfolio_value: Total portfolio value in USD (e.g. 100000.0)
        risk_tolerance: Risk profile - 'conservative', 'moderate', or 'aggressive'. Default 'moderate'.

    Returns:
        JSON string with risk metrics, position sizing, stop-loss/take-profit levels.
    """
    import yfinance as yf
    import json
    import numpy as np

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo", interval="1d")
        if hist.empty or len(hist) < 30:
            return json.dumps({"error": f"Insufficient data for {ticker}", "ticker": ticker})

        closes = hist["Close"].values
        returns = np.diff(closes) / closes[:-1]
        current_price = float(closes[-1])

        daily_volatility = float(np.std(returns))
        annualized_volatility = daily_volatility * np.sqrt(252)
        avg_daily_return = float(np.mean(returns))
        sharpe_approx = (avg_daily_return / daily_volatility * np.sqrt(252)) if daily_volatility > 0 else 0

        peak = np.maximum.accumulate(closes)
        drawdowns = (closes - peak) / peak
        max_drawdown = float(np.min(drawdowns))

        confidence_levels = {"conservative": 0.99, "moderate": 0.95, "aggressive": 0.90}
        confidence = confidence_levels.get(risk_tolerance, 0.95)
        z_scores = {0.99: 2.326, 0.95: 1.645, 0.90: 1.282}
        z = z_scores[confidence]
        daily_var = daily_volatility * z
        annual_var = annualized_volatility * z

        risk_per_trade_pct = {"conservative": 0.01, "moderate": 0.02, "aggressive": 0.04}
        risk_pct = risk_per_trade_pct.get(risk_tolerance, 0.02)
        risk_amount = portfolio_value * risk_pct

        highs = hist["High"].values[-14:]
        lows = hist["Low"].values[-14:]
        closes_14 = closes[-14:]
        tr = np.maximum(highs - lows, np.maximum(np.abs(highs - np.roll(closes_14, 1)), np.abs(lows - np.roll(closes_14, 1))))[1:]
        atr = float(np.mean(tr))

        atr_multipliers = {
            "conservative": {"sl": 1.5, "tp": 2.0},
            "moderate": {"sl": 2.0, "tp": 3.0},
            "aggressive": {"sl": 3.0, "tp": 4.5},
        }
        multipliers = atr_multipliers.get(risk_tolerance, atr_multipliers["moderate"])
        stop_loss_distance = atr * multipliers["sl"]
        take_profit_distance = atr * multipliers["tp"]
        stop_loss_price = round(current_price - stop_loss_distance, 2)
        take_profit_price = round(current_price + take_profit_distance, 2)

        if stop_loss_distance > 0:
            shares = int(risk_amount / stop_loss_distance)
            position_value = shares * current_price
            position_pct = (position_value / portfolio_value) * 100
        else:
            shares = position_value = position_pct = 0

        rr_ratio = take_profit_distance / stop_loss_distance if stop_loss_distance > 0 else 0

        if annualized_volatility < 0.15: vol_regime = "LOW"
        elif annualized_volatility < 0.30: vol_regime = "MODERATE"
        elif annualized_volatility < 0.50: vol_regime = "HIGH"
        else: vol_regime = "EXTREME"

        result = {
            "ticker": ticker.upper(), "current_price": round(current_price, 2),
            "risk_tolerance": risk_tolerance, "portfolio_value": portfolio_value,
            "volatility": {"daily": f"{daily_volatility*100:.2f}%", "annualized": f"{annualized_volatility*100:.1f}%", "regime": vol_regime, "atr_14": round(atr, 2)},
            "risk_metrics": {"var_daily": f"{daily_var*100:.2f}%", "var_daily_usd": round(daily_var * current_price, 2), "var_annual": f"{annual_var*100:.1f}%", "max_drawdown_6mo": f"{max_drawdown*100:.1f}%", "sharpe_ratio_approx": round(sharpe_approx, 2), "confidence_level": f"{confidence*100:.0f}%"},
            "position_sizing": {"risk_per_trade": f"{risk_pct*100:.0f}%", "risk_amount_usd": round(risk_amount, 2), "suggested_shares": shares, "position_value_usd": round(position_value, 2), "position_pct_of_portfolio": f"{position_pct:.1f}%"},
            "trade_levels": {"entry_price": round(current_price, 2), "stop_loss": stop_loss_price, "stop_loss_pct": f"{(stop_loss_distance/current_price)*100:.1f}%", "take_profit": take_profit_price, "take_profit_pct": f"{(take_profit_distance/current_price)*100:.1f}%", "risk_reward_ratio": f"1:{rr_ratio:.1f}"},
            "recommendations": [],
        }

        recs = result["recommendations"]
        if vol_regime == "EXTREME": recs.append("⚠️ EXTREME volatility - consider reducing position size by 50%")
        if max_drawdown < -0.20: recs.append("⚠️ Significant drawdown in last 6 months - exercise caution")
        if rr_ratio < 1.5: recs.append("⚠️ Risk/Reward ratio below 1.5 - consider wider take-profit")
        if position_pct > 20: recs.append("⚠️ Position exceeds 20% of portfolio - consider reducing")
        if sharpe_approx > 1: recs.append("✅ Good risk-adjusted returns historically (Sharpe > 1)")
        if vol_regime == "LOW": recs.append("✅ Low volatility environment - favorable for entry")

        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})
