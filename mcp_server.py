"""
🔌 MCP Server — Trade Bot Advisor Tools
==========================================
Exposes all trading agents as MCP (Model Context Protocol) tools.
Any MCP client (Claude Desktop, smolagents, Cursor, etc.) can connect.

Tools:
  - analyze_technical: RSI, SMA, MACD, Bollinger, Volume analysis
  - analyze_sentiment: Market mood from price momentum + volume
  - assess_risk: Volatility, drawdown, position risk evaluation
  - multi_agent_decision: Full 4-agent orchestrated analysis
  - run_backtest: Walk-forward strategy backtesting
  - ground_truth_eval: Model predictions vs actual outcomes
  - optimize_strategy: Dynamic indicator weight optimization
  - advanced_reasoning: Contextual analysis (traps, regime, macro)
  - generate_signal: Core signal engine (deterministic)
  - portfolio_execute: Execute signal in paper portfolio

Transports: stdio (Claude Desktop), sse, streamable-http
Usage:
  python mcp_server.py                    # streamable-http on :8000
  python mcp_server.py stdio              # for Claude Desktop
  python mcp_server.py sse                # SSE transport
"""

import sys
import json
from mcp.server.fastmcp import FastMCP
from typing import Annotated
from pydantic import Field

mcp = FastMCP(
    name="trade-bot-advisor",
    instructions=(
        "Trading signal analysis system. Uses rule-based technical indicators "
        "(RSI, SMA crossover, MACD, Bollinger, Volume) with multi-agent architecture. "
        "LLM never generates signals — only interprets them. "
        "All tools return structured JSON. Supported tickers: US stocks (AAPL), "
        "BIST (THYAO.IS), Crypto (BTC-USD), Commodities (GC=F)."
    ),
    host="0.0.0.0",
    port=8000,
)


# ═══════════════════════════════════════════════════════
# TOOL 1: Technical Analysis
# ═══════════════════════════════════════════════════════

@mcp.tool()
def analyze_technical(
    ticker: Annotated[str, Field(description="Ticker symbol (e.g. AAPL, THYAO.IS, BTC-USD)")],
    portfolio_value: Annotated[float, Field(description="Portfolio value in USD", gt=0)] = 100000,
    risk_tolerance: Annotated[str, Field(description="conservative, moderate, or aggressive")] = "moderate",
) -> dict:
    """
    Run full technical analysis on a ticker. Returns signal (BUY/SELL/HOLD),
    confidence (0-100), risk level, indicators (RSI, SMA, MACD, Bollinger),
    trade parameters (entry, SL, TP, shares), and explainability.
    """
    from core.signal_engine import generate_signal
    sig = generate_signal(ticker, portfolio_value, risk_tolerance)
    # Remove numpy arrays that aren't JSON serializable
    ind = sig.get("indicators", {})
    for key in ["macd"]:
        if key in ind:
            for sub in ["macd_series", "signal_series", "histogram_series"]:
                ind[key].pop(sub, None)
    return sig


# ═══════════════════════════════════════════════════════
# TOOL 2: Sentiment Analysis
# ═══════════════════════════════════════════════════════

@mcp.tool()
def analyze_sentiment(
    ticker: Annotated[str, Field(description="Ticker symbol")],
) -> dict:
    """
    Analyze market sentiment for a ticker via price momentum (5d, 20d),
    volume trend, and volatility expansion. Returns BULLISH/BEARISH/NEUTRAL
    with confidence score and reasoning.
    """
    from core.agents import SentimentAgent
    agent = SentimentAgent()
    report = agent.analyze(ticker)
    return report.to_dict()


# ═══════════════════════════════════════════════════════
# TOOL 3: Risk Assessment
# ═══════════════════════════════════════════════════════

@mcp.tool()
def assess_risk(
    ticker: Annotated[str, Field(description="Ticker symbol")],
    portfolio_value: Annotated[float, Field(description="Portfolio value USD", gt=0)] = 100000,
    risk_tolerance: Annotated[str, Field(description="conservative, moderate, aggressive")] = "moderate",
) -> dict:
    """
    Evaluate risk for a ticker: volatility regime, drawdown, ATR gap risk,
    trend strength. Returns verdict: APPROVE (safe), CAUTION (reduce size),
    or REJECT (don't trade). Includes risk score and detailed reasoning.
    """
    from core.agents import RiskAgent
    agent = RiskAgent()
    report = agent.analyze(ticker, portfolio_value, risk_tolerance)
    return report.to_dict()


# ═══════════════════════════════════════════════════════
# TOOL 4: Multi-Agent Decision
# ═══════════════════════════════════════════════════════

@mcp.tool()
def multi_agent_decision(
    ticker: Annotated[str, Field(description="Ticker symbol")],
    portfolio_value: Annotated[float, Field(description="Portfolio value USD", gt=0)] = 100000,
    risk_tolerance: Annotated[str, Field(description="conservative, moderate, aggressive")] = "moderate",
    weight_analyst: Annotated[float, Field(description="Weight for Analyst agent (0-1)", ge=0, le=1)] = 0.50,
    weight_sentiment: Annotated[float, Field(description="Weight for Sentiment agent (0-1)", ge=0, le=1)] = 0.25,
    weight_risk: Annotated[float, Field(description="Weight for Risk agent (0-1)", ge=0, le=1)] = 0.25,
) -> dict:
    """
    Run the full 4-agent analysis: Analyst + Sentiment + Risk agents independently
    analyze the ticker, then the Orchestrator synthesizes with weighted scoring.
    Risk Agent has veto power (REJECT overrides BUY).
    Returns per-agent reports, reasoning chain, and final signal.
    """
    from core.agents import run_multi_agent_analysis
    weights = {"Analyst": weight_analyst, "Sentiment": weight_sentiment, "Risk": weight_risk}
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    result = run_multi_agent_analysis(ticker, portfolio_value, risk_tolerance, weights)
    # Clean non-serializable data
    for name, agent_data in result.get("agents", {}).items():
        if "data" in agent_data and "full_signal" in agent_data["data"]:
            fs = agent_data["data"]["full_signal"]
            for key in ["indicators"]:
                if key in fs:
                    for sub_key in list(fs[key].get("macd", {}).keys()):
                        if "series" in sub_key:
                            del fs[key]["macd"][sub_key]
    return result


# ═══════════════════════════════════════════════════════
# TOOL 5: Backtesting
# ═══════════════════════════════════════════════════════

@mcp.tool()
def run_backtest(
    ticker: Annotated[str, Field(description="Ticker symbol")],
    period_years: Annotated[float, Field(description="Backtest period in years", ge=0.5, le=5)] = 1,
    holding_days: Annotated[int, Field(description="Days to hold after signal", ge=1, le=30)] = 10,
    rsi_buy: Annotated[float, Field(description="RSI buy threshold", ge=15, le=45)] = 35,
    rsi_sell: Annotated[float, Field(description="RSI sell threshold", ge=55, le=85)] = 65,
) -> dict:
    """
    Backtest the RSI+SMA signal strategy on historical data.
    Returns accuracy, total return, Sharpe ratio, max drawdown,
    profit factor, equity curve, and buy & hold comparison.
    """
    from core.backtester import run_backtest as _bt
    result = _bt(ticker, period_years=period_years, holding_days=holding_days,
                 rsi_buy=rsi_buy, rsi_sell=rsi_sell)
    return result


# ═══════════════════════════════════════════════════════
# TOOL 6: Ground Truth Evaluation
# ═══════════════════════════════════════════════════════

@mcp.tool()
def ground_truth_eval(
    ticker: Annotated[str, Field(description="Ticker symbol")],
    period_years: Annotated[float, Field(description="Evaluation period in years", ge=0.5, le=5)] = 2,
) -> dict:
    """
    Compare historical signal predictions vs actual price outcomes.
    For each signal, shows what the model predicted and what actually happened
    at 5, 10, and 20 day horizons. Returns per-signal verdicts and accuracy stats.
    The definitive answer to: 'Is this signal engine actually good?'
    """
    from core.evaluator import run_ground_truth_evaluation
    result = run_ground_truth_evaluation(ticker, period_years=period_years, eval_horizons=[5, 10, 20])
    # Limit evaluations to last 50 for response size
    if "evaluations" in result:
        result["evaluations"] = result["evaluations"][-50:]
    return result


# ═══════════════════════════════════════════════════════
# TOOL 7: Strategy Optimization
# ═══════════════════════════════════════════════════════

@mcp.tool()
def optimize_strategy(
    ticker: Annotated[str, Field(description="Ticker symbol (or comma-separated for multi)")],
    period_years: Annotated[float, Field(description="Analysis period in years", ge=0.5, le=5)] = 2,
) -> dict:
    """
    Optimize indicator weights based on historical accuracy.
    Walks through past data, measures each indicator's individual accuracy,
    and produces optimized weights. Better-performing indicators get more influence.
    Returns: per-indicator accuracy, default vs optimized weights, ranking.
    """
    from core.optimizer import compute_indicator_accuracy, optimize_for_tickers
    tickers = [t.strip().upper() for t in ticker.split(",") if t.strip()]
    if len(tickers) == 1:
        return compute_indicator_accuracy(tickers[0], period_years)
    else:
        return optimize_for_tickers(tickers, period_years)


# ═══════════════════════════════════════════════════════
# TOOL 8: Advanced Reasoning
# ═══════════════════════════════════════════════════════

@mcp.tool()
def advanced_reasoning(
    ticker: Annotated[str, Field(description="Ticker symbol")],
) -> dict:
    """
    Deep contextual analysis beyond basic indicators:
    - Market regime detection (trending/ranging/volatile)
    - Liquidity trap detection (bull/bear traps)
    - RSI-price divergence analysis
    - Macro context awareness (risk-on/off, inflation, policy)
    - Buying/selling exhaustion detection
    Adjusts signal confidence based on contextual factors.
    """
    from core.reasoning import generate_advanced_reasoning
    return generate_advanced_reasoning(ticker)


# ═══════════════════════════════════════════════════════
# TOOL 9: Quick Signal
# ═══════════════════════════════════════════════════════

@mcp.tool()
def quick_signal(
    ticker: Annotated[str, Field(description="Ticker symbol")],
) -> dict:
    """
    Get a quick trading signal for a ticker. Returns just the essentials:
    signal (BUY/SELL/HOLD), confidence, risk level, price, and top reason.
    Use analyze_technical for full details.
    """
    from core.signal_engine import generate_signal
    sig = generate_signal(ticker)
    return {
        "ticker": sig["ticker"],
        "signal": sig["signal"],
        "confidence": sig["confidence"],
        "risk": sig["risk"],
        "price": sig["price"],
        "reason": sig.get("reason", ""),
        "rsi": sig["indicators"]["rsi"]["value"],
        "sma": sig["indicators"]["sma_crossover"]["signal"],
        "macd": sig["indicators"]["macd"]["signal"],
    }


# ═══════════════════════════════════════════════════════
# TOOL 10: Compare Tickers
# ═══════════════════════════════════════════════════════

@mcp.tool()
def compare_tickers(
    tickers: Annotated[str, Field(description="Comma-separated ticker symbols (2-5)")],
) -> dict:
    """
    Compare 2-5 tickers side by side. Returns signal, confidence, risk,
    price, and key indicators for each ticker in a comparison table.
    """
    from core.signal_engine import generate_signal
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if len(ticker_list) < 2:
        return {"error": "Need at least 2 tickers"}
    results = {}
    for t in ticker_list[:5]:
        sig = generate_signal(t)
        results[t] = {
            "signal": sig["signal"],
            "confidence": sig["confidence"],
            "risk": sig["risk"],
            "price": sig["price"],
            "rsi": sig["indicators"]["rsi"]["value"],
            "sma": sig["indicators"]["sma_crossover"]["signal"],
            "volatility": sig["indicators"]["volatility"]["regime"],
        }
    return {"tickers": results}


# ═══════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    transport = sys.argv[1] if len(sys.argv) > 1 else "streamable-http"
    print(f"Starting Trade Bot MCP Server ({transport}) on 0.0.0.0:8000")
    print(f"Tools: {len(mcp._tool_manager._tools)} registered")
    mcp.run(transport)
