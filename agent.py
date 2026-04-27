"""
🤖 Agentic Trade Bot Advisor
=============================
TradingAgents paper (arxiv: 2412.20138) mimarisinden ilham alan
multi-agent trade advisor sistemi.

Mimari (6 Agent):
  - Technical Analyst Agent (RSI, MACD, Bollinger, SMA)
  - News & Sentiment Analyst Agent (haber + sentiment)
  - Fundamental Analyst Agent (P/E, earnings, financials)
  - BIST & Turkey Analyst Agent (BIST hisseler, altın, gümüş, KAP)
  - Risk Manager Agent (VaR, position sizing, stop-loss)
  - Fund Manager Agent (orchestrator - tüm raporları alır, final karar)

Framework: smolagents (HuggingFace)
"""

import os
import json
from smolagents import (
    CodeAgent,
    ToolCallingAgent,
    InferenceClientModel,
)

# --- Tool imports ---
from tools.price_history import get_price_history
from tools.technical_indicators import get_technical_indicators
from tools.news_sentiment import get_news_sentiment
from tools.fundamental_analysis import get_fundamental_data
from tools.risk_calculator import calculate_risk_metrics
from tools.market_overview import get_market_overview
from tools.bist_scanner import get_bist_scanner
from tools.kap_disclosures import get_kap_disclosures
from tools.daily_dashboard import get_daily_dashboard


def create_trade_advisor(hf_token: str | None = None, model_id: str = "Qwen/Qwen2.5-72B-Instruct"):
    token = hf_token or os.environ.get("HF_TOKEN", "")
    model = InferenceClientModel(model_id=model_id, token=token)

    # AGENT 1: Technical Analyst
    technical_analyst = ToolCallingAgent(
        tools=[get_price_history, get_technical_indicators],
        model=model, name="technical_analyst",
        description="Expert technical analyst — RSI, MACD, Bollinger Bands, SMA, EMA, ATR, Stochastic for any stock/crypto. Supports BIST stocks with .IS suffix (e.g. THYAO.IS) and US stocks (AAPL, NVDA).",
        max_steps=6,
        instructions="You are a professional technical analyst. Get price history then compute indicators. Report: trend direction, support/resistance, momentum, volatility, overall signal (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL). Be precise with numbers.",
    )

    # AGENT 2: News & Sentiment Analyst
    sentiment_analyst = ToolCallingAgent(
        tools=[get_news_sentiment, get_kap_disclosures],
        model=model, name="sentiment_analyst",
        description="Analyzes news sentiment and KAP disclosures. For BIST stocks, also checks KAP (Turkish Public Disclosure Platform) for company info and recent regulatory filings.",
        max_steps=5,
        instructions="You are a sentiment analyst. For US stocks: search news. For BIST/Turkish stocks: ALSO check KAP disclosures using get_kap_disclosures. Report: overall sentiment, key catalysts, risks, upcoming events, confidence level.",
    )

    # AGENT 3: Fundamental Analyst
    fundamental_analyst = ToolCallingAgent(
        tools=[get_fundamental_data],
        model=model, name="fundamental_analyst",
        description="Evaluates company fundamentals: P/E, P/B, PEG, margins, growth, ROE, analyst targets. Works for both US and BIST stocks.",
        max_steps=4,
        instructions="You are a fundamental analyst. Evaluate: valuation (P/E, PEG), growth, profitability, financial health, analyst consensus. Rate as UNDERVALUED/FAIR/OVERVALUED.",
    )

    # AGENT 4: BIST & Turkey Market Analyst (YENİ!)
    bist_analyst = ToolCallingAgent(
        tools=[get_bist_scanner, get_daily_dashboard],
        model=model, name="bist_analyst",
        description="Specialist in Turkish markets (Borsa Istanbul). Scans BIST30/banks/tech stocks, provides gold/silver prices in TRY, USD/TRY rates, BIST indices, and generates daily buy/sell dashboards covering both BIST and US markets.",
        max_steps=5,
        instructions="You are a Turkish market specialist. Use get_bist_scanner for BIST stock scans (bist30, bist_banks, commodities_try, indices). Use get_daily_dashboard for comprehensive daily AL/SAT recommendations. Report in Turkish when analyzing BIST stocks. Always include altın (gold) and USD/TRY rates.",
    )

    # AGENT 5: Risk Manager
    risk_manager = ToolCallingAgent(
        tools=[calculate_risk_metrics, get_market_overview],
        model=model, name="risk_manager",
        description="Assesses risk, position sizing, stop-loss/take-profit. Evaluates VIX, sector rotation, macro trends. Works for both US and BIST stocks.",
        max_steps=6,
        instructions="You are a senior risk manager. Check market conditions first, then calculate risk metrics. Report: market environment, volatility, position sizing, entry/exit levels, R/R ratio, key risks, verdict (APPROVE/CAUTION/REJECT). Capital preservation is #1.",
    )

    # FUND MANAGER (Master Orchestrator)
    fund_manager = CodeAgent(
        tools=[],
        model=model,
        managed_agents=[technical_analyst, sentiment_analyst, fundamental_analyst, bist_analyst, risk_manager],
        additional_authorized_imports=["json", "datetime"],
        planning_interval=3, max_steps=18,
        instructions=FUND_MANAGER_INSTRUCTIONS,
    )
    return fund_manager


FUND_MANAGER_INSTRUCTIONS = """
You are a senior fund manager coordinating 5 specialist agents for trade recommendations.
You support both US (NASDAQ) and Turkish (BIST) markets, plus gold/silver/forex.

## WORKFLOW:
1. Call technical_analyst with the ticker (for BIST: use THYAO.IS format)
2. Call sentiment_analyst with ticker and company name (for BIST: also checks KAP)
3. Call fundamental_analyst with the ticker
4. For BIST/Turkish stocks: Call bist_analyst for Turkish market context (indices, gold, USD/TRY)
5. Call risk_manager with ticker, portfolio value, and risk tolerance
6. Synthesize ALL reports into final recommendation

## BIST TICKER FORMAT:
- BIST stocks use .IS suffix: THYAO.IS, GARAN.IS, AKBNK.IS
- For KAP and bist_scanner: use without suffix: THYAO, GARAN
- US stocks: no suffix: AAPL, NVDA, MSFT

## OUTPUT FORMAT:
═══════════════════════════════════════════════════
🤖 TRADE ADVISOR REPORT
═══════════════════════════════════════════════════
📊 TICKER: [SYMBOL] | COMPANY: [NAME]
📅 DATE: [date] | 💵 USD/TRY: [rate] | 🥇 Altın: ₺[x]/gram

📈 SIGNAL: [BUY / SELL / HOLD] | 🎯 CONFIDENCE: [0-100]%

📋 TECHNICAL: [summary with RSI, MACD, trend]
📰 SENTIMENT: [summary with news + KAP if BIST]
📊 FUNDAMENTAL: [valuation + key metrics]
🇹🇷 BIST CONTEXT: [if applicable — index, sector, gold impact]
⚠️ RISK: [vol, VIX, verdict]

💰 TRADE: Entry $[x] | SL $[x] | TP $[x] | [N] shares | R/R 1:[x]
📝 REASONING: [2-3 sentences]
⚠️ DISCLAIMER: AI analysis, not financial advice.
═══════════════════════════════════════════════════

## RULES:
- Consult ALL relevant agents (skip bist_analyst for pure US stocks)
- For BIST: always include gold price in TRY and USD/TRY rate
- VIX > 30 → recommend HOLD
- Confidence < 50% → recommend HOLD
- Always include stop-loss and take-profit
"""


def run_analysis(query: str, hf_token: str | None = None, model_id: str = "Qwen/Qwen2.5-72B-Instruct"):
    advisor = create_trade_advisor(hf_token=hf_token, model_id=model_id)
    return advisor.run(query)


if __name__ == "__main__":
    print(run_analysis("Analyze NVDA. Portfolio: $50K, moderate risk."))
