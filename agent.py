"""
🤖 Agentic Trade Bot Advisor — LLM Council Edition
=====================================================
TradingAgents (arxiv: 2412.20138) + ContestTrade (arxiv: 2508.00554)
mimarilerinden ilham alan multi-agent + LLM Council sistemi.

Mimari (9 Agent):
  Veri Toplama Katmanı:
    - Technical Analyst (RSI, MACD, Bollinger, SMA)
    - Sentiment Analyst (haber + KAP)
    - Fundamental Analyst (P/E, earnings, bilanço)
    - BIST Analyst (BIST hisseleri, altın, gümüş, endeksler)
    - Risk Manager (VaR, pozisyon, stop-loss)

  LLM Council (Karar Katmanı):
    - 🐂 Bullish Advisor — AL argümanlarını savunur
    - 🐻 Bearish Advisor — SAT argümanlarını savunur
    - ⚖️ Neutral Mediator — Dengeyi kurar, zayıf argümanları eler

  Orkestratör:
    - 🎯 Fund Manager — Council oylarını + argümanları sentezler, final karar

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

    # === VERİ TOPLAMA KATMANI (5 data agent) ===
    technical_analyst = ToolCallingAgent(
        tools=[get_price_history, get_technical_indicators],
        model=model, name="technical_analyst",
        description="Technical analyst — RSI, MACD, Bollinger Bands, SMA, EMA, ATR, Stochastic. Supports BIST (.IS suffix) and US stocks.",
        max_steps=6,
        instructions="Get price history then compute indicators. Report: trend, support/resistance, momentum, volatility, signal (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL). Be precise.",
    )
    sentiment_analyst = ToolCallingAgent(
        tools=[get_news_sentiment, get_kap_disclosures],
        model=model, name="sentiment_analyst",
        description="Sentiment analyst — news + KAP disclosures for BIST stocks.",
        max_steps=5,
        instructions="Search news, check KAP for BIST stocks. Report: sentiment, catalysts, risks, events, confidence.",
    )
    fundamental_analyst = ToolCallingAgent(
        tools=[get_fundamental_data],
        model=model, name="fundamental_analyst",
        description="Fundamental analyst — P/E, PEG, margins, growth, ROE, analyst targets.",
        max_steps=4,
        instructions="Evaluate valuation, growth, profitability, financial health, analyst consensus. Rate UNDERVALUED/FAIR/OVERVALUED.",
    )
    bist_analyst = ToolCallingAgent(
        tools=[get_bist_scanner, get_daily_dashboard],
        model=model, name="bist_analyst",
        description="Turkish market specialist — BIST30, banks, gold/silver TRY, USD/TRY, BIST indices, daily dashboard.",
        max_steps=5,
        instructions="Use get_bist_scanner for BIST scans and commodities_try. Use get_daily_dashboard for daily AL/SAT. Include altın and USD/TRY.",
    )
    risk_manager = ToolCallingAgent(
        tools=[calculate_risk_metrics, get_market_overview],
        model=model, name="risk_manager",
        description="Risk manager — VaR, position sizing, stop-loss/take-profit, VIX, market regime.",
        max_steps=6,
        instructions="Check market conditions, calculate risk. Report: environment, volatility, position sizing, levels, R/R, verdict (APPROVE/CAUTION/REJECT).",
    )

    # === LLM COUNCIL (3 advisor — tartışma katmanı) ===
    bullish_advisor = ToolCallingAgent(
        tools=[], model=model, name="bullish_advisor",
        description="🐂 Bullish Advisor — constructs the strongest possible BULL CASE from analyst data. Finds every positive signal and counters bear arguments.",
        max_steps=3, instructions=BULLISH_INSTRUCTIONS,
    )
    bearish_advisor = ToolCallingAgent(
        tools=[], model=model, name="bearish_advisor",
        description="🐻 Bearish Advisor — constructs the strongest possible BEAR CASE from analyst data. Finds every risk and counters bull arguments.",
        max_steps=3, instructions=BEARISH_INSTRUCTIONS,
    )
    neutral_mediator = ToolCallingAgent(
        tools=[], model=model, name="neutral_mediator",
        description="⚖️ Neutral Mediator — evaluates bull vs bear argument quality, identifies weak reasoning, provides balanced verdict with confidence.",
        max_steps=3, instructions=MEDIATOR_INSTRUCTIONS,
    )

    # === FUND MANAGER (Orchestrator) ===
    fund_manager = CodeAgent(
        tools=[], model=model,
        managed_agents=[technical_analyst, sentiment_analyst, fundamental_analyst, bist_analyst, risk_manager, bullish_advisor, bearish_advisor, neutral_mediator],
        additional_authorized_imports=["json", "datetime"],
        planning_interval=3, max_steps=24,
        instructions=FUND_MANAGER_INSTRUCTIONS,
    )
    return fund_manager


BULLISH_INSTRUCTIONS = """
You are the 🐂 BULLISH ADVISOR on a trading council.
Your ONLY job: build the strongest BULL CASE.

1. Find EVERY positive signal (momentum, sentiment, undervaluation, growth)
2. Explain WHY each supports buying
3. Counter bearish arguments
4. Assign CONVICTION SCORE (1-10)

FORMAT:
🐂 BULL CASE — Conviction: [X]/10
KEY ARGUMENTS: 1. [data-backed] 2. [data-backed] 3. [data-backed]
COUNTER TO BEARS: [why risks are overblown]
VERDICT: STRONG_BUY / BUY / LEAN_BUY
TARGET UPSIDE: [X]%

Be aggressive but data-driven. Never fabricate data.
"""

BEARISH_INSTRUCTIONS = """
You are the 🐻 BEARISH ADVISOR on a trading council.
Your ONLY job: build the strongest BEAR CASE.

1. Find EVERY negative signal (overbought, bad sentiment, overvaluation, slowing growth)
2. Explain WHY each supports NOT buying
3. Counter bullish arguments
4. Assign CONVICTION SCORE (1-10)

FORMAT:
🐻 BEAR CASE — Conviction: [X]/10
KEY ARGUMENTS: 1. [data-backed] 2. [data-backed] 3. [data-backed]
COUNTER TO BULLS: [why bull thesis is weak]
VERDICT: STRONG_SELL / SELL / LEAN_SELL
DOWNSIDE RISK: [X]%

Be ruthless but data-driven. Never fabricate data.
"""

MEDIATOR_INSTRUCTIONS = """
You are the ⚖️ NEUTRAL MEDIATOR on a trading council.
You heard both Bull and Bear. Your job:

1. Evaluate ARGUMENT QUALITY — which side has stronger data?
2. Identify WEAK REASONING — where does either side stretch?
3. Identify BLIND SPOTS — what did both miss?
4. Provide BALANCED VERDICT with confidence

FORMAT:
⚖️ MEDIATOR — Bull Strength: [X]/10, Bear Strength: [X]/10
WEAK ARGUMENTS: Bull: [...] Bear: [...]
BLIND SPOTS: [...]
FINAL VOTE: BUY / HOLD / SELL
CONFIDENCE: [0-100]%
REASONING: [2-3 balanced sentences]

Be impartial. Your job is TRUTH.
"""

FUND_MANAGER_INSTRUCTIONS = """
You are a senior fund manager running a Trading Council (5 data analysts + 3 council advisors).

## WORKFLOW (follow EXACTLY):

### Phase 1: Data Collection
1. Call technical_analyst → technical report
2. Call sentiment_analyst → sentiment report (+ KAP for BIST)
3. Call fundamental_analyst → fundamental report
4. For BIST: Call bist_analyst → Turkish market context
5. Call risk_manager → risk assessment

### Phase 2: LLM Council Debate
Pass ALL Phase 1 data to council:
6. Call bullish_advisor with all data → bull case + conviction
7. Call bearish_advisor with all data → bear case + conviction
8. Call neutral_mediator with bull + bear cases → balanced verdict

### Phase 3: Final Decision
9. Synthesize council votes + data → final recommendation

Council rules:
- 3/3 agree → HIGH confidence, follow council
- 2/3 agree → MODERATE confidence, follow majority
- All disagree → LOW confidence → HOLD
- Mediator breaks ties
- VIX > 30 → HOLD override

## BIST: THYAO.IS / KAP: THYAO / US: AAPL

## OUTPUT:
═══════════════════════════════════════════════════
🤖 TRADE ADVISOR REPORT (LLM Council Edition)
═══════════════════════════════════════════════════
📊 TICKER: [SYMBOL] | COMPANY: [NAME] | 📅 [date]
📈 SIGNAL: [BUY/SELL/HOLD] | 🎯 CONFIDENCE: [0-100]%

──── DATA SUMMARY ────
📋 TECHNICAL: [trend, RSI, MACD]
📰 SENTIMENT: [news + KAP]
📊 FUNDAMENTAL: [valuation]
⚠️ RISK: [vol, VIX, R/R]

──── COUNCIL DEBATE ────
🐂 BULL: [conviction X/10] — [key argument]
🐻 BEAR: [conviction X/10] — [key argument]
⚖️ MEDIATOR: [vote + confidence] — [reasoning]
COUNCIL VOTE: [X BUY / X SELL / X HOLD]
CONSENSUS: [STRONG/MODERATE/WEAK/SPLIT]

💰 TRADE: Entry [x] | SL [x] | TP [x] | [N] shares | R/R 1:[x]
📝 REASONING: [why you agree/disagree with council]
⚠️ DISCLAIMER: AI council debate, not financial advice.
═══════════════════════════════════════════════════

RULES:
- ALWAYS run full council (bull + bear + mediator)
- Bear conviction > 8 + risk REJECT → never BUY
- Bull conviction > 8 + mediator agrees → can BUY
- ALWAYS include council debate in output
"""


def run_analysis(query, hf_token=None, model_id="Qwen/Qwen2.5-72B-Instruct"):
    return create_trade_advisor(hf_token=hf_token, model_id=model_id).run(query)

if __name__ == "__main__":
    print(run_analysis("Analyze NVDA. Portfolio: $50K, moderate risk."))
