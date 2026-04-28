"""
🤖 Agentic Trade Bot Advisor — LLM Council Edition v2.0
=========================================================
TradingAgents (arxiv: 2412.20138) + ContestTrade (arxiv: 2508.00554)
mimarilerinden ilham alan multi-agent + LLM Council sistemi.

v2.0 Yenilikler:
  - Dual model routing (fast/deep)
  - Structured JSON output (TradeReport schema)
  - 12 agents (5 data + 1 macro + 3 council + 1 orchestrator + 2 new tools)
  - FinBERT sentiment, crypto fundamentals, insider trades, options flow
  - Graceful error handling — agents continue with partial data
  - SQLite logging for every analysis
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
from tools.crypto_fundamental import get_crypto_fundamentals
from tools.insider_trades import get_insider_trades
from tools.options_flow import get_options_flow
from tools.macro_data import get_macro_data


def create_trade_advisor(hf_token: str | None = None,
                          model_id: str = "Qwen/Qwen2.5-72B-Instruct",
                          fast_model_id: str | None = None,
                          deep_model_id: str | None = None):
    """
    Create the trade advisor multi-agent system.
    
    Dual model routing:
      - fast_model: Data collection agents (technical, sentiment, fundamental, bist, risk, macro)
      - deep_model: Council + Fund Manager (complex reasoning)
    
    If fast_model_id/deep_model_id not specified, falls back to model_id for all.
    """
    token = hf_token or os.environ.get("HF_TOKEN", "")

    # Dual model setup
    fast_id = fast_model_id or os.environ.get("FAST_MODEL_ID", model_id)
    deep_id = deep_model_id or os.environ.get("DEEP_MODEL_ID", model_id)

    model_fast = InferenceClientModel(model_id=fast_id, token=token)

    # Only create separate deep model if it's actually different
    if deep_id != fast_id:
        model_deep = InferenceClientModel(model_id=deep_id, token=token)
    else:
        model_deep = model_fast

    # === VERİ TOPLAMA KATMANI (6 data agents — fast model) ===
    technical_analyst = ToolCallingAgent(
        tools=[get_price_history, get_technical_indicators],
        model=model_fast, name="technical_analyst",
        description="Technical analyst — RSI, MACD, Bollinger Bands, SMA, EMA, ATR, Stochastic. Supports BIST (.IS suffix) and US stocks.",
        max_steps=6,
        instructions="""Get price history then compute indicators. Report: trend, support/resistance, momentum, volatility, signal (STRONG_BUY/BUY/HOLD/SELL/STRONG_SELL). Be precise with numbers.
If a tool returns an error, note it and provide your assessment with available data.""",
    )
    sentiment_analyst = ToolCallingAgent(
        tools=[get_news_sentiment, get_kap_disclosures, get_insider_trades],
        model=model_fast, name="sentiment_analyst",
        description="Sentiment analyst — news + KAP disclosures + SEC insider trades.",
        max_steps=6,
        instructions="""Search news, check KAP for BIST stocks, check insider trades for US stocks.
Report: sentiment, catalysts, risks, events, confidence.
If any tool returns an error, note it and continue with available data.""",
    )
    fundamental_analyst = ToolCallingAgent(
        tools=[get_fundamental_data, get_crypto_fundamentals],
        model=model_fast, name="fundamental_analyst",
        description="Fundamental analyst — P/E, PEG, margins, growth, ROE for stocks. CoinGecko data for crypto.",
        max_steps=5,
        instructions="""For stocks: use get_fundamental_data. For crypto (BTC, ETH, etc.): use get_crypto_fundamentals.
Evaluate valuation, growth, profitability, financial health, analyst consensus.
Rate UNDERVALUED/FAIR/OVERVALUED. If a tool errors, note it and continue.""",
    )
    bist_analyst = ToolCallingAgent(
        tools=[get_bist_scanner, get_daily_dashboard],
        model=model_fast, name="bist_analyst",
        description="Turkish market specialist — BIST30, banks, gold/silver TRY, USD/TRY, BIST indices, daily dashboard.",
        max_steps=5,
        instructions="""Use get_bist_scanner for BIST scans and commodities_try.
Use get_daily_dashboard for daily AL/SAT. Include altın and USD/TRY.
If tools error, note it and continue.""",
    )
    risk_manager = ToolCallingAgent(
        tools=[calculate_risk_metrics, get_market_overview, get_options_flow],
        model=model_fast, name="risk_manager",
        description="Risk manager — VaR, position sizing, stop-loss/take-profit, VIX, market regime, options flow.",
        max_steps=7,
        instructions="""Check market conditions, calculate risk, and analyze options flow (for US stocks with options).
Report: environment, volatility, position sizing, levels, R/R, put/call ratio, verdict (APPROVE/CAUTION/REJECT).
High put/call ratio (>1.3) is a bearish signal override.
If any tool returns an error, note it and continue with available data.""",
    )
    macro_analyst = ToolCallingAgent(
        tools=[get_macro_data],
        model=model_fast, name="macro_analyst",
        description="Macro economist — TCMB rates, USD/TRY, US treasury yields, VIX, yield curve, macro regime.",
        max_steps=4,
        instructions="""Evaluate macro environment for both Turkey and US.
Report: macro regime (TIGHTENING/EASING/NEUTRAL), risk factor (1-10), upcoming catalysts.
For BIST stocks, focus on Turkey macro. For US stocks, focus on US macro. For both, report global regime.
If tools error, note it and provide general macro assessment.""",
    )

    # === LLM COUNCIL (3 advisors — deep model for complex reasoning) ===
    bullish_advisor = ToolCallingAgent(
        tools=[], model=model_deep, name="bullish_advisor",
        description="🐂 Bullish Advisor — constructs the strongest possible BULL CASE from analyst data.",
        max_steps=3, instructions=BULLISH_INSTRUCTIONS,
    )
    bearish_advisor = ToolCallingAgent(
        tools=[], model=model_deep, name="bearish_advisor",
        description="🐻 Bearish Advisor — constructs the strongest possible BEAR CASE from analyst data.",
        max_steps=3, instructions=BEARISH_INSTRUCTIONS,
    )
    neutral_mediator = ToolCallingAgent(
        tools=[], model=model_deep, name="neutral_mediator",
        description="⚖️ Neutral Mediator — evaluates bull vs bear argument quality, provides balanced verdict.",
        max_steps=3, instructions=MEDIATOR_INSTRUCTIONS,
    )

    # === FUND MANAGER (Orchestrator — deep model) ===
    fund_manager = CodeAgent(
        tools=[], model=model_deep,
        managed_agents=[
            technical_analyst, sentiment_analyst, fundamental_analyst,
            bist_analyst, risk_manager, macro_analyst,
            bullish_advisor, bearish_advisor, neutral_mediator,
        ],
        additional_authorized_imports=["json", "datetime"],
        planning_interval=3, max_steps=28,
        instructions=FUND_MANAGER_INSTRUCTIONS,
    )
    return fund_manager


BULLISH_INSTRUCTIONS = """
You are the 🐂 BULLISH ADVISOR on a trading council.
Your ONLY job: build the strongest BULL CASE.

1. Find EVERY positive signal (momentum, sentiment, undervaluation, growth, insider buying)
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
If some data was unavailable (errors), note it but don't let it weaken your case unnecessarily.
"""

BEARISH_INSTRUCTIONS = """
You are the 🐻 BEARISH ADVISOR on a trading council.
Your ONLY job: build the strongest BEAR CASE.

1. Find EVERY negative signal (overbought, bad sentiment, overvaluation, slowing growth, insider selling)
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
If some data was unavailable (errors), flag it as additional uncertainty/risk.
"""

MEDIATOR_INSTRUCTIONS = """
You are the ⚖️ NEUTRAL MEDIATOR on a trading council.
You heard both Bull and Bear. Your job:

1. Evaluate ARGUMENT QUALITY — which side has stronger data?
2. Identify WEAK REASONING — where does either side stretch?
3. Identify BLIND SPOTS — what did both miss?
4. Account for DATA GAPS — if some tools failed, how does that affect confidence?
5. Provide BALANCED VERDICT with confidence

FORMAT:
⚖️ MEDIATOR — Bull Strength: [X]/10, Bear Strength: [X]/10
WEAK ARGUMENTS: Bull: [...] Bear: [...]
BLIND SPOTS: [...]
DATA GAPS: [any missing data from failed tools]
FINAL VOTE: BUY / HOLD / SELL
CONFIDENCE: [0-100]%
REASONING: [2-3 balanced sentences]

Be impartial. Your job is TRUTH.
"""

FUND_MANAGER_INSTRUCTIONS = """
You are a senior fund manager running a Trading Council (6 data analysts + 3 council advisors).

## WORKFLOW (follow EXACTLY):

### Phase 1: Data Collection
1. Call technical_analyst → technical report
2. Call sentiment_analyst → sentiment + KAP + insider report
3. Call fundamental_analyst → fundamental report (use crypto tool for crypto tickers)
4. For BIST: Call bist_analyst → Turkish market context
5. Call risk_manager → risk assessment + options flow
6. Call macro_analyst → macro environment

### Phase 2: LLM Council Debate
Pass ALL Phase 1 data to council (include any errors/missing data):
7. Call bullish_advisor with all data → bull case + conviction
8. Call bearish_advisor with all data → bear case + conviction
9. Call neutral_mediator with bull + bear cases → balanced verdict

### Phase 3: Final Decision
10. Synthesize council votes + data → final recommendation

## ERROR HANDLING:
- If any agent returns an error, NOTE IT in your report
- Continue analysis with available data
- Reduce confidence if key data is missing
- Never silently ignore errors

## Council rules:
- 3/3 agree → HIGH confidence, follow council
- 2/3 agree → MODERATE confidence, follow majority
- All disagree → LOW confidence → HOLD
- Mediator breaks ties
- VIX > 30 → HOLD override
- Put/Call > 1.5 → bearish override (reduce BUY confidence)

## BIST: THYAO.IS / KAP: THYAO / US: AAPL

## RESPOND WITH BOTH:
1. A structured JSON block (for programmatic parsing)
2. A human-readable report (for display)

### JSON FORMAT (wrap in ```json ... ```):
```json
{
  "ticker": "SYMBOL",
  "company_name": "Name",
  "date": "YYYY-MM-DD",
  "signal": "BUY|SELL|HOLD|STRONG_BUY|STRONG_SELL",
  "confidence": 0-100,
  "entry_price": 0.0,
  "stop_loss": 0.0,
  "take_profit": 0.0,
  "shares": 0,
  "risk_reward": "1:X.X",
  "technical_signal": "BUY/SELL/HOLD",
  "rsi": 0.0,
  "macd_signal": "BULLISH/BEARISH",
  "sentiment": "BULLISH/BEARISH/NEUTRAL",
  "fundamental_signal": "UNDERVALUED/FAIR/OVERVALUED",
  "pe_ratio": 0.0,
  "market_regime": "BULLISH/BEARISH",
  "vix": 0.0,
  "volatility_regime": "LOW/MODERATE/HIGH",
  "council_votes": [
    {"advisor": "bull", "vote": "BUY", "conviction": 8, "key_argument": "..."},
    {"advisor": "bear", "vote": "SELL", "conviction": 5, "key_argument": "..."},
    {"advisor": "mediator", "vote": "BUY", "conviction": 7, "key_argument": "..."}
  ],
  "consensus": "STRONG|MODERATE|WEAK|SPLIT",
  "reasoning": "Final reasoning...",
  "warnings": ["warning1", "warning2"],
  "data_errors": ["tool X failed: reason"]
}
```

### HUMAN-READABLE REPORT (after JSON):
═══════════════════════════════════════════════════
🤖 TRADE ADVISOR REPORT (LLM Council Edition)
═══════════════════════════════════════════════════
📊 TICKER: [SYMBOL] | COMPANY: [NAME] | 📅 [date]
📈 SIGNAL: [BUY/SELL/HOLD] | 🎯 CONFIDENCE: [0-100]%

──── DATA SUMMARY ────
📋 TECHNICAL: [trend, RSI, MACD]
📰 SENTIMENT: [news + KAP + insider]
📊 FUNDAMENTAL: [valuation]
⚠️ RISK: [vol, VIX, R/R, options]
🌍 MACRO: [regime, risk factor]

──── COUNCIL DEBATE ────
🐂 BULL: [conviction X/10] — [key argument]
🐻 BEAR: [conviction X/10] — [key argument]
⚖️ MEDIATOR: [vote + confidence] — [reasoning]
COUNCIL VOTE: [X BUY / X SELL / X HOLD]
CONSENSUS: [STRONG/MODERATE/WEAK/SPLIT]

💰 TRADE: Entry [x] | SL [x] | TP [x] | [N] shares | R/R 1:[x]
📝 REASONING: [why you agree/disagree with council]

❌ DATA ISSUES: [list any tools that failed]
⚠️ DISCLAIMER: AI council debate, not financial advice.
═══════════════════════════════════════════════════

RULES:
- ALWAYS run full council (bull + bear + mediator)
- Bear conviction > 8 + risk REJECT → never BUY
- Bull conviction > 8 + mediator agrees → can BUY
- ALWAYS include council debate in output
- ALWAYS include JSON block for structured parsing
"""


def run_analysis(query, hf_token=None, model_id="Qwen/Qwen2.5-72B-Instruct",
                  fast_model_id=None, deep_model_id=None):
    """Run a full analysis and return the result."""
    advisor = create_trade_advisor(
        hf_token=hf_token, model_id=model_id,
        fast_model_id=fast_model_id, deep_model_id=deep_model_id,
    )
    result = advisor.run(query)

    # Try to log to database
    try:
        from db import save_analysis
        from output_schema import parse_trade_report
        report, _ = parse_trade_report(str(result))
        if report:
            save_analysis(
                ticker=report.ticker,
                signal=report.signal.value,
                confidence=report.confidence,
                entry_price=report.entry_price,
                stop_loss=report.stop_loss,
                take_profit=report.take_profit,
                shares=report.shares,
                risk_reward=report.risk_reward,
                technical_signal=report.technical_signal,
                sentiment=report.sentiment,
                fundamental_signal=report.fundamental_signal,
                market_regime=report.market_regime,
                vix=report.vix,
                council_votes=[v.model_dump() for v in report.council_votes] if report.council_votes else None,
                reasoning=report.reasoning,
                raw_report=str(result),
            )
    except Exception:
        pass  # Don't fail the analysis if logging fails

    return result


if __name__ == "__main__":
    print(run_analysis("Analyze NVDA. Portfolio: $50K, moderate risk."))
