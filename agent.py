"""
🤖 Agentic Trade Bot Advisor
=============================
TradingAgents paper (arxiv: 2412.20138) mimarisinden ilham alan
multi-agent trade advisor sistemi.

Mimari:
  - Technical Analyst Agent (RSI, MACD, Bollinger, SMA)
  - News & Sentiment Analyst Agent (haber + sentiment)
  - Fundamental Analyst Agent (P/E, earnings, financials)
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


def create_trade_advisor(hf_token: str | None = None, model_id: str = "Qwen/Qwen2.5-72B-Instruct"):
    """Multi-agent trade advisor sistemini oluşturur.

    Args:
        hf_token: HuggingFace API token. None ise env'den alır.
        model_id: Kullanılacak LLM model ID.

    Returns:
        Fund Manager CodeAgent (ana orchestrator)
    """
    token = hf_token or os.environ.get("HF_TOKEN", "")

    # --- Model ---
    model = InferenceClientModel(
        model_id=model_id,
        token=token,
    )

    # ============================================================
    # AGENT 1: Technical Analyst
    # ============================================================
    technical_analyst = ToolCallingAgent(
        tools=[get_price_history, get_technical_indicators],
        model=model,
        name="technical_analyst",
        description=(
            "Expert technical analyst that computes and interprets technical indicators "
            "(RSI, MACD, Bollinger Bands, SMA, EMA, ATR, Stochastic) and analyzes price "
            "action patterns. Call this agent to get a technical analysis report for any "
            "stock or crypto ticker. Provide the ticker symbol in your request."
        ),
        max_steps=6,
        instructions=(
            "You are a professional technical analyst. When analyzing a ticker:\n"
            "1. First get the price history to understand recent price action\n"
            "2. Then compute technical indicators for a comprehensive view\n"
            "3. Synthesize all data into a clear STRUCTURED report with:\n"
            "   - Current trend direction (UPTREND/DOWNTREND/SIDEWAYS)\n"
            "   - Key support and resistance levels\n"
            "   - Momentum assessment (RSI, MACD signals)\n"
            "   - Volatility assessment (Bollinger Bands, ATR)\n"
            "   - Overall technical signal: STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL\n"
            "Be precise with numbers. Always mention the timeframe of your analysis."
        ),
    )

    # ============================================================
    # AGENT 2: News & Sentiment Analyst
    # ============================================================
    sentiment_analyst = ToolCallingAgent(
        tools=[get_news_sentiment],
        model=model,
        name="sentiment_analyst",
        description=(
            "Analyzes recent news and market sentiment for a given ticker. "
            "Searches financial news, evaluates article tone, and provides "
            "an overall sentiment score (BULLISH/NEUTRAL/BEARISH). "
            "Provide the ticker symbol and optionally the company name."
        ),
        max_steps=4,
        instructions=(
            "You are a sentiment analyst specialized in financial markets.\n"
            "When analyzing a ticker:\n"
            "1. Search for recent news using the company name and ticker\n"
            "2. Evaluate each news article's impact on the stock\n"
            "3. Provide a structured report with:\n"
            "   - Overall sentiment: BULLISH / NEUTRAL / BEARISH\n"
            "   - Key positive catalysts (if any)\n"
            "   - Key negative risks (if any)\n"
            "   - Notable upcoming events (earnings, FDA approvals, etc.)\n"
            "   - Sentiment confidence level: HIGH / MEDIUM / LOW\n"
            "Focus on market-moving news. Ignore noise."
        ),
    )

    # ============================================================
    # AGENT 3: Fundamental Analyst
    # ============================================================
    fundamental_analyst = ToolCallingAgent(
        tools=[get_fundamental_data],
        model=model,
        name="fundamental_analyst",
        description=(
            "Evaluates company fundamentals: valuation ratios (P/E, P/B, PEG), "
            "financial performance (margins, growth, ROE), earnings outlook, "
            "analyst targets, and balance sheet health. Provide a ticker symbol."
        ),
        max_steps=4,
        instructions=(
            "You are a fundamental analyst with deep expertise in financial valuation.\n"
            "When analyzing a ticker:\n"
            "1. Fetch comprehensive fundamental data\n"
            "2. Evaluate the company across these dimensions:\n"
            "   - VALUATION: Is the stock cheap or expensive? (P/E, PEG, P/S vs sector)\n"
            "   - GROWTH: Revenue and earnings growth trajectory\n"
            "   - PROFITABILITY: Margins, ROE, ROA quality\n"
            "   - FINANCIAL HEALTH: Debt levels, cash flow, current ratio\n"
            "   - ANALYST CONSENSUS: Target price upside/downside\n"
            "3. Provide an overall fundamental rating: UNDERVALUED / FAIR / OVERVALUED\n"
            "Be specific with numbers. Compare to general market benchmarks when possible."
        ),
    )

    # ============================================================
    # AGENT 4: Risk Manager
    # ============================================================
    risk_manager = ToolCallingAgent(
        tools=[calculate_risk_metrics, get_market_overview],
        model=model,
        name="risk_manager",
        description=(
            "Assesses portfolio risk, calculates position sizing, sets stop-loss and "
            "take-profit levels. Also evaluates overall market conditions (VIX, sector "
            "rotation, macro trends). Provide ticker, portfolio value, and risk tolerance."
        ),
        max_steps=6,
        instructions=(
            "You are a senior risk manager at a trading firm.\n"
            "When evaluating a trade:\n"
            "1. First check overall market conditions (get_market_overview)\n"
            "2. Then calculate risk metrics for the specific ticker\n"
            "3. Provide a structured risk report with:\n"
            "   - MARKET ENVIRONMENT: Bull/Bear/Neutral + VIX regime\n"
            "   - VOLATILITY ASSESSMENT: Current vol regime for the stock\n"
            "   - POSITION SIZING: Exact shares and dollar amount\n"
            "   - ENTRY/EXIT LEVELS: Stop-loss and take-profit prices\n"
            "   - RISK/REWARD RATIO: Is the trade worth it?\n"
            "   - KEY RISKS: What could go wrong?\n"
            "   - RISK VERDICT: APPROVE / APPROVE_WITH_CAUTION / REJECT\n"
            "Capital preservation is your #1 priority."
        ),
    )

    # ============================================================
    # FUND MANAGER (Master Orchestrator)
    # ============================================================
    fund_manager = CodeAgent(
        tools=[],  # Direkt tool yok, sub-agent'lara delege eder
        model=model,
        managed_agents=[
            technical_analyst,
            sentiment_analyst,
            fundamental_analyst,
            risk_manager,
        ],
        additional_authorized_imports=["json", "datetime"],
        planning_interval=3,
        max_steps=15,
        instructions=FUND_MANAGER_INSTRUCTIONS,
    )

    return fund_manager


# ============================================================
# Fund Manager System Prompt
# ============================================================
FUND_MANAGER_INSTRUCTIONS = """
You are a senior fund manager and the head of a trading advisory team.
You coordinate 4 specialist agents to produce comprehensive trade recommendations.

## YOUR WORKFLOW (follow this EXACTLY):

### Step 1: Technical Analysis
Call the technical_analyst with the ticker to get price action and indicator analysis.

### Step 2: Sentiment Analysis  
Call the sentiment_analyst with the ticker and company name to assess market mood.

### Step 3: Fundamental Analysis
Call the fundamental_analyst with the ticker to evaluate intrinsic value.

### Step 4: Risk Assessment
Call the risk_manager with the ticker, portfolio value, and risk tolerance.

### Step 5: Final Decision
Synthesize ALL reports into a final trade recommendation.

## OUTPUT FORMAT (ALWAYS use this structure):

```
═══════════════════════════════════════════════════
🤖 TRADE ADVISOR REPORT
═══════════════════════════════════════════════════

📊 TICKER: [SYMBOL] | COMPANY: [NAME]
📅 DATE: [Current date]

───────────────────────────────────────────────────
📈 SIGNAL: [BUY / SELL / HOLD]
🎯 CONFIDENCE: [0-100]%
───────────────────────────────────────────────────

📋 TECHNICAL ANALYSIS SUMMARY:
• Trend: [UPTREND/DOWNTREND/SIDEWAYS]
• Key Indicators: [RSI, MACD signals]
• Support/Resistance: [levels]
• Technical Score: [BULLISH/NEUTRAL/BEARISH]

📰 SENTIMENT ANALYSIS SUMMARY:
• Overall Sentiment: [BULLISH/NEUTRAL/BEARISH]
• Key Catalysts: [positive news]
• Key Risks: [negative news]
• Sentiment Score: [BULLISH/NEUTRAL/BEARISH]

📊 FUNDAMENTAL ANALYSIS SUMMARY:
• Valuation: [UNDERVALUED/FAIR/OVERVALUED]
• Key Metrics: [P/E, Growth, Margins]
• Analyst Target: [price + upside %]
• Fundamental Score: [BULLISH/NEUTRAL/BEARISH]

⚠️ RISK ASSESSMENT:
• Market Regime: [BULLISH/BEARISH/NEUTRAL]
• Volatility: [LOW/MODERATE/HIGH]
• VIX: [value + regime]
• Risk Verdict: [APPROVE/CAUTION/REJECT]

───────────────────────────────────────────────────
💰 TRADE PARAMETERS:
• Entry Price: $[price]
• Stop-Loss: $[price] ([x]%)
• Take-Profit: $[price] ([x]%)
• Position Size: [X shares] ($[value])
• Risk/Reward: 1:[X]

📝 REASONING:
[2-3 sentences explaining the key factors behind your decision]

⚠️ DISCLAIMER: This is AI-generated analysis for educational 
purposes only. Not financial advice. Always do your own research.
═══════════════════════════════════════════════════
```

## RULES:
- ALWAYS consult ALL 4 agents before making a decision
- If signals conflict (e.g. technical bullish but fundamental bearish), weight risk management highest
- Never recommend a trade without stop-loss and take-profit levels
- If market conditions are extreme (VIX > 30), recommend HOLD regardless of other signals
- Be honest about uncertainty — if confidence is below 50%, recommend HOLD
- ALWAYS include the disclaimer
"""


def run_analysis(query: str, hf_token: str | None = None, model_id: str = "Qwen/Qwen2.5-72B-Instruct"):
    """Tek seferlik analiz çalıştırır.

    Args:
        query: Kullanıcı sorusu (e.g. "Analyze AAPL for a potential trade. Portfolio: $100k, moderate risk.")
        hf_token: HF API token
        model_id: LLM model ID

    Returns:
        Agent'ın son yanıtı (string)
    """
    advisor = create_trade_advisor(hf_token=hf_token, model_id=model_id)
    result = advisor.run(query)
    return result


if __name__ == "__main__":
    # Hızlı test
    result = run_analysis(
        "Analyze NVDA for a potential trade entry. Portfolio: $50,000. Risk tolerance: moderate.",
    )
    print(result)
