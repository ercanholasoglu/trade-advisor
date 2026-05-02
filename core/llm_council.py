"""
🏛️ LLM Council — Real Agentic Trading Decision System
=========================================================
NOT rule-based. Real LLM agents that think, debate, and decide.

Architecture (inspired by TradingAgents paper arXiv:2412.20138):

  DATA LAYER (ToolCallingAgent — fast model):
    - Technical Analyst: calls get_price, get_indicators tools
    - Sentiment Analyst: calls get_news, get_sentiment tools
    - Risk Manager: calls get_volatility, get_market_overview tools

  COUNCIL LAYER (ToolCallingAgent — deep model):
    - Bull Advisor: builds strongest BUY case from data
    - Bear Advisor: builds strongest SELL case from data
    - Mediator: evaluates both sides, finds blind spots

  DECISION LAYER (CodeAgent — deep model):
    - Fund Manager: orchestrates all, produces final JSON decision

Each agent is a REAL LLM call. No hardcoded rules.
The LLM reasons, debates, and decides.
"""

import os
import json
import logging
import datetime

logger = logging.getLogger("trade_bot.council")


def _get_token():
    return os.environ.get("HF_TOKEN", "")


def _make_tools():
    """Create @tool decorated functions for smolagents."""
    from smolagents import tool

    @tool
    def get_price_data(ticker: str) -> str:
        """Fetches current price, recent OHLCV data, and price change for a stock or crypto.
        Args:
            ticker: Stock ticker symbol (e.g. AAPL, THYAO.IS, BTC-USD, GC=F).
        Returns:
            JSON string with current price, 30-day price history, volume, and change percentage.
        """
        from core.data_ingest import fetch_ohlcv
        result = fetch_ohlcv(ticker.strip().upper(), period="1mo", interval="1d")
        df = result.get("df")
        if df is None or len(df) < 2:
            return json.dumps({"error": f"No data for {ticker}", "source": result.get("source")})
        closes = df["Close"].values.astype(float)
        current = float(closes[-1])
        prev = float(closes[-2])
        change_pct = (current - prev) / prev * 100
        return json.dumps({"ticker": ticker.upper(), "price": round(current, 2),
            "change_pct": round(change_pct, 2), "high_30d": round(float(df["High"].max()), 2),
            "low_30d": round(float(df["Low"].min()), 2), "avg_volume": int(df["Volume"].mean()),
            "data_points": len(df), "source": result.get("source", "unknown")})

    @tool
    def get_technical_indicators(ticker: str) -> str:
        """Computes key technical indicators: RSI, MACD, SMA crossover, Bollinger Bands, ATR, and volatility regime.
        Args:
            ticker: Stock ticker symbol.
        Returns:
            JSON with RSI value/signal, MACD histogram/signal, SMA20 vs SMA50, Bollinger position, ATR, volatility.
        """
        from core.signal_engine import generate_signal
        sig = generate_signal(ticker.strip().upper())
        ind = sig.get("indicators", {})
        macd = ind.get("macd", {})
        for k in ["macd_series", "signal_series", "histogram_series"]:
            macd.pop(k, None)
        return json.dumps({"ticker": sig["ticker"], "price": sig["price"],
            "rsi": ind.get("rsi", {}), "sma_crossover": ind.get("sma_crossover", {}),
            "macd": macd, "bollinger": ind.get("bollinger", {}),
            "atr": ind.get("atr"), "volume_ratio": ind.get("volume_ratio"),
            "volatility": ind.get("volatility", {})})

    @tool
    def get_market_sentiment(ticker: str) -> str:
        """Analyzes market sentiment via price momentum, volume trends, and volatility patterns.
        Args:
            ticker: Stock ticker symbol.
        Returns:
            JSON with sentiment direction, confidence, and reasoning.
        """
        from core.agents import SentimentAgent
        agent = SentimentAgent()
        report = agent.analyze(ticker.strip().upper())
        return json.dumps(report.to_dict(), default=str)

    @tool
    def get_risk_assessment(ticker: str) -> str:
        """Evaluates risk: volatility regime, drawdown depth, ATR gap risk, and trend strength.
        Args:
            ticker: Stock ticker symbol.
        Returns:
            JSON with risk verdict (APPROVE/CAUTION/REJECT), risk score, and reasoning.
        """
        from core.agents import RiskAgent
        agent = RiskAgent()
        report = agent.analyze(ticker.strip().upper())
        return json.dumps(report.to_dict(), default=str)

    @tool
    def get_backtest_performance(ticker: str) -> str:
        """Runs a backtest of the RSI+SMA strategy on this ticker over 1 year.
        Args:
            ticker: Stock ticker symbol.
        Returns:
            JSON with accuracy, Sharpe ratio, total return, max drawdown, profit factor.
        """
        from core.backtester import run_backtest
        bt = run_backtest(ticker.strip().upper(), period_years=1, holding_days=10)
        if "error" in bt:
            return json.dumps({"error": bt["error"]})
        return json.dumps({"ticker": bt["ticker"], "accuracy_pct": bt["summary"]["accuracy_pct"],
            "total_return_pct": bt["returns"]["total_return_pct"],
            "sharpe_ratio": bt["returns"]["sharpe_ratio"],
            "max_drawdown_pct": bt["returns"]["max_drawdown_pct"],
            "profit_factor": bt["returns"]["profit_factor"],
            "total_signals": bt["summary"]["total_signals"]})

    @tool
    def get_ground_truth(ticker: str) -> str:
        """Checks historical signal accuracy at 5/10/20 day horizons.
        Args:
            ticker: Stock ticker symbol.
        Returns:
            JSON with accuracy per horizon and calibration data.
        """
        from core.evaluator import run_ground_truth_evaluation
        gt = run_ground_truth_evaluation(ticker.strip().upper(), period_years=1, eval_horizons=[5, 10, 20])
        if "error" in gt:
            return json.dumps({"error": gt["error"]})
        return json.dumps({"ticker": gt["ticker"], "total_evaluations": gt["total_evaluations"],
            "horizon_stats": {k: {"accuracy": v["accuracy_pct"], "avg_return": v["avg_directional_return_pct"]}
                             for k, v in gt["horizon_stats"].items()},
            "calibration": gt["calibration"]})

    return [get_price_data, get_technical_indicators, get_market_sentiment,
            get_risk_assessment, get_backtest_performance, get_ground_truth]


def build_council(model_id: str = None, fast_model_id: str = None):
    """Build the full LLM trading council. Returns Fund Manager CodeAgent."""
    from smolagents import CodeAgent, ToolCallingAgent, InferenceClientModel

    token = _get_token()
    if not token:
        raise ValueError("HF_TOKEN required for LLM Council.")

    model_id = model_id or os.environ.get("MODEL_ID", "Qwen/Qwen2.5-72B-Instruct")
    fast_model_id = fast_model_id or os.environ.get("FAST_MODEL_ID", model_id)

    deep_model = InferenceClientModel(model_id=model_id, token=token, temperature=0.3)
    fast_model = InferenceClientModel(model_id=fast_model_id, token=token, temperature=0.1)

    tools = _make_tools()

    technical_analyst = ToolCallingAgent(
        tools=[tools[0], tools[1]], model=fast_model, name="technical_analyst",
        description="Technical analyst — fetches price data and computes RSI, MACD, SMA, Bollinger, ATR, volatility.",
        max_steps=5, instructions="Fetch price and indicators, summarize trend, momentum, support/resistance, volatility.")

    sentiment_analyst = ToolCallingAgent(
        tools=[tools[2]], model=fast_model, name="sentiment_analyst",
        description="Sentiment analyst — analyzes market mood, volume trends, and momentum.",
        max_steps=4, instructions="Analyze sentiment. Report: mood, key drivers, volume trends.")

    risk_analyst = ToolCallingAgent(
        tools=[tools[3], tools[4], tools[5]], model=fast_model, name="risk_analyst",
        description="Risk manager — evaluates risk, runs backtests, checks ground truth accuracy.",
        max_steps=6, instructions="Assess risk: volatility, drawdown, backtest, ground truth. Give verdict: APPROVE/CAUTION/REJECT.")

    bull_advisor = ToolCallingAgent(
        tools=[], model=deep_model, name="bull_advisor",
        description="Bullish advisor — builds strongest BUY case from data.", max_steps=3,
        instructions="You are the BULL. Find every reason to BUY. Cite data. Conviction 1-10. Counter bears.")

    bear_advisor = ToolCallingAgent(
        tools=[], model=deep_model, name="bear_advisor",
        description="Bearish advisor — builds strongest SELL case from data.", max_steps=3,
        instructions="You are the BEAR. Find every risk. Cite data. Conviction 1-10. Counter bulls.")

    mediator = ToolCallingAgent(
        tools=[], model=deep_model, name="mediator",
        description="Neutral mediator — evaluates both sides, finds blind spots, votes.", max_steps=3,
        instructions="Evaluate bull vs bear argument quality. Find blind spots. Vote: BUY/HOLD/SELL with confidence 0-100%.")

    fund_manager = CodeAgent(
        tools=[], model=deep_model,
        managed_agents=[technical_analyst, sentiment_analyst, risk_analyst, bull_advisor, bear_advisor, mediator],
        additional_authorized_imports=["json", "datetime"],
        planning_interval=3, max_steps=20,
        instructions="""You are a fund manager. Follow this workflow:
Phase 1: Call technical_analyst, sentiment_analyst, risk_analyst for data.
Phase 2: Pass data to bull_advisor, bear_advisor, mediator for debate.
Phase 3: Synthesize into final JSON decision.
Rules: 3/3 agree=HIGH conf. 2/3=MODERATE. Split=HOLD. Risk REJECT=HOLD override.
Final answer must be JSON: {"ticker":str,"action":"BUY|SELL|HOLD","confidence":0-1,"rationale":str,"bull_conviction":0-10,"bear_conviction":0-10,"mediator_vote":str,"risk_verdict":str,"council_consensus":str}""")

    return fund_manager


_council = None

def get_council():
    global _council
    if _council is None:
        _council = build_council()
    return _council


def run_council_analysis(ticker: str, portfolio_context: str = "") -> dict:
    """Run the full LLM council analysis on a ticker."""
    ticker = ticker.strip().upper()
    if not ticker:
        return {"error": "No ticker provided"}
    try:
        council = get_council()
        prompt = f"Analyze {ticker} and make a trading decision. Follow full workflow: data → council debate → JSON."
        if portfolio_context:
            prompt += f" Context: {portfolio_context}"
        raw = str(council.run(prompt))
        decision = None
        try:
            import re
            json_match = re.search(r'\{[^{}]*"action"[^{}]*\}', raw, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
        except Exception:
            pass
        if decision is None:
            try:
                decision = json.loads(raw)
            except Exception:
                decision = {"action": "HOLD", "confidence": 0.3, "rationale": raw[:500], "parse_error": True}
        return {"ticker": ticker, "decision": decision, "raw_output": raw,
                "timestamp": datetime.datetime.now().isoformat()}
    except Exception as e:
        return {"ticker": ticker, "error": str(e),
                "decision": {"action": "HOLD", "confidence": 0.0, "rationale": f"Error: {e}"},
                "timestamp": datetime.datetime.now().isoformat()}
