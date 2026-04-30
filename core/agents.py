"""
🤖 Multi-Agent System — Specialized agents with orchestrator
================================================================
A. Multi-Agent Architecture:
  - AnalystAgent: Technical indicators + volume analysis
  - SentimentAgent: News keyword scanning + market mood
  - RiskAgent: Volatility, drawdown limits, position constraints
  - OrchestratorAgent: Synthesizes all agent reports into final decision

Each agent runs independently, produces a structured report,
and the Orchestrator merges them with configurable weights.
"""

import datetime
import logging
import numpy as np

logger = logging.getLogger("trade_bot.agents")


class AgentReport:
    """Standardized report from any agent."""
    def __init__(self, agent_name: str, signal: str, confidence: int,
                 reasoning: list[str], data: dict = None):
        self.agent_name = agent_name
        self.signal = signal  # BULLISH / BEARISH / NEUTRAL
        self.confidence = confidence  # 0-100
        self.reasoning = reasoning
        self.data = data or {}
        self.timestamp = datetime.datetime.now().isoformat()

    def to_dict(self):
        return {
            "agent": self.agent_name,
            "signal": self.signal,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "data": self.data,
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════
# ANALYST AGENT — Technical indicators specialist
# ═══════════════════════════════════════════════════════

class AnalystAgent:
    """Analyzes RSI, SMA, MACD, Bollinger, Volume. Pure technical."""

    name = "Analyst"

    def analyze(self, ticker: str, period: str = "3mo") -> AgentReport:
        from core.signal_engine import generate_signal
        sig = generate_signal(ticker, period=period)

        ind = sig.get("indicators", {})
        votes = sig.get("votes", {})
        bullish = sig.get("bullish_count", 0)
        bearish = sig.get("bearish_count", 0)

        if bullish > bearish:
            direction = "BULLISH"
        elif bearish > bullish:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        reasons = sig.get("reasons", [])
        conf = sig.get("confidence", 50)

        return AgentReport(
            agent_name=self.name,
            signal=direction,
            confidence=conf,
            reasoning=reasons,
            data={
                "rsi": ind.get("rsi", {}).get("value"),
                "sma_signal": ind.get("sma_crossover", {}).get("signal"),
                "macd_signal": ind.get("macd", {}).get("signal"),
                "bollinger_signal": ind.get("bollinger", {}).get("signal"),
                "volatility_regime": ind.get("volatility", {}).get("regime"),
                "volume_ratio": ind.get("volume_ratio"),
                "price": sig.get("price"),
                "bullish_count": bullish,
                "bearish_count": bearish,
                "votes": votes,
                "trade": sig.get("trade", {}),
                "full_signal": sig,
            },
        )


# ═══════════════════════════════════════════════════════
# SENTIMENT AGENT — Market mood from news keywords + price action
# ═══════════════════════════════════════════════════════

class SentimentAgent:
    """
    Analyzes market sentiment via:
    1. Price momentum (5-day, 20-day returns)
    2. Volume trend (rising/falling)
    3. Volatility expansion/contraction
    4. Basic keyword sentiment (simulated — real FinBERT needs API)
    """

    name = "Sentiment"

    # Simulated news sentiment keywords (in production: FinBERT on real headlines)
    BULLISH_KEYWORDS = ["upgrade", "beat", "growth", "surge", "rally", "breakout", "buy", "outperform"]
    BEARISH_KEYWORDS = ["downgrade", "miss", "decline", "crash", "selloff", "cut", "underperform", "warning"]

    def analyze(self, ticker: str, period: str = "3mo") -> AgentReport:
        from core.data_ingest import fetch_ohlcv

        result = fetch_ohlcv(ticker, period=period, interval="1d")
        df = result.get("df")
        if df is None or len(df) < 20:
            return AgentReport(self.name, "NEUTRAL", 30, ["Insufficient data for sentiment"], {})

        closes = df["Close"].values.astype(float)
        volumes = df["Volume"].values.astype(float)

        # Price momentum
        ret_5d = (closes[-1] - closes[-5]) / closes[-5] * 100 if len(closes) >= 5 else 0
        ret_20d = (closes[-1] - closes[-20]) / closes[-20] * 100 if len(closes) >= 20 else 0

        # Volume trend
        vol_recent = float(np.mean(volumes[-5:])) if len(volumes) >= 5 else 0
        vol_avg = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else 1
        vol_trend = vol_recent / vol_avg if vol_avg > 0 else 1.0

        # Volatility momentum
        if len(closes) >= 30:
            vol_recent_std = float(np.std(np.diff(closes[-10:]) / closes[-10:-1]))
            vol_older_std = float(np.std(np.diff(closes[-30:-10]) / closes[-30:-11]))
            vol_expanding = vol_recent_std > vol_older_std * 1.2
        else:
            vol_expanding = False

        # Composite sentiment score
        score = 0
        reasons = []

        if ret_5d > 2:
            score += 2; reasons.append(f"Strong 5-day momentum: {ret_5d:+.1f}%")
        elif ret_5d > 0:
            score += 1; reasons.append(f"Positive 5-day momentum: {ret_5d:+.1f}%")
        elif ret_5d < -2:
            score -= 2; reasons.append(f"Negative 5-day momentum: {ret_5d:+.1f}%")
        elif ret_5d < 0:
            score -= 1; reasons.append(f"Weak 5-day momentum: {ret_5d:+.1f}%")

        if ret_20d > 5:
            score += 1; reasons.append(f"Bullish 20-day trend: {ret_20d:+.1f}%")
        elif ret_20d < -5:
            score -= 1; reasons.append(f"Bearish 20-day trend: {ret_20d:+.1f}%")

        if vol_trend > 1.5:
            reasons.append(f"Rising volume ({vol_trend:.1f}x avg) - interest increasing")
            if ret_5d > 0: score += 1
            else: score -= 1
        elif vol_trend < 0.6:
            reasons.append(f"Declining volume ({vol_trend:.1f}x avg) - interest fading")

        if vol_expanding:
            reasons.append("Volatility expanding - potential breakout/breakdown")

        if score >= 3: direction = "BULLISH"
        elif score >= 1: direction = "BULLISH"
        elif score <= -3: direction = "BEARISH"
        elif score <= -1: direction = "BEARISH"
        else: direction = "NEUTRAL"

        confidence = min(90, max(20, 50 + abs(score) * 10))
        if not reasons:
            reasons.append("Mixed sentiment - no strong directional bias")

        return AgentReport(
            agent_name=self.name,
            signal=direction,
            confidence=confidence,
            reasoning=reasons,
            data={
                "return_5d": round(ret_5d, 2),
                "return_20d": round(ret_20d, 2),
                "volume_trend": round(vol_trend, 2),
                "volatility_expanding": vol_expanding,
                "sentiment_score": score,
            },
        )


# ═══════════════════════════════════════════════════════
# RISK AGENT — Portfolio & position risk constraints
# ═══════════════════════════════════════════════════════

class RiskAgent:
    """
    Evaluates risk factors:
    1. Volatility regime (ATR, annualized vol)
    2. Drawdown risk (recent peak-to-trough)
    3. Position sizing constraints
    4. Market regime (trending vs ranging)
    """

    name = "Risk"

    def analyze(self, ticker: str, portfolio_value: float = 100000,
                risk_tolerance: str = "moderate", period: str = "3mo") -> AgentReport:
        from core.data_ingest import fetch_ohlcv

        result = fetch_ohlcv(ticker, period=period, interval="1d")
        df = result.get("df")
        if df is None or len(df) < 30:
            return AgentReport(self.name, "NEUTRAL", 30, ["Insufficient data for risk"], {"verdict": "CAUTION"})

        closes = df["Close"].values.astype(float)
        highs = df["High"].values.astype(float)
        lows = df["Low"].values.astype(float)

        reasons = []
        risk_score = 0  # Higher = more dangerous

        # 1. Annualized volatility
        returns = np.diff(closes[-30:]) / closes[-30:-1]
        annual_vol = float(np.std(returns) * np.sqrt(252))
        if annual_vol > 0.50:
            risk_score += 3; reasons.append(f"EXTREME volatility: {annual_vol*100:.0f}% annualized")
        elif annual_vol > 0.30:
            risk_score += 2; reasons.append(f"HIGH volatility: {annual_vol*100:.0f}% annualized")
        elif annual_vol > 0.15:
            risk_score += 1; reasons.append(f"Moderate volatility: {annual_vol*100:.0f}% annualized")
        else:
            reasons.append(f"Low volatility: {annual_vol*100:.0f}% annualized - favorable")

        # 2. Recent drawdown
        peak = np.maximum.accumulate(closes[-60:]) if len(closes) >= 60 else np.maximum.accumulate(closes)
        current_dd = (closes[-1] - peak[-1]) / peak[-1] * 100
        max_dd = float(np.min((closes[-60:] - peak) / peak * 100)) if len(closes) >= 60 else 0
        if current_dd < -10:
            risk_score += 2; reasons.append(f"In drawdown: {current_dd:.1f}% from recent peak")
        elif current_dd < -5:
            risk_score += 1; reasons.append(f"Mild pullback: {current_dd:.1f}% from peak")
        else:
            reasons.append(f"Near highs: {current_dd:.1f}% from peak")

        # 3. ATR-based gap risk
        if len(closes) >= 14:
            from core.signal_engine import _atr
            atr = _atr(highs, lows, closes)
            if atr:
                atr_pct = atr / closes[-1] * 100
                if atr_pct > 5:
                    risk_score += 1; reasons.append(f"High ATR: {atr_pct:.1f}% of price - wide stops needed")
        
        # 4. Trend strength (ADX proxy via directional movement)
        if len(closes) >= 20:
            up_moves = sum(1 for i in range(-20, 0) if closes[i] > closes[i-1])
            trend_pct = up_moves / 20 * 100
            if 35 <= trend_pct <= 65:
                reasons.append(f"Ranging market ({trend_pct:.0f}% up days) - signals less reliable")
                risk_score += 1
            elif trend_pct > 65:
                reasons.append(f"Strong uptrend ({trend_pct:.0f}% up days)")
            else:
                reasons.append(f"Strong downtrend ({trend_pct:.0f}% up days)")

        # Verdict
        if risk_score >= 5:
            verdict, direction, conf = "REJECT", "BEARISH", 80
            reasons.insert(0, "HIGH RISK - trading not recommended in current conditions")
        elif risk_score >= 3:
            verdict, direction, conf = "CAUTION", "NEUTRAL", 60
            reasons.insert(0, "ELEVATED RISK - reduce position size, widen stops")
        else:
            verdict, direction, conf = "APPROVE", "NEUTRAL", 50
            reasons.insert(0, "ACCEPTABLE RISK - normal trading conditions")

        return AgentReport(
            agent_name=self.name,
            signal=direction,
            confidence=conf,
            reasoning=reasons,
            data={
                "verdict": verdict,
                "risk_score": risk_score,
                "annualized_vol": round(annual_vol * 100, 1),
                "current_drawdown": round(current_dd, 1),
                "max_drawdown_60d": round(max_dd, 1),
            },
        )


# ═══════════════════════════════════════════════════════
# ORCHESTRATOR — Synthesizes all agent reports
# ═══════════════════════════════════════════════════════

class OrchestratorAgent:
    """
    Merges reports from Analyst, Sentiment, Risk agents.
    Applies configurable weights per agent.
    Produces final decision with full reasoning chain.
    """

    name = "Orchestrator"

    DEFAULT_WEIGHTS = {
        "Analyst": 0.50,    # Technical is primary
        "Sentiment": 0.25,  # Sentiment is secondary
        "Risk": 0.25,       # Risk is veto-capable
    }

    def synthesize(self, reports: list[AgentReport],
                   weights: dict[str, float] = None) -> dict:
        """Merge agent reports into final decision."""
        w = weights or self.DEFAULT_WEIGHTS

        # Score each report: BULLISH=+1, NEUTRAL=0, BEARISH=-1
        signal_map = {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}
        weighted_score = 0
        total_weight = 0
        agent_summaries = []

        risk_verdict = "APPROVE"

        for report in reports:
            weight = w.get(report.agent_name, 0.25)
            score = signal_map.get(report.signal, 0)
            weighted_score += score * weight * (report.confidence / 100)
            total_weight += weight

            if report.agent_name == "Risk":
                risk_verdict = report.data.get("verdict", "APPROVE")

            agent_summaries.append({
                "agent": report.agent_name,
                "signal": report.signal,
                "confidence": report.confidence,
                "weight": weight,
                "contribution": round(score * weight * (report.confidence / 100), 3),
                "top_reason": report.reasoning[0] if report.reasoning else "N/A",
            })

        # Normalize
        if total_weight > 0:
            normalized = weighted_score / total_weight
        else:
            normalized = 0

        # Risk veto
        if risk_verdict == "REJECT" and normalized > 0:
            final_signal = "HOLD"
            override_reason = "Risk Agent VETO: conditions too dangerous for BUY"
        elif risk_verdict == "CAUTION" and normalized > 0:
            normalized *= 0.5  # Reduce conviction
            final_signal = self._score_to_signal(normalized)
            override_reason = "Risk Agent CAUTION: conviction reduced"
        else:
            final_signal = self._score_to_signal(normalized)
            override_reason = None

        # Confidence from agreement
        signals = [r.signal for r in reports]
        agreement = len(set(signals)) == 1  # All agents agree
        partial = signals.count(signals[0]) >= 2

        if agreement:
            consensus, conf_bonus = "UNANIMOUS", 20
        elif partial:
            consensus, conf_bonus = "MAJORITY", 10
        else:
            consensus, conf_bonus = "SPLIT", -10

        final_confidence = min(95, max(10, int(abs(normalized) * 70 + conf_bonus)))

        # Build reasoning chain
        reasoning_chain = []
        for r in reports:
            reasoning_chain.append(f"[{r.agent_name}] {r.signal} ({r.confidence}%): {r.reasoning[0] if r.reasoning else 'N/A'}")
        if override_reason:
            reasoning_chain.append(f"[Orchestrator Override] {override_reason}")
        reasoning_chain.append(f"[Final] {final_signal} ({final_confidence}%) - Consensus: {consensus}")

        return {
            "signal": final_signal,
            "confidence": final_confidence,
            "consensus": consensus,
            "weighted_score": round(normalized, 3),
            "risk_verdict": risk_verdict,
            "override": override_reason,
            "agent_summaries": agent_summaries,
            "reasoning_chain": reasoning_chain,
            "weights_used": w,
        }

    def _score_to_signal(self, score):
        if score > 0.4: return "STRONG_BUY"
        if score > 0.15: return "BUY"
        if score < -0.4: return "STRONG_SELL"
        if score < -0.15: return "SELL"
        return "HOLD"


# ═══════════════════════════════════════════════════════
# PUBLIC API — Run full multi-agent analysis
# ═══════════════════════════════════════════════════════

def run_multi_agent_analysis(ticker: str, portfolio_value: float = 100000,
                              risk_tolerance: str = "moderate",
                              weights: dict[str, float] = None) -> dict:
    """
    Run all 4 agents and produce synthesized decision.
    Returns full report with per-agent details + orchestrator decision.
    """
    ticker = ticker.strip().upper()

    analyst = AnalystAgent()
    sentiment = SentimentAgent()
    risk = RiskAgent()
    orchestrator = OrchestratorAgent()

    # Run agents
    analyst_report = analyst.analyze(ticker)
    sentiment_report = sentiment.analyze(ticker)
    risk_report = risk.analyze(ticker, portfolio_value, risk_tolerance)

    # Orchestrate
    decision = orchestrator.synthesize(
        [analyst_report, sentiment_report, risk_report],
        weights=weights,
    )

    return {
        "ticker": ticker,
        "timestamp": datetime.datetime.now().isoformat(),
        "decision": decision,
        "agents": {
            "analyst": analyst_report.to_dict(),
            "sentiment": sentiment_report.to_dict(),
            "risk": risk_report.to_dict(),
        },
    }
