"""
📋 Structured Output Schema — Trade Report
============================================
Pydantic modeli ile yapılandırılmış çıktı.
Fund Manager'ın JSON formatında döndürdüğü rapor şeması.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class Signal(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


class CouncilVote(BaseModel):
    advisor: str = Field(description="bull / bear / mediator")
    vote: str = Field(description="BUY / SELL / HOLD")
    conviction: int = Field(ge=1, le=10, description="Conviction score 1-10")
    key_argument: str = Field(description="One-line summary of main argument")


class TradeReport(BaseModel):
    ticker: str
    company_name: str = ""
    date: str = ""
    signal: Signal = Signal.HOLD
    confidence: int = Field(ge=0, le=100, default=50)
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    shares: Optional[int] = None
    risk_reward: Optional[str] = None
    technical_signal: str = ""
    rsi: Optional[float] = None
    macd_signal: str = ""
    sentiment: str = ""
    fundamental_signal: str = ""
    pe_ratio: Optional[float] = None
    market_regime: str = ""
    vix: Optional[float] = None
    volatility_regime: str = ""
    council_votes: list[CouncilVote] = Field(default_factory=list)
    consensus: str = ""
    reasoning: str = ""
    warnings: list[str] = Field(default_factory=list)
    data_errors: list[str] = Field(default_factory=list)


def parse_trade_report(raw: str) -> tuple[TradeReport | None, str]:
    import json, re
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return TradeReport(**data), raw
        except Exception: pass
    try:
        data = json.loads(raw)
        return TradeReport(**data), raw
    except Exception: pass
    try:
        start = raw.index('{')
        depth = 0
        for i, c in enumerate(raw[start:], start):
            if c == '{': depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    data = json.loads(raw[start:i+1])
                    return TradeReport(**data), raw
    except Exception: pass
    return None, raw


def report_to_markdown(report: TradeReport) -> str:
    emoji = {"STRONG_BUY": "🟢🟢", "BUY": "🟢", "HOLD": "🟡", "SELL": "🔴", "STRONG_SELL": "🔴🔴"}
    signal_emoji = emoji.get(report.signal.value, "🟡")
    md = f"""
═══════════════════════════════════════════════════
## {signal_emoji} TRADE ADVISOR REPORT (LLM Council Edition)
═══════════════════════════════════════════════════

| | |
|---|---|
| 📊 **Ticker** | **{report.ticker}** — {report.company_name} |
| 📅 **Tarih** | {report.date} |
| 📈 **Sinyal** | **{report.signal.value}** |
| 🎯 **Güven** | **{report.confidence}%** |

### 📋 Veri Özeti

| Kategori | Değer |
|---|---|
| 📈 Teknik | {report.technical_signal} (RSI: {report.rsi or 'N/A'}, MACD: {report.macd_signal or 'N/A'}) |
| 📰 Sentiment | {report.sentiment} |
| 📊 Temel | {report.fundamental_signal} (P/E: {report.pe_ratio or 'N/A'}) |
| 🌍 Piyasa | {report.market_regime} (VIX: {report.vix or 'N/A'}) |
| ⚡ Volatilite | {report.volatility_regime} |
"""
    if report.council_votes:
        md += "\n### 🏛️ Council Tartışması\n\n| Advisor | Oy | Kanaat | Ana Argüman |\n|---|---|---|---|\n"
        for v in report.council_votes:
            icon = "🐂" if v.advisor == "bull" else ("🐻" if v.advisor == "bear" else "⚖️")
            md += f"| {icon} {v.advisor.title()} | **{v.vote}** | {v.conviction}/10 | {v.key_argument} |\n"
        md += f"\n**Konsensüs:** {report.consensus}\n"
    if report.entry_price:
        md += f"\n### 💰 Trade Parametreleri\n\n| | |\n|---|---|\n| 💰 Giriş | **${report.entry_price:,.2f}** |\n| 🛑 Stop-Loss | **${report.stop_loss:,.2f}** |\n| 🎯 Take-Profit | **${report.take_profit:,.2f}** |\n| 📦 Pozisyon | **{report.shares}** hisse |\n| ⚖️ R/R | **{report.risk_reward}** |\n"
    if report.reasoning: md += f"\n### 📝 Gerekçe\n\n{report.reasoning}\n"
    if report.warnings:
        md += "\n### ⚠️ Uyarılar\n\n"
        for w in report.warnings: md += f"- {w}\n"
    if report.data_errors:
        md += "\n### ❌ Veri Hataları\n\n"
        for e in report.data_errors: md += f"- ⚠️ {e}\n"
    md += "\n---\n⚠️ *Bu AI council tartışmasıdır, yatırım tavsiyesi değildir.*"
    return md
