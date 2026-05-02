"""LLM Council UI wrapper — called from app.py"""

import json
from core.llm_council import run_council_analysis


def run_llm_council_ui(ticker, portfolio_context):
    """Run LLM Council analysis and format output for Gradio."""
    if not ticker or not ticker.strip():
        return "❌ Ticker girin (HF_TOKEN gerekli)", {}

    result = run_council_analysis(ticker.strip(), portfolio_context or "")

    if "error" in result and not result.get("decision"):
        return f"❌ Hata: {result['error']}\n\n**Not:** LLM Council icin HF_TOKEN environment variable gerekli.", result

    d = result.get("decision", {})
    raw = result.get("raw_output", "")

    se = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(d.get("action", "HOLD"), "⚪")

    md = f"## 🏛️ LLM Council Karari: {se} **{d.get('action', 'HOLD')}**\n\n"
    md += f"**Ticker:** {result.get('ticker', '?')} | **Guven:** {d.get('confidence', 0):.0%}\n\n"

    if d.get("rationale"):
        md += f"**Gerekce:** {d['rationale']}\n\n"

    md += "---\n### Council Detaylari\n\n"
    md += f"| Metrik | Deger |\n|---|---|\n"
    md += f"| Bull Conviction | {d.get('bull_conviction', '?')}/10 |\n"
    md += f"| Bear Conviction | {d.get('bear_conviction', '?')}/10 |\n"
    md += f"| Mediator Vote | {d.get('mediator_vote', '?')} |\n"
    md += f"| Risk Verdict | {d.get('risk_verdict', '?')} |\n"
    md += f"| Consensus | {d.get('council_consensus', '?')} |\n"

    if raw and len(raw) > 100:
        md += f"\n---\n### Ham LLM Ciktisi\n\n```\n{raw[:2000]}\n```\n"

    md += "\n⚠️ *Bu gercek LLM ajanlarinin tartismasi ve kararadir. Yatirim tavsiyesi degildir.*"

    return md, result
