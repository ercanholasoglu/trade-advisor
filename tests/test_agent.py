"""
🧪 Agent Tests — Mock LLM + schema validation
================================================
Tests agent creation, output schema validation with mocked LLM responses.
"""

import json
import pytest
from unittest.mock import patch, MagicMock


def test_output_schema_validation():
    """Test that TradeReport schema validates correctly."""
    from output_schema import TradeReport, Signal, CouncilVote

    # Valid full report
    report = TradeReport(
        ticker="NVDA",
        company_name="NVIDIA",
        signal=Signal.BUY,
        confidence=75,
        entry_price=135.0,
        stop_loss=124.0,
        take_profit=152.0,
        shares=176,
        risk_reward="1:1.5",
        council_votes=[
            CouncilVote(advisor="bull", vote="BUY", conviction=8, key_argument="Strong momentum"),
            CouncilVote(advisor="bear", vote="SELL", conviction=5, key_argument="High valuation"),
            CouncilVote(advisor="mediator", vote="BUY", conviction=7, key_argument="Growth justifies premium"),
        ],
        consensus="MODERATE",
    )
    assert report.signal == Signal.BUY
    assert len(report.council_votes) == 3


def test_signal_enum():
    from output_schema import Signal

    assert Signal.BUY.value == "BUY"
    assert Signal.HOLD.value == "HOLD"
    assert Signal("STRONG_SELL") == Signal.STRONG_SELL


def test_confidence_bounds():
    from output_schema import TradeReport, Signal

    # Should work with 0-100
    r1 = TradeReport(ticker="TEST", confidence=0)
    assert r1.confidence == 0

    r2 = TradeReport(ticker="TEST", confidence=100)
    assert r2.confidence == 100

    # Should fail outside bounds
    with pytest.raises(Exception):
        TradeReport(ticker="TEST", confidence=101)


def test_parse_embedded_json():
    """Test parsing JSON embedded in markdown."""
    from output_schema import parse_trade_report

    raw = """
I've analyzed the stock and here's my report:

```json
{
    "ticker": "AAPL",
    "company_name": "Apple Inc",
    "signal": "HOLD",
    "confidence": 55,
    "entry_price": 175.0,
    "reasoning": "Mixed signals from council"
}
```

═══════════════════════════════════════════════════
🤖 TRADE ADVISOR REPORT
═══════════════════════════════════════════════════
📊 TICKER: AAPL
"""

    report, _ = parse_trade_report(raw)
    assert report is not None
    assert report.ticker == "AAPL"
    assert report.confidence == 55
    assert report.entry_price == 175.0


def test_error_handling_in_report():
    """Test that data_errors field works."""
    from output_schema import TradeReport, Signal

    report = TradeReport(
        ticker="TEST",
        signal=Signal.HOLD,
        confidence=40,
        data_errors=[
            "Technical analyst failed: timeout",
            "Options flow unavailable for BIST stocks",
        ],
        warnings=[
            "Reduced confidence due to missing technical data",
        ],
    )
    assert len(report.data_errors) == 2
    assert len(report.warnings) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
