"""
B. Dynamic Strategy Optimizer — Adaptive indicator weights from ground truth
================================================================================
Uses historical ground truth results to learn which indicators performed
best in recent periods, then adjusts voting weights accordingly.

Instead of fixed 1/1/1/1/1 equal voting:
  - Runs ground truth evaluation per indicator
  - Computes accuracy per indicator over the last N signals
  - Adjusts weights: better-performing indicators get more influence
  - Feeds optimized weights back into the signal engine

This is the "self-improving" loop the system needs.
"""

import numpy as np
import logging

logger = logging.getLogger("trade_bot.optimizer")


def compute_indicator_accuracy(ticker: str, period_years: float = 2,
                                holding_days: int = 10) -> dict:
    """
    Walk through historical data and measure accuracy of EACH indicator independently.
    
    For each day where any indicator fires:
      - RSI: was its individual signal correct?
      - SMA: was its individual signal correct?
      - MACD: was its individual signal correct?
      - Bollinger: was its individual signal correct?
      - Volume: was its individual signal correct?
    
    Returns per-indicator accuracy + optimized weights.
    """
    from core.data_ingest import fetch_ohlcv
    from core.signal_engine import _rsi, _classify_rsi, _ema, _bollinger

    period_map = {0.5: "6mo", 1: "1y", 2: "2y", 3: "3y", 5: "5y"}
    result = fetch_ohlcv(ticker, period=period_map.get(period_years, "2y"), interval="1d")
    df = result.get("df")

    if df is None or len(df) < 80:
        return {"error": f"Insufficient data for {ticker}", "ticker": ticker}

    closes = df["Close"].values.astype(float)
    volumes = df["Volume"].values.astype(float)
    source = result.get("source", "unknown")

    # Pre-compute SMA series
    sma20 = [None] * len(closes)
    sma50 = [None] * len(closes)
    for i in range(19, len(closes)):
        sma20[i] = float(np.mean(closes[i - 19:i + 1]))
    for i in range(49, len(closes)):
        sma50[i] = float(np.mean(closes[i - 49:i + 1]))

    # Track each indicator's accuracy independently
    stats = {
        "RSI": {"correct": 0, "total": 0, "signals": []},
        "SMA": {"correct": 0, "total": 0, "signals": []},
        "MACD": {"correct": 0, "total": 0, "signals": []},
        "Bollinger": {"correct": 0, "total": 0, "signals": []},
        "Volume": {"correct": 0, "total": 0, "signals": []},
    }

    start = 50
    for i in range(start, len(closes) - holding_days):
        future_return = (closes[i + holding_days] - closes[i]) / closes[i] * 100

        # RSI signal at this point
        seg = closes[:i + 1]
        rsi_val = _rsi(seg, 14)
        rsi_sig = _classify_rsi(rsi_val)
        if rsi_sig in ("BULLISH", "OVERSOLD"):
            stats["RSI"]["total"] += 1
            if future_return > 0:
                stats["RSI"]["correct"] += 1
        elif rsi_sig in ("BEARISH", "OVERBOUGHT"):
            stats["RSI"]["total"] += 1
            if future_return < 0:
                stats["RSI"]["correct"] += 1

        # SMA crossover signal
        if sma20[i] is not None and sma50[i] is not None:
            if sma20[i] > sma50[i]:
                stats["SMA"]["total"] += 1
                if future_return > 0:
                    stats["SMA"]["correct"] += 1
            elif sma20[i] < sma50[i]:
                stats["SMA"]["total"] += 1
                if future_return < 0:
                    stats["SMA"]["correct"] += 1

        # MACD signal
        if i >= 26:
            ema12 = _ema(closes[:i+1], 12)
            ema26 = _ema(closes[:i+1], 26)
            macd_hist = (ema12 - ema26) - _ema(ema12 - ema26, 9)
            if len(macd_hist) > 0:
                h = float(macd_hist[-1])
                if h > 0:
                    stats["MACD"]["total"] += 1
                    if future_return > 0:
                        stats["MACD"]["correct"] += 1
                elif h < 0:
                    stats["MACD"]["total"] += 1
                    if future_return < 0:
                        stats["MACD"]["correct"] += 1

        # Bollinger signal
        bb = _bollinger(closes[:i+1])
        if bb["signal"] in ("BULLISH", "OVERSOLD"):
            stats["Bollinger"]["total"] += 1
            if future_return > 0:
                stats["Bollinger"]["correct"] += 1
        elif bb["signal"] in ("BEARISH", "OVERBOUGHT"):
            stats["Bollinger"]["total"] += 1
            if future_return < 0:
                stats["Bollinger"]["correct"] += 1

        # Volume signal
        if i >= 20:
            vol_avg = float(np.mean(volumes[i-19:i+1]))
            if vol_avg > 0:
                vol_ratio = float(volumes[i] / vol_avg)
                if vol_ratio > 1.5:
                    stats["Volume"]["total"] += 1
                    if future_return > 0:
                        stats["Volume"]["correct"] += 1
                elif vol_ratio < 0.5:
                    stats["Volume"]["total"] += 1
                    if future_return < 0:
                        stats["Volume"]["correct"] += 1

    # Compute accuracies
    accuracies = {}
    for name, s in stats.items():
        if s["total"] > 0:
            accuracies[name] = {
                "accuracy_pct": round(s["correct"] / s["total"] * 100, 1),
                "correct": s["correct"],
                "total": s["total"],
            }
        else:
            accuracies[name] = {"accuracy_pct": 50.0, "correct": 0, "total": 0}

    # Compute optimized weights
    # Weight = accuracy normalized so they sum to 1
    # Minimum weight = 0.05 (never fully ignore any indicator)
    raw_weights = {}
    for name, acc in accuracies.items():
        # Use accuracy above 50% (random baseline) as the "skill" measure
        skill = max(0.05, (acc["accuracy_pct"] - 40) / 100)  # 40% = minimum useful
        raw_weights[name] = skill

    total_w = sum(raw_weights.values())
    optimized_weights = {k: round(v / total_w, 3) for k, v in raw_weights.items()}

    # Rank indicators by performance
    ranked = sorted(accuracies.items(), key=lambda x: x[1]["accuracy_pct"], reverse=True)

    return {
        "ticker": ticker,
        "data_source": source,
        "period": f"{period_years} year(s)",
        "holding_days": holding_days,
        "indicator_accuracy": accuracies,
        "optimized_weights": optimized_weights,
        "default_weights": {"RSI": 0.2, "SMA": 0.2, "MACD": 0.2, "Bollinger": 0.2, "Volume": 0.2},
        "ranking": [{"indicator": name, "accuracy": acc["accuracy_pct"], "samples": acc["total"]}
                    for name, acc in ranked],
        "recommendation": f"Best: {ranked[0][0]} ({ranked[0][1]['accuracy_pct']}%), Worst: {ranked[-1][0]} ({ranked[-1][1]['accuracy_pct']}%)" if ranked else "N/A",
    }


def optimize_for_tickers(tickers: list[str], period_years: float = 2) -> dict:
    """Run optimization across multiple tickers and average the weights."""
    all_weights = []
    all_accuracies = {}
    ticker_results = []

    for ticker in tickers:
        r = compute_indicator_accuracy(ticker, period_years)
        ticker_results.append(r)
        if "error" not in r:
            all_weights.append(r["optimized_weights"])
            for ind, acc in r["indicator_accuracy"].items():
                if ind not in all_accuracies:
                    all_accuracies[ind] = []
                all_accuracies[ind].append(acc["accuracy_pct"])

    if not all_weights:
        return {"error": "No valid results", "tickers": tickers}

    # Average weights across tickers
    avg_weights = {}
    for key in all_weights[0]:
        avg_weights[key] = round(np.mean([w.get(key, 0.2) for w in all_weights]), 3)

    # Normalize
    tw = sum(avg_weights.values())
    avg_weights = {k: round(v / tw, 3) for k, v in avg_weights.items()}

    # Average accuracies
    avg_acc = {k: round(float(np.mean(v)), 1) for k, v in all_accuracies.items()}

    return {
        "tickers": tickers,
        "period": f"{period_years} year(s)",
        "average_weights": avg_weights,
        "average_accuracy": avg_acc,
        "per_ticker": ticker_results,
    }
