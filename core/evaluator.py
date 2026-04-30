"""
🎯 Ground Truth Evaluator — Model ne dedi? Gerçek ne oldu?
=============================================================
For each ticker:
  1. Walk back through historical data day by day
  2. At each day: generate the signal the engine WOULD have produced
  3. Record what price actually did over the next N days
  4. Compare prediction vs reality → verdict: ✅ or ❌

This is the definitive answer to: "Is this signal engine actually good?"
"""

import numpy as np
import datetime
import logging

logger = logging.getLogger("trade_bot.evaluator")


def _calc_rsi_at(closes, idx, period=14):
    """RSI at a specific index in the array."""
    if idx < period:
        return 50.0
    seg = closes[:idx + 1]
    deltas = np.diff(seg)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    return float(100 - (100 / (1 + avg_gain / avg_loss)))


def run_ground_truth_evaluation(ticker: str, period_years: float = 2,
                                 eval_horizons: list[int] | None = None,
                                 rsi_buy: float = 35, rsi_sell: float = 65,
                                 use_sma_filter: bool = True) -> dict:
    """
    Walk through historical data and evaluate every signal against ground truth.

    For each day where our engine would fire a BUY or SELL:
      - Record: date, signal, price, RSI, SMA state, confidence
      - Look ahead 5, 10, 20 days and record actual price changes
      - Verdict: did price move in the direction we predicted?

    Returns a rich dict with per-signal verdicts and aggregate stats.
    """
    from core.data_ingest import fetch_ohlcv

    if eval_horizons is None:
        eval_horizons = [5, 10, 20]

    period_map = {0.5: "6mo", 1: "1y", 2: "2y", 3: "3y", 5: "5y"}
    yf_period = period_map.get(period_years, "2y")

    result = fetch_ohlcv(ticker, period=yf_period, interval="1d")
    df = result.get("df")
    source = result.get("source", "unknown")

    if df is None or len(df) < 80:
        return {"error": f"Insufficient data for {ticker}", "ticker": ticker}

    closes = df["Close"].values.astype(float)
    dates = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
             for d in df.index]

    max_horizon = max(eval_horizons)
    start = 50  # need SMA50

    # Pre-compute SMA series
    sma20 = [None] * len(closes)
    sma50 = [None] * len(closes)
    for i in range(19, len(closes)):
        sma20[i] = float(np.mean(closes[i - 19:i + 1]))
    for i in range(49, len(closes)):
        sma50[i] = float(np.mean(closes[i - 49:i + 1]))

    evaluations: list[dict] = []

    for i in range(start, len(closes) - max_horizon):
        rsi = _calc_rsi_at(closes, i)
        price = float(closes[i])

        # Determine signal at this point
        if use_sma_filter:
            if sma20[i] is None or sma50[i] is None:
                continue
            sma_bull = sma20[i] > sma50[i]
            sma_bear = sma20[i] < sma50[i]
        else:
            sma_bull = sma_bear = True

        if rsi < rsi_buy and sma_bull:
            signal = "BUY"
        elif rsi > rsi_sell and sma_bear:
            signal = "SELL"
        else:
            continue

        # Ground truth: what actually happened?
        outcomes = {}
        for h in eval_horizons:
            future_price = float(closes[i + h])
            change_pct = (future_price - price) / price * 100

            if signal == "BUY":
                correct = change_pct > 0
            else:  # SELL
                correct = change_pct < 0

            outcomes[f"{h}d"] = {
                "future_price": round(future_price, 2),
                "change_pct": round(change_pct, 2),
                "direction": "UP" if change_pct > 0 else ("DOWN" if change_pct < 0 else "FLAT"),
                "correct": correct,
            }

        evaluations.append({
            "date": dates[i],
            "signal": signal,
            "price": round(price, 2),
            "rsi": round(rsi, 1),
            "sma20": round(sma20[i], 2) if sma20[i] else None,
            "sma50": round(sma50[i], 2) if sma50[i] else None,
            "outcomes": outcomes,
        })

    if not evaluations:
        return {"error": "No signals found in evaluation period", "ticker": ticker}

    # ── Aggregate metrics per horizon ──
    horizon_stats = {}
    for h in eval_horizons:
        key = f"{h}d"
        correct = sum(1 for e in evaluations if e["outcomes"][key]["correct"])
        total = len(evaluations)
        returns = [e["outcomes"][key]["change_pct"] for e in evaluations]
        buy_evals = [e for e in evaluations if e["signal"] == "BUY"]
        sell_evals = [e for e in evaluations if e["signal"] == "SELL"]

        buy_correct = sum(1 for e in buy_evals if e["outcomes"][key]["correct"])
        sell_correct = sum(1 for e in sell_evals if e["outcomes"][key]["correct"])

        # Directional returns (positive = signal was right)
        dir_returns = []
        for e in evaluations:
            chg = e["outcomes"][key]["change_pct"]
            dir_returns.append(chg if e["signal"] == "BUY" else -chg)

        horizon_stats[key] = {
            "horizon_days": h,
            "total_signals": total,
            "correct": correct,
            "accuracy_pct": round(correct / total * 100, 1),
            "buy_signals": len(buy_evals),
            "buy_correct": buy_correct,
            "buy_accuracy_pct": round(buy_correct / len(buy_evals) * 100, 1) if buy_evals else 0,
            "sell_signals": len(sell_evals),
            "sell_correct": sell_correct,
            "sell_accuracy_pct": round(sell_correct / len(sell_evals) * 100, 1) if sell_evals else 0,
            "avg_return_pct": round(float(np.mean(returns)), 2),
            "avg_directional_return_pct": round(float(np.mean(dir_returns)), 2),
            "median_return_pct": round(float(np.median(returns)), 2),
            "best_return_pct": round(float(max(returns)), 2),
            "worst_return_pct": round(float(min(returns)), 2),
            "std_return_pct": round(float(np.std(returns)), 2),
        }

    # ── Confidence calibration: group by RSI extremity ──
    # Closer to 0/100 RSI → higher conviction → should be more accurate
    high_conv = [e for e in evaluations if e["rsi"] < 25 or e["rsi"] > 75]
    low_conv = [e for e in evaluations if 30 <= e["rsi"] <= 70]
    ref_h = f"{eval_horizons[1]}d" if len(eval_horizons) > 1 else f"{eval_horizons[0]}d"
    hc_correct = sum(1 for e in high_conv if e["outcomes"][ref_h]["correct"])
    lc_correct = sum(1 for e in low_conv if e["outcomes"][ref_h]["correct"])

    calibration = {
        "high_conviction_signals": len(high_conv),
        "high_conviction_accuracy": round(hc_correct / len(high_conv) * 100, 1) if high_conv else 0,
        "low_conviction_signals": len(low_conv),
        "low_conviction_accuracy": round(lc_correct / len(low_conv) * 100, 1) if low_conv else 0,
        "reference_horizon": ref_h,
    }

    return {
        "ticker": ticker.upper(),
        "data_source": source,
        "period": f"{period_years} year(s)",
        "parameters": {
            "rsi_buy": rsi_buy,
            "rsi_sell": rsi_sell,
            "sma_filter": use_sma_filter,
            "eval_horizons": eval_horizons,
        },
        "total_evaluations": len(evaluations),
        "horizon_stats": horizon_stats,
        "calibration": calibration,
        "evaluations": evaluations,  # full detail
    }
