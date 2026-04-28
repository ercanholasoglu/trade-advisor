"""
📈 Portfolio Optimizer — Markowitz Efficient Frontier
=====================================================
Mean-variance optimization: max Sharpe, min volatility portfolios.
Uses scipy.optimize for constrained optimization.
"""

import json
import numpy as np


def get_portfolio_data(tickers: list[str], period: str = "1y") -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Fetch historical returns for a list of tickers.
    Returns (returns_matrix, valid_tickers, mean_prices).
    """
    import yfinance as yf

    all_data = {}
    valid_tickers = []

    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker.strip()).history(period=period, interval="1d")
            if not hist.empty and len(hist) > 60:
                all_data[ticker.strip().upper()] = hist["Close"].values
                valid_tickers.append(ticker.strip().upper())
        except Exception:
            continue

    if len(valid_tickers) < 2:
        raise ValueError(f"Need at least 2 tickers with data. Got: {valid_tickers}")

    # Align lengths
    min_len = min(len(v) for v in all_data.values())
    prices = np.array([all_data[t][-min_len:] for t in valid_tickers])

    # Calculate daily returns
    returns = np.diff(prices, axis=1) / prices[:, :-1]

    return returns, valid_tickers, prices[:, -1]


def optimize_portfolio(tickers: list[str], period: str = "1y",
                        risk_free_rate: float = 0.04,
                        num_portfolios: int = 5000) -> dict:
    """
    Run Markowitz portfolio optimization.

    Args:
        tickers: List of ticker symbols (min 2)
        period: Historical data period ('6mo', '1y', '2y')
        risk_free_rate: Annual risk-free rate (default 4%)
        num_portfolios: Number of random portfolios to simulate

    Returns:
        Dict with optimal portfolios, efficient frontier points, and correlation matrix.
    """
    from scipy.optimize import minimize

    returns, valid_tickers, current_prices = get_portfolio_data(tickers, period)
    n_assets = len(valid_tickers)

    # Expected returns (annualized)
    mean_returns = np.mean(returns, axis=1) * 252
    cov_matrix = np.cov(returns) * 252

    # Correlation matrix
    corr_matrix = np.corrcoef(returns)

    def portfolio_performance(weights):
        """Calculate return, volatility, sharpe."""
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
        return ret, vol, sharpe

    def neg_sharpe(weights):
        return -portfolio_performance(weights)[2]

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Constraints: weights sum to 1
    constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    bounds = tuple((0, 1) for _ in range(n_assets))  # No short selling

    # Initial guess: equal weight
    init_weights = np.array([1.0 / n_assets] * n_assets)

    # 1. Maximum Sharpe Ratio Portfolio
    max_sharpe_result = minimize(neg_sharpe, init_weights,
                                  method="SLSQP", bounds=bounds, constraints=constraints)
    max_sharpe_weights = max_sharpe_result.x
    ms_ret, ms_vol, ms_sharpe = portfolio_performance(max_sharpe_weights)

    # 2. Minimum Volatility Portfolio
    min_vol_result = minimize(portfolio_volatility, init_weights,
                               method="SLSQP", bounds=bounds, constraints=constraints)
    min_vol_weights = min_vol_result.x
    mv_ret, mv_vol, mv_sharpe = portfolio_performance(min_vol_weights)

    # 3. Monte Carlo simulation for efficient frontier visualization
    mc_results = []
    for _ in range(num_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        ret, vol, sharpe = portfolio_performance(weights)
        mc_results.append({
            "return": round(float(ret * 100), 2),
            "volatility": round(float(vol * 100), 2),
            "sharpe": round(float(sharpe), 3),
            "weights": {valid_tickers[i]: round(float(weights[i]) * 100, 1)
                        for i in range(n_assets)},
        })

    # 4. Equal weight portfolio
    eq_ret, eq_vol, eq_sharpe = portfolio_performance(init_weights)

    # Format correlation matrix
    corr_formatted = {}
    for i, t1 in enumerate(valid_tickers):
        corr_formatted[t1] = {}
        for j, t2 in enumerate(valid_tickers):
            corr_formatted[t1][t2] = round(float(corr_matrix[i][j]), 3)

    # Individual asset stats
    asset_stats = []
    for i, ticker in enumerate(valid_tickers):
        ann_ret = float(mean_returns[i])
        ann_vol = float(np.sqrt(cov_matrix[i][i]))
        asset_stats.append({
            "ticker": ticker,
            "current_price": round(float(current_prices[i]), 2),
            "annualized_return": f"{ann_ret * 100:.1f}%",
            "annualized_volatility": f"{ann_vol * 100:.1f}%",
            "sharpe": round((ann_ret - risk_free_rate) / ann_vol, 2) if ann_vol > 0 else 0,
        })

    result = {
        "tickers": valid_tickers,
        "period": period,
        "risk_free_rate": f"{risk_free_rate * 100:.1f}%",
        "asset_stats": asset_stats,
        "max_sharpe_portfolio": {
            "weights": {valid_tickers[i]: round(float(max_sharpe_weights[i]) * 100, 1)
                        for i in range(n_assets)},
            "expected_return": f"{ms_ret * 100:.1f}%",
            "volatility": f"{ms_vol * 100:.1f}%",
            "sharpe_ratio": round(float(ms_sharpe), 3),
        },
        "min_volatility_portfolio": {
            "weights": {valid_tickers[i]: round(float(min_vol_weights[i]) * 100, 1)
                        for i in range(n_assets)},
            "expected_return": f"{mv_ret * 100:.1f}%",
            "volatility": f"{mv_vol * 100:.1f}%",
            "sharpe_ratio": round(float(mv_sharpe), 3),
        },
        "equal_weight_portfolio": {
            "weights": {t: round(100 / n_assets, 1) for t in valid_tickers},
            "expected_return": f"{eq_ret * 100:.1f}%",
            "volatility": f"{eq_vol * 100:.1f}%",
            "sharpe_ratio": round(float(eq_sharpe), 3),
        },
        "correlation_matrix": corr_formatted,
        "efficient_frontier_sample": sorted(mc_results, key=lambda x: x["sharpe"], reverse=True)[:20],
    }
    return result


def optimize_portfolio_json(tickers_str: str, period: str = "1y") -> str:
    """Optimize portfolio from comma-separated tickers string."""
    tickers = [t.strip() for t in tickers_str.split(",") if t.strip()]
    try:
        result = optimize_portfolio(tickers, period)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


def format_portfolio_markdown(result: dict) -> str:
    """Format portfolio optimization result as markdown."""
    if "error" in result:
        return f"❌ {result['error']}"

    md = f"""
## 📈 Portföy Optimizasyonu — Markowitz

**Varlıklar:** {', '.join(result['tickers'])} | **Dönem:** {result['period']} | **Risksiz Faiz:** {result['risk_free_rate']}

### 📊 Bireysel Varlık İstatistikleri

| Varlık | Fiyat | Yıllık Getiri | Volatilite | Sharpe |
|---|---|---|---|---|
"""
    for a in result["asset_stats"]:
        md += f"| **{a['ticker']}** | ${a['current_price']} | {a['annualized_return']} | {a['annualized_volatility']} | {a['sharpe']} |\n"

    # Max Sharpe
    ms = result["max_sharpe_portfolio"]
    md += f"\n### 🏆 Maksimum Sharpe Portföyü\n\n"
    md += f"**Getiri:** {ms['expected_return']} | **Vol:** {ms['volatility']} | **Sharpe:** {ms['sharpe_ratio']}\n\n"
    md += "| Varlık | Ağırlık |\n|---|---|\n"
    for t, w in sorted(ms["weights"].items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(w / 5)
        md += f"| **{t}** | {bar} {w}% |\n"

    # Min Vol
    mv = result["min_volatility_portfolio"]
    md += f"\n### 🛡️ Minimum Volatilite Portföyü\n\n"
    md += f"**Getiri:** {mv['expected_return']} | **Vol:** {mv['volatility']} | **Sharpe:** {mv['sharpe_ratio']}\n\n"
    md += "| Varlık | Ağırlık |\n|---|---|\n"
    for t, w in sorted(mv["weights"].items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(w / 5)
        md += f"| **{t}** | {bar} {w}% |\n"

    # Equal Weight
    eq = result["equal_weight_portfolio"]
    md += f"\n### ⚖️ Eşit Ağırlık Portföyü\n\n"
    md += f"**Getiri:** {eq['expected_return']} | **Vol:** {eq['volatility']} | **Sharpe:** {eq['sharpe_ratio']}\n"

    # Correlation matrix
    md += "\n### 🔗 Korelasyon Matrisi\n\n"
    tickers = result["tickers"]
    md += "| | " + " | ".join(tickers) + " |\n"
    md += "|---|" + "|".join(["---"] * len(tickers)) + "|\n"
    corr = result["correlation_matrix"]
    for t1 in tickers:
        row = f"| **{t1}** |"
        for t2 in tickers:
            val = corr[t1][t2]
            color = "🟢" if val < 0.3 else ("🟡" if val < 0.7 else "🔴")
            row += f" {color} {val} |"
        md += row + "\n"

    md += "\n---\n⚠️ *Geçmiş performans gelecek getiriyi garanti etmez. Yatırım tavsiyesi değildir.*"
    return md
