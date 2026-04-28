"""
Tool: BIST Sektör Korelasyon Matrisi
======================================
Sektör endeksleri arasındaki korelasyonu hesaplar.
Hangi sektörler birlikte hareket ediyor? Diversifikasyon analizi.
"""

from smolagents import tool
from tools.cache import cache

TTL_CORRELATION = 3600  # 1 saat


@cache(ttl=TTL_CORRELATION)
def _cached_sector_correlation(period: str, interval: str) -> str:
    import yfinance as yf
    import numpy as np
    import json

    # BIST sector indices
    sector_indices = {
        "BIST Banka": "XBANK.IS",
        "BIST Sınai": "XUSIN.IS",
        "BIST Teknoloji": "XUTEK.IS",
        "BIST Gıda": "XGIDA.IS",
        "BIST Hizmet": "XUHIZ.IS",
        "BIST Elektrik": "XELKT.IS",
        "BIST Kimya": "XKMYA.IS",
        "BIST Metal": "XMESY.IS",
        "BIST Temettü": "XTMTU.IS",
        "BIST 100": "XU100.IS",
    }

    # Fetch closing prices
    prices = {}
    valid_sectors = []
    for name, ticker in sector_indices.items():
        try:
            h = yf.Ticker(ticker).history(period=period, interval=interval)
            if not h.empty and len(h) > 20:
                prices[name] = h["Close"].values
                valid_sectors.append(name)
        except Exception:
            continue

    if len(valid_sectors) < 2:
        return json.dumps({"error": "Yeterli sektör verisi bulunamadı", "valid_sectors": valid_sectors})

    # Align lengths
    min_len = min(len(prices[s]) for s in valid_sectors)
    aligned = np.array([prices[s][-min_len:] for s in valid_sectors])

    # Calculate returns
    returns = np.diff(aligned, axis=1) / aligned[:, :-1]

    # Correlation matrix
    corr = np.corrcoef(returns)

    # Individual stats
    sector_stats = []
    for i, name in enumerate(valid_sectors):
        ann_ret = float(np.mean(returns[i]) * 252 * 100)
        ann_vol = float(np.std(returns[i]) * np.sqrt(252) * 100)
        total_ret = float((aligned[i][-1] / aligned[i][0] - 1) * 100)
        sector_stats.append({
            "sector": name,
            "ticker": sector_indices[name].replace(".IS", ""),
            "current_value": round(float(aligned[i][-1]), 2),
            "period_return_pct": round(total_ret, 2),
            "annualized_return_pct": round(ann_ret, 1),
            "annualized_volatility_pct": round(ann_vol, 1),
        })

    # Format correlation matrix
    corr_matrix = {}
    for i, s1 in enumerate(valid_sectors):
        corr_matrix[s1] = {}
        for j, s2 in enumerate(valid_sectors):
            corr_matrix[s1][s2] = round(float(corr[i][j]), 3)

    # Find strongly correlated pairs
    high_corr_pairs = []
    low_corr_pairs = []
    for i in range(len(valid_sectors)):
        for j in range(i + 1, len(valid_sectors)):
            val = float(corr[i][j])
            pair = f"{valid_sectors[i]} ↔ {valid_sectors[j]}"
            if val > 0.8:
                high_corr_pairs.append({"pair": pair, "correlation": round(val, 3), "note": "Birlikte hareket ediyor"})
            elif val < 0.3:
                low_corr_pairs.append({"pair": pair, "correlation": round(val, 3), "note": "Diversifikasyon fırsatı"})

    # Sort by return for ranking
    sector_stats.sort(key=lambda x: x["period_return_pct"], reverse=True)

    return json.dumps({
        "period": period,
        "data_points": min_len,
        "sectors_analyzed": len(valid_sectors),
        "sector_stats": sector_stats,
        "correlation_matrix": corr_matrix,
        "high_correlation_pairs": high_corr_pairs,
        "low_correlation_pairs": low_corr_pairs,
        "diversification_tip": (
            "Düşük korelasyonlu sektörlerden seçim yaparak portföy riski azaltılabilir. "
            "Yüksek korelasyonlu sektörlere aynı anda ağırlık vermekten kaçının."
        ),
    }, indent=2, ensure_ascii=False)


@tool
def get_sector_correlation(period: str = "6mo", interval: str = "1d") -> str:
    """
    Calculates correlation matrix between BIST sector indices.
    Shows which sectors move together and which provide diversification.
    Includes BIST Banka, Sınai, Teknoloji, Gıda, Hizmet, Elektrik, Kimya, Metal, Temettü.

    Args:
        period: Data period for correlation calculation. Options: '3mo','6mo','1y','2y'. Default '6mo'.
        interval: Data interval. Options: '1d','1wk'. Default '1d'.

    Returns:
        JSON string with correlation matrix, sector stats, high/low correlation pairs.
    """
    return _cached_sector_correlation(period, interval)


def format_correlation_markdown(result_json: str) -> str:
    """Format correlation result as readable markdown."""
    import json
    result = json.loads(result_json) if isinstance(result_json, str) else result_json

    if "error" in result:
        return f"❌ {result['error']}"

    md = f"## 🔗 BIST Sektör Korelasyon Matrisi\n\n**Dönem:** {result['period']} | **Veri noktası:** {result['data_points']}\n\n"

    # Sector performance ranking
    md += "### 📊 Sektör Performans Sıralaması\n\n"
    md += "| # | Sektör | Değer | Dönem Getiri | Yıllık Getiri | Volatilite |\n|---|---|---|---|---|---|\n"
    for i, s in enumerate(result["sector_stats"], 1):
        emoji = "🟢" if s["period_return_pct"] > 0 else "🔴"
        md += f"| {emoji} {i} | **{s['sector']}** ({s['ticker']}) | {s['current_value']:,.0f} | {s['period_return_pct']:+.1f}% | {s['annualized_return_pct']:+.1f}% | {s['annualized_volatility_pct']:.1f}% |\n"

    # Correlation matrix
    sectors = list(result["correlation_matrix"].keys())
    short_names = [s.replace("BIST ", "") for s in sectors]
    md += "\n### 🔗 Korelasyon Matrisi\n\n"
    md += "| | " + " | ".join(short_names) + " |\n"
    md += "|---|" + "|".join(["---"] * len(sectors)) + "|\n"
    for s1, short1 in zip(sectors, short_names):
        row = f"| **{short1}** |"
        for s2 in sectors:
            val = result["correlation_matrix"][s1][s2]
            if s1 == s2:
                row += " 1.00 |"
            elif val >= 0.8:
                row += f" 🔴 {val:.2f} |"
            elif val >= 0.5:
                row += f" 🟡 {val:.2f} |"
            else:
                row += f" 🟢 {val:.2f} |"
        md += row + "\n"

    md += "\n*🔴 >0.8 yüksek korelasyon | 🟡 0.5-0.8 orta | 🟢 <0.5 düşük (diversifikasyon)*\n"

    # High correlation pairs
    if result["high_correlation_pairs"]:
        md += "\n### 🔴 Yüksek Korelasyon (birlikte hareket)\n\n"
        for p in result["high_correlation_pairs"]:
            md += f"- **{p['pair']}** — {p['correlation']:.3f}\n"

    # Low correlation pairs
    if result["low_correlation_pairs"]:
        md += "\n### 🟢 Düşük Korelasyon (diversifikasyon fırsatı)\n\n"
        for p in result["low_correlation_pairs"]:
            md += f"- **{p['pair']}** — {p['correlation']:.3f}\n"

    md += f"\n💡 {result.get('diversification_tip', '')}\n"
    md += "\n---\n⚠️ *Geçmiş korelasyon gelecekte değişebilir. Yatırım tavsiyesi değildir.*"
    return md
