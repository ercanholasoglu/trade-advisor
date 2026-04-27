"""Tool: Haber ve sentiment analizi (DuckDuckGo + basit sentiment)."""

from smolagents import tool


@tool
def get_news_sentiment(ticker: str, company_name: str = "") -> str:
    """
    Searches for recent financial news about a ticker/company and analyzes sentiment.
    Uses web search to find latest news and provides sentiment scoring.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL', 'NVDA', 'TSLA')
        company_name: Full company name for better search results (e.g. 'Apple Inc', 'NVIDIA'). If empty, only ticker is used.

    Returns:
        JSON string with news articles, individual sentiment scores, and overall sentiment summary.
    """
    import json

    try:
        from duckduckgo_search import DDGS

        search_query = f"{ticker} {company_name} stock news financial".strip()

        with DDGS() as ddgs:
            results = list(ddgs.news(search_query, max_results=8))

        if not results:
            return json.dumps({
                "ticker": ticker.upper(),
                "error": "No recent news found",
                "news_count": 0,
            })

        # Basit keyword-based sentiment (FinBERT yerine lightweight)
        positive_words = {
            "surge", "soar", "jump", "gain", "rally", "rise", "bull", "bullish",
            "record", "high", "growth", "profit", "beat", "exceed", "upgrade",
            "buy", "outperform", "strong", "positive", "boom", "breakout",
            "optimistic", "upbeat", "recovery", "innovation", "expand",
        }
        negative_words = {
            "drop", "fall", "crash", "plunge", "decline", "loss", "bear", "bearish",
            "low", "weak", "miss", "downgrade", "sell", "underperform", "risk",
            "fear", "concern", "warning", "cut", "layoff", "recession", "debt",
            "slump", "volatile", "uncertainty", "lawsuit", "investigation",
        }

        articles = []
        sentiment_scores = []

        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            text = (title + " " + body).lower()

            pos_count = sum(1 for w in positive_words if w in text)
            neg_count = sum(1 for w in negative_words if w in text)

            if pos_count > neg_count:
                sentiment = "POSITIVE"
                score = min(1.0, pos_count * 0.2)
            elif neg_count > pos_count:
                sentiment = "NEGATIVE"
                score = max(-1.0, -neg_count * 0.2)
            else:
                sentiment = "NEUTRAL"
                score = 0.0

            sentiment_scores.append(score)

            articles.append({
                "title": title,
                "source": r.get("source", "Unknown"),
                "date": r.get("date", "Unknown"),
                "url": r.get("url", ""),
                "sentiment": sentiment,
                "sentiment_score": round(score, 2),
                "snippet": body[:200] + "..." if len(body) > 200 else body,
            })

        # Genel sentiment özeti
        avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        pos_articles = sum(1 for s in sentiment_scores if s > 0)
        neg_articles = sum(1 for s in sentiment_scores if s < 0)
        neutral_articles = sum(1 for s in sentiment_scores if s == 0)

        if avg_score > 0.2:
            overall_sentiment = "BULLISH"
        elif avg_score < -0.2:
            overall_sentiment = "BEARISH"
        else:
            overall_sentiment = "NEUTRAL"

        result = {
            "ticker": ticker.upper(),
            "news_count": len(articles),
            "overall_sentiment": overall_sentiment,
            "avg_sentiment_score": round(avg_score, 3),
            "sentiment_breakdown": {
                "positive": pos_articles,
                "negative": neg_articles,
                "neutral": neutral_articles,
            },
            "articles": articles,
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})
