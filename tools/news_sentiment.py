"""Tool: Haber ve sentiment analizi — FinBERT + keyword fallback."""

from smolagents import tool


@tool
def get_news_sentiment(ticker: str, company_name: str = "") -> str:
    """
    Searches for recent financial news about a ticker/company and analyzes sentiment.
    Uses FinBERT (ProsusAI/finbert) via HF Inference API for accurate NLP sentiment.
    Falls back to keyword-based sentiment if FinBERT is unavailable.

    Args:
        ticker: Stock ticker symbol (e.g. 'AAPL', 'NVDA', 'TSLA')
        company_name: Full company name for better search results (e.g. 'Apple Inc', 'NVIDIA'). If empty, only ticker is used.

    Returns:
        JSON string with news articles, individual sentiment scores, and overall sentiment summary.
    """
    import json
    import os

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

        # Try FinBERT via HF Inference API
        finbert_available = False
        hf_token = os.environ.get("HF_TOKEN", "")

        if hf_token:
            try:
                import requests
                finbert_url = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
                # Test with first headline
                test_resp = requests.post(
                    finbert_url,
                    headers={"Authorization": f"Bearer {hf_token}"},
                    json={"inputs": results[0].get("title", "test")},
                    timeout=10,
                )
                if test_resp.ok and isinstance(test_resp.json(), list):
                    finbert_available = True
            except Exception:
                pass

        articles = []
        sentiment_scores = []

        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            text_for_sentiment = title[:512]  # FinBERT has token limit

            if finbert_available:
                sentiment, score = _finbert_sentiment(text_for_sentiment, hf_token)
            else:
                sentiment, score = _keyword_sentiment(title + " " + body)

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
            "sentiment_method": "FinBERT" if finbert_available else "keyword",
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


def _finbert_sentiment(text: str, hf_token: str) -> tuple[str, float]:
    """Get sentiment from FinBERT via HF Inference API."""
    import requests

    try:
        resp = requests.post(
            "https://api-inference.huggingface.co/models/ProsusAI/finbert",
            headers={"Authorization": f"Bearer {hf_token}"},
            json={"inputs": text},
            timeout=15,
        )
        if resp.ok:
            results = resp.json()
            if isinstance(results, list) and len(results) > 0:
                # Results is [[{"label": "positive", "score": 0.94}, ...]]
                if isinstance(results[0], list):
                    results = results[0]

                # Find best label
                best = max(results, key=lambda x: x.get("score", 0))
                label = best.get("label", "neutral").lower()
                confidence = best.get("score", 0.5)

                if label == "positive":
                    return "POSITIVE", confidence
                elif label == "negative":
                    return "NEGATIVE", -confidence
                else:
                    return "NEUTRAL", 0.0
    except Exception:
        pass

    # Fallback to keyword if FinBERT fails for this specific text
    return _keyword_sentiment(text)


def _keyword_sentiment(text: str) -> tuple[str, float]:
    """Fallback keyword-based sentiment analysis."""
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

    text_lower = text.lower()
    pos_count = sum(1 for w in positive_words if w in text_lower)
    neg_count = sum(1 for w in negative_words if w in text_lower)

    if pos_count > neg_count:
        return "POSITIVE", min(1.0, pos_count * 0.2)
    elif neg_count > pos_count:
        return "NEGATIVE", max(-1.0, -neg_count * 0.2)
    else:
        return "NEUTRAL", 0.0
