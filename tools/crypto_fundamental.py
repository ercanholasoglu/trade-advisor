"""
Tool: Kripto temel analiz — CoinGecko Free API
================================================
Market cap, volume, developer activity, community score, price changes.
"""

from smolagents import tool


@tool
def get_crypto_fundamentals(crypto_id: str) -> str:
    """
    Fetches fundamental data for a cryptocurrency from CoinGecko free API.
    Includes market cap, volume, developer activity, community scores, and price changes.

    Args:
        crypto_id: CoinGecko coin ID or ticker symbol (e.g. 'bitcoin', 'ethereum', 'solana', 'BTC', 'ETH').
                   Common mappings: BTC→bitcoin, ETH→ethereum, SOL→solana, ADA→cardano, DOGE→dogecoin,
                   XRP→ripple, DOT→polkadot, AVAX→avalanche-2, MATIC→matic-network, LINK→chainlink.

    Returns:
        JSON string with market data, developer activity, community scores, and fundamental signals.
    """
    import json
    import requests

    # Ticker → CoinGecko ID mapping
    TICKER_MAP = {
        "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "ADA": "cardano",
        "DOGE": "dogecoin", "XRP": "ripple", "DOT": "polkadot", "AVAX": "avalanche-2",
        "MATIC": "matic-network", "LINK": "chainlink", "UNI": "uniswap", "ATOM": "cosmos",
        "LTC": "litecoin", "FIL": "filecoin", "NEAR": "near", "APT": "aptos",
        "ARB": "arbitrum", "OP": "optimism", "PEPE": "pepe", "SHIB": "shiba-inu",
        "BNB": "binancecoin", "TRX": "tron", "TON": "the-open-network",
        "SUI": "sui", "SEI": "sei-network", "INJ": "injective-protocol",
    }

    # Normalize input
    cid = crypto_id.strip().upper().replace("-USD", "").replace("-USDT", "")
    coin_id = TICKER_MAP.get(cid, crypto_id.lower().strip())

    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "true",
            "community_data": "true",
            "developer_data": "true",
            "sparkline": "false",
        }
        headers = {"accept": "application/json"}
        resp = requests.get(url, params=params, headers=headers, timeout=15)

        if resp.status_code == 429:
            return json.dumps({"error": "CoinGecko rate limit — try again in 60s", "crypto_id": coin_id})

        if not resp.ok:
            return json.dumps({"error": f"CoinGecko API error: {resp.status_code}", "crypto_id": coin_id})

        data = resp.json()
        md = data.get("market_data", {})
        dd = data.get("developer_data", {})
        cd = data.get("community_data", {})

        # Market data
        market = {
            "current_price_usd": md.get("current_price", {}).get("usd"),
            "market_cap_usd": md.get("market_cap", {}).get("usd"),
            "market_cap_rank": data.get("market_cap_rank"),
            "total_volume_24h": md.get("total_volume", {}).get("usd"),
            "circulating_supply": md.get("circulating_supply"),
            "total_supply": md.get("total_supply"),
            "max_supply": md.get("max_supply"),
            "ath_usd": md.get("ath", {}).get("usd"),
            "ath_change_pct": md.get("ath_change_percentage", {}).get("usd"),
            "atl_usd": md.get("atl", {}).get("usd"),
        }

        # Format market cap
        mc = market["market_cap_usd"]
        if mc:
            if mc >= 1e12:
                market["market_cap_formatted"] = f"${mc/1e12:.2f}T"
            elif mc >= 1e9:
                market["market_cap_formatted"] = f"${mc/1e9:.2f}B"
            elif mc >= 1e6:
                market["market_cap_formatted"] = f"${mc/1e6:.2f}M"

        # Price changes
        price_changes = {
            "1h": md.get("price_change_percentage_1h_in_currency", {}).get("usd"),
            "24h": md.get("price_change_percentage_24h"),
            "7d": md.get("price_change_percentage_7d"),
            "14d": md.get("price_change_percentage_14d"),
            "30d": md.get("price_change_percentage_30d"),
            "60d": md.get("price_change_percentage_60d"),
            "200d": md.get("price_change_percentage_200d"),
            "1y": md.get("price_change_percentage_1y"),
        }
        # Round
        price_changes = {k: round(v, 2) if v is not None else None for k, v in price_changes.items()}

        # Developer activity
        developer = {
            "forks": dd.get("forks"),
            "stars": dd.get("stars"),
            "subscribers": dd.get("subscribers"),
            "total_issues": dd.get("total_issues"),
            "closed_issues": dd.get("closed_issues"),
            "pull_requests_merged": dd.get("pull_requests_merged"),
            "pull_request_contributors": dd.get("pull_request_contributors"),
            "commit_count_4_weeks": dd.get("commit_count_4_weeks"),
            "code_additions_4_weeks": dd.get("code_additions_deletions_4_weeks", {}).get("additions"),
            "code_deletions_4_weeks": dd.get("code_additions_deletions_4_weeks", {}).get("deletions"),
        }

        # Community
        community = {
            "twitter_followers": cd.get("twitter_followers"),
            "reddit_subscribers": cd.get("reddit_subscribers"),
            "reddit_active_48h": cd.get("reddit_accounts_active_48h"),
            "telegram_members": cd.get("telegram_channel_user_count"),
        }

        # Scores
        scores = {
            "coingecko_score": data.get("coingecko_score"),
            "developer_score": data.get("developer_score"),
            "community_score": data.get("community_score"),
            "liquidity_score": data.get("liquidity_score"),
            "public_interest_score": data.get("public_interest_score"),
        }

        # Generate signals
        signals = []
        # Momentum signals
        chg_30d = price_changes.get("30d")
        chg_7d = price_changes.get("7d")
        if chg_30d is not None:
            if chg_30d > 30:
                signals.append("STRONG_MOMENTUM_UP")
            elif chg_30d > 10:
                signals.append("MOMENTUM_UP")
            elif chg_30d < -30:
                signals.append("STRONG_MOMENTUM_DOWN")
            elif chg_30d < -10:
                signals.append("MOMENTUM_DOWN")

        # Developer activity
        commits = developer.get("commit_count_4_weeks")
        if commits is not None:
            if commits > 100:
                signals.append("HIGH_DEV_ACTIVITY")
            elif commits < 5:
                signals.append("LOW_DEV_ACTIVITY")

        # Supply scarcity
        circ = market.get("circulating_supply")
        max_s = market.get("max_supply")
        if circ and max_s and max_s > 0:
            supply_ratio = circ / max_s
            if supply_ratio > 0.9:
                signals.append("HIGH_SUPPLY_RATIO")
            elif supply_ratio < 0.5:
                signals.append("LOW_SUPPLY_RATIO_POTENTIAL_INFLATION")

        # Volume / mcap ratio
        vol = market.get("total_volume_24h")
        mcap = market.get("market_cap_usd")
        if vol and mcap and mcap > 0:
            vol_mcap = vol / mcap
            if vol_mcap > 0.2:
                signals.append("HIGH_VOLUME_ACTIVITY")
            elif vol_mcap < 0.02:
                signals.append("LOW_VOLUME_ACTIVITY")

        result = {
            "crypto_id": coin_id,
            "name": data.get("name", ""),
            "symbol": data.get("symbol", "").upper(),
            "market": market,
            "price_changes": price_changes,
            "developer": developer,
            "community": community,
            "scores": scores,
            "fundamental_signals": signals,
            "categories": data.get("categories", []),
        }
        return json.dumps(result, indent=2)

    except requests.exceptions.Timeout:
        return json.dumps({"error": "CoinGecko timeout — API may be slow", "crypto_id": coin_id})
    except Exception as e:
        return json.dumps({"error": str(e), "crypto_id": coin_id})
