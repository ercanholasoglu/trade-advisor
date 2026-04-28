"""
🛡️ Error Handler — Graceful degradation for all tools
========================================================
Wraps tool functions with try/except, returns structured error JSON.
Fund Manager can continue analysis with partial data.
"""

import json
import functools
import traceback
import logging

logger = logging.getLogger("trade_bot")


def safe_tool(fallback_data: dict | None = None):
    """
    Decorator that wraps tool functions with error handling.
    On error, returns {"error": "...", "fallback": true, ...fallback_data}.
    
    Usage:
        @safe_tool(fallback_data={"overall_signal": "N/A"})
        def get_technical_indicators(ticker, period="3mo"):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error_msg = str(e)
                tb = traceback.format_exc()
                logger.error(f"Tool {func.__name__} failed: {error_msg}\n{tb}")

                error_response = {
                    "error": error_msg,
                    "fallback": True,
                    "tool": func.__name__,
                    "args": [str(a) for a in args],
                }
                if fallback_data:
                    error_response.update(fallback_data)

                return json.dumps(error_response, indent=2)
        return wrapper
    return decorator


class ToolError:
    """Utility class to check if a tool result is an error."""
    
    @staticmethod
    def is_error(result: str) -> bool:
        """Check if tool result contains an error."""
        try:
            data = json.loads(result)
            return "error" in data
        except Exception:
            return False

    @staticmethod
    def get_error_message(result: str) -> str | None:
        """Extract error message from tool result."""
        try:
            data = json.loads(result)
            return data.get("error")
        except Exception:
            return None

    @staticmethod
    def format_user_message(tool_name: str, error: str) -> str:
        """Format error for user display."""
        tool_names_tr = {
            "get_price_history": "Fiyat verisi",
            "get_technical_indicators": "Teknik analiz",
            "get_news_sentiment": "Haber sentiment",
            "get_fundamental_data": "Temel veriler",
            "calculate_risk_metrics": "Risk metrikleri",
            "get_market_overview": "Piyasa durumu",
            "get_bist_scanner": "BIST tarama",
            "get_kap_disclosures": "KAP bildirimleri",
            "get_daily_dashboard": "Günlük dashboard",
            "get_crypto_fundamentals": "Kripto temel verileri",
            "get_insider_trades": "İçeriden işlemler",
            "get_options_flow": "Opsiyon akışı",
            "get_macro_data": "Makro veriler",
        }
        name_tr = tool_names_tr.get(tool_name, tool_name)
        return f"⚠️ {name_tr} alınamadı: {error}. Diğer verilerle devam ediliyor."
