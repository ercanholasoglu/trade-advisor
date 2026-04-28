"""
📋 Structured Logging + Health Monitoring
============================================
JSON structured logs. Health endpoint for monitoring.
"""

import json
import logging
import datetime
import os
import sys


class JSONFormatter(logging.Formatter):
    """Emit logs as JSON lines for parsing by monitoring tools."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logging(level: str = "INFO"):
    """
    Configure structured JSON logging for the whole application.
    Call once at startup.
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root.handlers.clear()

    # JSON handler → stderr
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)

    # Quiet noisy libraries
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)

    logging.getLogger("trade_bot").info("Structured logging initialized", extra={})


def get_health() -> dict:
    """
    Health check endpoint data. Returns system status.
    """
    import time

    health = {
        "status": "healthy",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "uptime_seconds": time.monotonic(),
        "components": {},
    }

    # Check yfinance
    try:
        import yfinance as yf
        h = yf.Ticker("AAPL").history(period="1d")
        health["components"]["yfinance"] = {
            "status": "up" if not h.empty else "degraded",
            "last_price": round(float(h["Close"].iloc[-1]), 2) if not h.empty else None,
        }
    except Exception as e:
        health["components"]["yfinance"] = {"status": "down", "error": str(e)}

    # Check SQLite
    try:
        from db import _get_conn
        conn = _get_conn()
        conn.execute("SELECT 1").fetchone()
        health["components"]["sqlite"] = {"status": "up"}
    except Exception as e:
        health["components"]["sqlite"] = {"status": "down", "error": str(e)}

    # Check cache
    try:
        from tools.cache import cache_stats
        stats = cache_stats()
        health["components"]["cache"] = {"status": "up", **stats}
    except Exception as e:
        health["components"]["cache"] = {"status": "down", "error": str(e)}

    # Check HF token
    hf_token = os.environ.get("HF_TOKEN", "")
    health["components"]["hf_token"] = {
        "status": "configured" if hf_token else "missing",
        "length": len(hf_token),
    }

    # Check scheduler
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        health["components"]["apscheduler"] = {"status": "available"}
    except ImportError:
        health["components"]["apscheduler"] = {"status": "not_installed"}

    # Overall status
    down_count = sum(1 for c in health["components"].values() if c.get("status") == "down")
    if down_count > 0:
        health["status"] = "degraded"

    return health
