"""
🗄️ Database — SQLite analysis logging + actual price tracking
================================================================
Stores every analysis result. APScheduler fills in actual prices
after 7 and 30 days to calculate signal accuracy.
"""

import sqlite3
import json
import os
import datetime
import threading

DB_PATH = os.environ.get("TRADE_BOT_DB", "/tmp/trade_bot.db")
_local = threading.local()
_last_db_path = None


def _get_conn():
    """Get thread-local database connection."""
    global _last_db_path
    current_path = os.environ.get("TRADE_BOT_DB", "/tmp/trade_bot.db")
    # Reconnect if DB path changed (e.g., in tests)
    if not hasattr(_local, "conn") or _local.conn is None or _last_db_path != current_path:
        if hasattr(_local, "conn") and _local.conn is not None:
            try:
                _local.conn.close()
            except Exception:
                pass
        _local.conn = sqlite3.connect(current_path, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _last_db_path = current_path
    return _local.conn


def init_db():
    """Create tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            signal TEXT,
            confidence INTEGER,
            entry_price REAL,
            stop_loss REAL,
            take_profit REAL,
            shares INTEGER,
            risk_reward TEXT,
            technical_signal TEXT,
            sentiment TEXT,
            fundamental_signal TEXT,
            market_regime TEXT,
            vix REAL,
            council_votes TEXT,  -- JSON
            reasoning TEXT,
            raw_report TEXT,     -- full report markdown
            actual_price_7d REAL,
            actual_price_30d REAL,
            signal_correct_7d INTEGER,  -- 1=correct, 0=wrong, NULL=pending
            signal_correct_30d INTEGER,
            return_7d_pct REAL,
            return_30d_pct REAL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            name TEXT,
            alert_type TEXT,        -- price_above, price_below, rsi_below, rsi_above, vix_above
            alert_value REAL,
            active INTEGER DEFAULT 1,
            notification_channel TEXT,  -- telegram, email, discord
            notification_config TEXT,   -- JSON with channel-specific config
            created_at TEXT DEFAULT (datetime('now')),
            last_triggered TEXT
        );

        CREATE TABLE IF NOT EXISTS alert_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            watchlist_id INTEGER,
            ticker TEXT,
            alert_type TEXT,
            alert_value REAL,
            actual_value REAL,
            message TEXT,
            channel TEXT,
            sent_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (watchlist_id) REFERENCES watchlist(id)
        );

        CREATE INDEX IF NOT EXISTS idx_analyses_ticker ON analyses(ticker);
        CREATE INDEX IF NOT EXISTS idx_analyses_timestamp ON analyses(timestamp);
        CREATE INDEX IF NOT EXISTS idx_watchlist_active ON watchlist(active);
    """)
    conn.commit()


def save_analysis(ticker: str, signal: str, confidence: int = 0,
                   entry_price: float = None, stop_loss: float = None,
                   take_profit: float = None, shares: int = None,
                   risk_reward: str = None, technical_signal: str = None,
                   sentiment: str = None, fundamental_signal: str = None,
                   market_regime: str = None, vix: float = None,
                   council_votes: list = None, reasoning: str = None,
                   raw_report: str = None) -> int:
    """Save an analysis result to the database. Returns the analysis ID."""
    conn = _get_conn()
    cursor = conn.execute("""
        INSERT INTO analyses (ticker, timestamp, signal, confidence, entry_price,
            stop_loss, take_profit, shares, risk_reward, technical_signal,
            sentiment, fundamental_signal, market_regime, vix,
            council_votes, reasoning, raw_report)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        ticker.upper(),
        datetime.datetime.now().isoformat(),
        signal, confidence, entry_price, stop_loss, take_profit, shares,
        risk_reward, technical_signal, sentiment, fundamental_signal,
        market_regime, vix,
        json.dumps(council_votes) if council_votes else None,
        reasoning, raw_report,
    ))
    conn.commit()
    return cursor.lastrowid


def update_actual_prices():
    """
    Check analyses that are 7+ or 30+ days old and fill in actual prices.
    Called by APScheduler daily.
    """
    import yfinance as yf

    conn = _get_conn()
    now = datetime.datetime.now()

    # Get analyses needing 7-day update
    rows_7d = conn.execute("""
        SELECT id, ticker, entry_price, signal, timestamp
        FROM analyses
        WHERE actual_price_7d IS NULL
        AND datetime(timestamp) < datetime('now', '-7 days')
        AND entry_price IS NOT NULL
    """).fetchall()

    for row in rows_7d:
        try:
            target_date = datetime.datetime.fromisoformat(row["timestamp"]) + datetime.timedelta(days=7)
            hist = yf.Ticker(row["ticker"]).history(
                start=target_date.strftime("%Y-%m-%d"),
                end=(target_date + datetime.timedelta(days=3)).strftime("%Y-%m-%d")
            )
            if not hist.empty:
                actual = float(hist["Close"].iloc[0])
                ret = (actual - row["entry_price"]) / row["entry_price"] * 100
                correct = 1 if ((row["signal"] in ("BUY", "STRONG_BUY") and ret > 0) or
                                (row["signal"] in ("SELL", "STRONG_SELL") and ret < 0)) else 0
                conn.execute("""
                    UPDATE analyses SET actual_price_7d=?, return_7d_pct=?, signal_correct_7d=?
                    WHERE id=?
                """, (round(actual, 2), round(ret, 2), correct, row["id"]))
        except Exception:
            continue

    # Get analyses needing 30-day update
    rows_30d = conn.execute("""
        SELECT id, ticker, entry_price, signal, timestamp
        FROM analyses
        WHERE actual_price_30d IS NULL
        AND datetime(timestamp) < datetime('now', '-30 days')
        AND entry_price IS NOT NULL
    """).fetchall()

    for row in rows_30d:
        try:
            target_date = datetime.datetime.fromisoformat(row["timestamp"]) + datetime.timedelta(days=30)
            hist = yf.Ticker(row["ticker"]).history(
                start=target_date.strftime("%Y-%m-%d"),
                end=(target_date + datetime.timedelta(days=3)).strftime("%Y-%m-%d")
            )
            if not hist.empty:
                actual = float(hist["Close"].iloc[0])
                ret = (actual - row["entry_price"]) / row["entry_price"] * 100
                correct = 1 if ((row["signal"] in ("BUY", "STRONG_BUY") and ret > 0) or
                                (row["signal"] in ("SELL", "STRONG_SELL") and ret < 0)) else 0
                conn.execute("""
                    UPDATE analyses SET actual_price_30d=?, return_30d_pct=?, signal_correct_30d=?
                    WHERE id=?
                """, (round(actual, 2), round(ret, 2), correct, row["id"]))
        except Exception:
            continue

    conn.commit()


def get_accuracy_stats(ticker: str = None) -> dict:
    """Get signal accuracy statistics, optionally filtered by ticker."""
    conn = _get_conn()
    where = "WHERE signal_correct_7d IS NOT NULL"
    params = []
    if ticker:
        where += " AND ticker = ?"
        params.append(ticker.upper())

    stats = {}

    # 7-day accuracy
    row = conn.execute(f"""
        SELECT COUNT(*) as total,
               SUM(signal_correct_7d) as correct_7d,
               AVG(return_7d_pct) as avg_return_7d
        FROM analyses {where}
    """, params).fetchone()

    if row and row["total"] > 0:
        stats["accuracy_7d"] = {
            "total": row["total"],
            "correct": row["correct_7d"] or 0,
            "accuracy_pct": round((row["correct_7d"] or 0) / row["total"] * 100, 1),
            "avg_return_pct": round(row["avg_return_7d"] or 0, 2),
        }

    # 30-day accuracy
    where_30 = "WHERE signal_correct_30d IS NOT NULL"
    params_30 = []
    if ticker:
        where_30 += " AND ticker = ?"
        params_30.append(ticker.upper())

    row_30 = conn.execute(f"""
        SELECT COUNT(*) as total,
               SUM(signal_correct_30d) as correct_30d,
               AVG(return_30d_pct) as avg_return_30d
        FROM analyses {where_30}
    """, params_30).fetchone()

    if row_30 and row_30["total"] > 0:
        stats["accuracy_30d"] = {
            "total": row_30["total"],
            "correct": row_30["correct_30d"] or 0,
            "accuracy_pct": round((row_30["correct_30d"] or 0) / row_30["total"] * 100, 1),
            "avg_return_pct": round(row_30["avg_return_30d"] or 0, 2),
        }

    # Recent analyses
    limit_where = "WHERE 1=1"
    limit_params = []
    if ticker:
        limit_where += " AND ticker = ?"
        limit_params.append(ticker.upper())

    recent = conn.execute(f"""
        SELECT ticker, timestamp, signal, confidence, entry_price,
               actual_price_7d, return_7d_pct, signal_correct_7d,
               actual_price_30d, return_30d_pct, signal_correct_30d
        FROM analyses {limit_where}
        ORDER BY timestamp DESC LIMIT 20
    """, limit_params).fetchall()

    stats["recent_analyses"] = [dict(r) for r in recent]
    return stats


def get_recent_analyses(limit: int = 20) -> list[dict]:
    """Get most recent analyses."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT * FROM analyses ORDER BY timestamp DESC LIMIT ?
    """, (limit,)).fetchall()
    return [dict(r) for r in rows]


# ──────── Watchlist ────────

def add_watchlist(ticker: str, name: str = "", alert_type: str = "price_below",
                   alert_value: float = 0, notification_channel: str = "telegram",
                   notification_config: dict = None) -> int:
    """Add a ticker to the watchlist with alert conditions."""
    conn = _get_conn()
    cursor = conn.execute("""
        INSERT INTO watchlist (ticker, name, alert_type, alert_value,
            notification_channel, notification_config)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (ticker.upper(), name, alert_type, alert_value,
          notification_channel, json.dumps(notification_config or {})))
    conn.commit()
    return cursor.lastrowid


def get_watchlist(active_only: bool = True) -> list[dict]:
    """Get watchlist items."""
    conn = _get_conn()
    where = "WHERE active = 1" if active_only else ""
    rows = conn.execute(f"SELECT * FROM watchlist {where} ORDER BY ticker").fetchall()
    return [dict(r) for r in rows]


def remove_watchlist(watchlist_id: int):
    """Deactivate a watchlist item."""
    conn = _get_conn()
    conn.execute("UPDATE watchlist SET active = 0 WHERE id = ?", (watchlist_id,))
    conn.commit()


def save_alert_history(watchlist_id: int, ticker: str, alert_type: str,
                        alert_value: float, actual_value: float,
                        message: str, channel: str):
    """Log a triggered alert."""
    conn = _get_conn()
    conn.execute("""
        INSERT INTO alert_history (watchlist_id, ticker, alert_type,
            alert_value, actual_value, message, channel)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (watchlist_id, ticker, alert_type, alert_value, actual_value, message, channel))
    conn.execute("UPDATE watchlist SET last_triggered = datetime('now') WHERE id = ?",
                  (watchlist_id,))
    conn.commit()


# Initialize DB on import
init_db()
