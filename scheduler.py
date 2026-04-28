"""
⏰ Scheduler v2 — Zamanlı Raporlar + Multi-Channel Alerts
============================================================
- Sabah raporu 09:30 (Türkiye) — BIST açılış öncesi tarama
- Kapanış raporu 18:30 — günün özeti
- Saatlik watchlist scan (borsa saatleri: 10:00-18:00 weekdays)
- Kanallar: Telegram, Email, Discord, WhatsApp (Twilio)
"""

import os
import json
import logging
import datetime

logger = logging.getLogger("trade_bot.scheduler")

# ═══════════════════════════════════════════════════════════════
# NOTIFICATION CHANNELS
# ═══════════════════════════════════════════════════════════════

def send_notification(channel: str, config: dict, message: str):
    """Route notification to the right channel."""
    if channel == "telegram":
        send_telegram(
            config.get("bot_token", os.environ.get("TELEGRAM_BOT_TOKEN", "")),
            config.get("chat_id", os.environ.get("TELEGRAM_CHAT_ID", "")),
            message)
    elif channel == "email":
        send_email(
            config.get("smtp_server", "smtp.gmail.com"),
            config.get("smtp_port", 587),
            config.get("email_from", ""),
            config.get("email_password", ""),
            config.get("email_to", ""),
            "🤖 Trade Bot Alert", message)
    elif channel == "discord":
        send_discord(
            config.get("webhook_url", os.environ.get("DISCORD_WEBHOOK_URL", "")),
            message)
    elif channel == "whatsapp":
        send_whatsapp(
            config.get("account_sid", os.environ.get("TWILIO_ACCOUNT_SID", "")),
            config.get("auth_token", os.environ.get("TWILIO_AUTH_TOKEN", "")),
            config.get("from_number", os.environ.get("TWILIO_WHATSAPP_FROM", "")),
            config.get("to_number", os.environ.get("TWILIO_WHATSAPP_TO", "")),
            message)


def send_telegram(bot_token: str, chat_id: str, message: str):
    if not bot_token or not chat_id:
        logger.warning("Telegram config missing")
        return
    import requests
    try:
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {e}")


def send_email(smtp_server, smtp_port, email_from, email_password, email_to, subject, body):
    if not email_from or not email_to:
        logger.warning("Email config missing")
        return
    try:
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = email_from
        msg["To"] = email_to
        with smtplib.SMTP(smtp_server, int(smtp_port)) as server:
            server.starttls()
            server.login(email_from, email_password)
            server.send_message(msg)
    except Exception as e:
        logger.error(f"Email error: {e}")


def send_discord(webhook_url: str, message: str):
    if not webhook_url:
        logger.warning("Discord webhook missing")
        return
    import requests
    try:
        requests.post(webhook_url, json={"content": message}, timeout=10)
    except Exception as e:
        logger.error(f"Discord error: {e}")


def send_whatsapp(account_sid: str, auth_token: str, from_number: str, to_number: str, message: str):
    """Send WhatsApp via Twilio API."""
    if not account_sid or not auth_token or not from_number or not to_number:
        logger.warning("Twilio WhatsApp config missing")
        return
    import requests
    try:
        url = f"https://api.twilio.com/2010-04-01/Accounts/{account_sid}/Messages.json"
        data = {
            "From": f"whatsapp:{from_number}",
            "To": f"whatsapp:{to_number}",
            "Body": message,
        }
        resp = requests.post(url, data=data, auth=(account_sid, auth_token), timeout=15)
        if resp.ok:
            logger.info(f"WhatsApp sent to {to_number}")
        else:
            logger.error(f"WhatsApp error: {resp.text}")
    except Exception as e:
        logger.error(f"WhatsApp error: {e}")


def _broadcast(message: str):
    """Send to all configured default channels."""
    channels = []
    if os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
        channels.append(("telegram", {}))
    if os.environ.get("DISCORD_WEBHOOK_URL"):
        channels.append(("discord", {}))
    if os.environ.get("TWILIO_ACCOUNT_SID"):
        channels.append(("whatsapp", {}))
    for ch, cfg in channels:
        try:
            send_notification(ch, cfg, message)
        except Exception as e:
            logger.error(f"Broadcast {ch} error: {e}")


# ═══════════════════════════════════════════════════════════════
# WATCHLIST CHECKER
# ═══════════════════════════════════════════════════════════════

def check_watchlist():
    """Check active watchlist items against current data."""
    import yfinance as yf
    from db import get_watchlist, save_alert_history

    items = get_watchlist(active_only=True)
    if not items:
        return

    for item in items:
        try:
            ticker = item["ticker"]
            alert_type = item["alert_type"]
            alert_value = item["alert_value"]
            hist = yf.Ticker(ticker).history(period="5d", interval="1d")
            if hist.empty:
                continue

            current_price = float(hist["Close"].iloc[-1])
            triggered = False
            actual_value = current_price
            message = ""

            if alert_type == "price_below" and current_price < alert_value:
                triggered = True
                message = f"🔔 {ticker} fiyatı ₺{current_price:.2f} — hedef ₺{alert_value:.2f} altına düştü!"
            elif alert_type == "price_above" and current_price > alert_value:
                triggered = True
                message = f"🔔 {ticker} fiyatı ₺{current_price:.2f} — hedef ₺{alert_value:.2f} üzerine çıktı!"
            elif alert_type in ("rsi_below", "rsi_above"):
                closes = hist["Close"].values
                if len(closes) >= 15:
                    from tools.bist_scanner import _calc_rsi_simple
                    rsi = _calc_rsi_simple(closes[-15:])
                    actual_value = rsi
                    if alert_type == "rsi_below" and rsi < alert_value:
                        triggered = True
                        message = f"📊 {ticker} RSI: {rsi:.1f} — hedef {alert_value} altına düştü (oversold!)"
                    elif alert_type == "rsi_above" and rsi > alert_value:
                        triggered = True
                        message = f"📊 {ticker} RSI: {rsi:.1f} — hedef {alert_value} üzerine çıktı (overbought!)"
            elif alert_type == "vix_above":
                try:
                    vix_val = float(yf.Ticker("^VIX").history(period="2d")["Close"].iloc[-1])
                    actual_value = vix_val
                    if vix_val > alert_value:
                        triggered = True
                        message = f"🚨 VIX: {vix_val:.1f} — hedef {alert_value} aşıldı!"
                except Exception:
                    pass

            if triggered:
                channel = item.get("notification_channel", "telegram")
                config = json.loads(item.get("notification_config", "{}"))
                send_notification(channel, config, message)
                save_alert_history(
                    watchlist_id=item["id"], ticker=ticker,
                    alert_type=alert_type, alert_value=alert_value,
                    actual_value=actual_value, message=message, channel=channel)
                logger.info(f"Alert: {message}")

        except Exception as e:
            logger.error(f"Watchlist error {item.get('ticker')}: {e}")


# ═══════════════════════════════════════════════════════════════
# TIMED REPORTS
# ═══════════════════════════════════════════════════════════════

def generate_morning_report():
    """
    Sabah Raporu (09:30 Türkiye) — BIST açılış öncesi.
    KAP gece bildirimleri + makro takvim + piyasa durumu + AL/SAT sinyalleri.
    """
    logger.info("Generating morning report...")
    import yfinance as yf
    lines = ["🌅 *SABAH RAPORU — BIST Açılış Öncesi*", f"📅 {datetime.date.today().strftime('%d.%m.%Y')}", ""]

    # Global overnight
    try:
        futures = {"S&P 500 Futures": "ES=F", "NASDAQ Futures": "NQ=F", "DAX": "^GDAXI", "Nikkei": "^N225"}
        for name, tick in futures.items():
            h = yf.Ticker(tick).history(period="2d")
            if not h.empty and len(h) >= 2:
                chg = (float(h["Close"].iloc[-1]) - float(h["Close"].iloc[-2])) / float(h["Close"].iloc[-2]) * 100
                emoji = "🟢" if chg >= 0 else "🔴"
                lines.append(f"{emoji} {name}: {chg:+.2f}%")
    except Exception:
        pass

    lines.append("")

    # Gold + USD/TRY
    try:
        gold = float(yf.Ticker("GC=F").history(period="2d")["Close"].iloc[-1])
        usd = float(yf.Ticker("USDTRY=X").history(period="2d")["Close"].iloc[-1])
        ons_try = gold * usd
        lines.append(f"🥇 Altın: ${gold:.0f}/oz — ₺{ons_try/31.1035:.0f}/gram")
        lines.append(f"💵 USD/TRY: {usd:.4f}")
    except Exception:
        pass

    lines.append("")

    # BIST scan — quick AL/SAT
    try:
        from tools.bist_scanner import _scan_ticker_group, BIST30
        stocks, buys, sells = _scan_ticker_group(BIST30)
        if buys:
            lines.append(f"📈 *AL Sinyalleri ({len(buys)}):*")
            for b in buys[:5]:
                lines.append(f"  {b['ticker']} — RSI {b['rsi']} (oversold)")
        if sells:
            lines.append(f"📉 *SAT Sinyalleri ({len(sells)}):*")
            for s in sells[:5]:
                lines.append(f"  {s['ticker']} — RSI {s['rsi']} (overbought)")
        if not buys and not sells:
            lines.append("📊 Bugün belirgin AL/SAT sinyali yok.")
    except Exception as e:
        lines.append(f"⚠️ BIST tarama hatası: {e}")

    lines.append("")

    # KAP overnight check
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            news = list(ddgs.news("KAP bildirim BIST", max_results=3))
        if news:
            lines.append("📋 *Son KAP Haberleri:*")
            for n in news:
                lines.append(f"  • {n.get('title', '')[:80]}")
    except Exception:
        pass

    lines.extend(["", "⚠️ _Yatırım tavsiyesi değildir._"])
    msg = "\n".join(lines)
    _broadcast(msg)
    logger.info("Morning report sent")
    return msg


def generate_closing_report():
    """
    Kapanış Raporu (18:30 Türkiye) — günün özeti.
    BIST kapanış, performans, en çok değişenler, yarın izlenecekler.
    """
    logger.info("Generating closing report...")
    import yfinance as yf
    lines = ["🌆 *KAPANIŞ RAPORU — Günün Özeti*", f"📅 {datetime.date.today().strftime('%d.%m.%Y')}", ""]

    # BIST 100
    try:
        h = yf.Ticker("XU100.IS").history(period="5d")
        if not h.empty and len(h) >= 2:
            cur, prev = float(h["Close"].iloc[-1]), float(h["Close"].iloc[-2])
            chg = (cur - prev) / prev * 100
            emoji = "🟢" if chg >= 0 else "🔴"
            lines.append(f"{emoji} *BIST 100:* {cur:,.0f} ({chg:+.2f}%)")
    except Exception:
        pass

    # Top movers
    try:
        from tools.bist_scanner import _scan_ticker_group, BIST30
        stocks, _, _ = _scan_ticker_group(BIST30)
        if stocks:
            gainers = stocks[:3]
            losers = stocks[-3:]
            lines.append("")
            lines.append("🏆 *Günün Yıldızları:*")
            for g in gainers:
                lines.append(f"  🟢 {g['ticker']}: {g['change_pct']:+.2f}% (₺{g['price_try']})")
            lines.append("📉 *Günün Düşenleri:*")
            for l in losers:
                lines.append(f"  🔴 {l['ticker']}: {l['change_pct']:+.2f}% (₺{l['price_try']})")
    except Exception:
        pass

    # Commodities
    try:
        gold = float(yf.Ticker("GC=F").history(period="2d")["Close"].iloc[-1])
        usd = float(yf.Ticker("USDTRY=X").history(period="2d")["Close"].iloc[-1])
        lines.append("")
        lines.append(f"🥇 Altın: ₺{gold*usd/31.1035:.0f}/gram")
        lines.append(f"💵 USD/TRY: {usd:.4f}")
    except Exception:
        pass

    # Signal accuracy if available
    try:
        from db import get_accuracy_stats
        stats = get_accuracy_stats()
        if "accuracy_7d" in stats:
            a = stats["accuracy_7d"]
            lines.append(f"\n📊 *Sinyal doğruluğu (7G):* {a['accuracy_pct']}% ({a['correct']}/{a['total']})")
    except Exception:
        pass

    lines.extend(["", "⚠️ _Yatırım tavsiyesi değildir._"])
    msg = "\n".join(lines)
    _broadcast(msg)
    logger.info("Closing report sent")
    return msg


def _is_trading_hours():
    """Check if within BIST trading hours (10:00-18:00 Turkey time, weekdays)."""
    try:
        import pytz
        turkey_tz = pytz.timezone("Europe/Istanbul")
        now = datetime.datetime.now(turkey_tz)
    except ImportError:
        # Fallback: UTC+3
        now = datetime.datetime.utcnow() + datetime.timedelta(hours=3)

    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False
    return 10 <= now.hour < 18


def hourly_watchlist_check():
    """Run watchlist check only during trading hours."""
    if _is_trading_hours():
        logger.info("Hourly watchlist check (trading hours)")
        check_watchlist()
    else:
        logger.debug("Skipping hourly check — outside trading hours")


# ═══════════════════════════════════════════════════════════════
# SCHEDULER STARTUP
# ═══════════════════════════════════════════════════════════════

def start_scheduler():
    """
    Start APScheduler with background jobs:
    1. Morning report: 06:30 UTC (09:30 Turkey)
    2. Closing report: 15:30 UTC (18:30 Turkey)
    3. Hourly watchlist scan: every hour (only during trading hours)
    4. Update actual prices: daily at 00:30 UTC
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler

        scheduler = BackgroundScheduler(timezone="UTC")

        # Morning report — 06:30 UTC = 09:30 Turkey
        scheduler.add_job(generate_morning_report, 'cron',
                          hour=6, minute=30, day_of_week='mon-fri',
                          id='morning_report', replace_existing=True)

        # Closing report — 15:30 UTC = 18:30 Turkey
        scheduler.add_job(generate_closing_report, 'cron',
                          hour=15, minute=30, day_of_week='mon-fri',
                          id='closing_report', replace_existing=True)

        # Hourly watchlist — every hour
        scheduler.add_job(hourly_watchlist_check, 'interval', hours=1,
                          id='hourly_watchlist', replace_existing=True)

        # Update actual prices daily
        from db import update_actual_prices
        scheduler.add_job(update_actual_prices, 'cron', hour=0, minute=30,
                          id='update_prices', replace_existing=True)

        scheduler.start()
        logger.info(
            "Scheduler started: "
            "morning report (09:30 TR), closing report (18:30 TR), "
            "hourly watchlist (trading hours), daily price update"
        )
        return scheduler

    except ImportError:
        logger.warning("APScheduler not installed — scheduled jobs disabled")
        return None
    except Exception as e:
        logger.error(f"Scheduler start error: {e}")
        return None
