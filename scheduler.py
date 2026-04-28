"""
⏰ Scheduler — Watchlist scanner + multi-channel alerts
=========================================================
APScheduler checks watchlist every 15 minutes.
Sends alerts via Telegram, Email (SMTP), Discord webhook.
"""

import os
import json
import logging
import datetime

logger = logging.getLogger("trade_bot.scheduler")


def check_watchlist():
    """
    Check all active watchlist items against current prices/indicators.
    Trigger alerts for matched conditions.
    """
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
                message = f"🔔 {ticker} fiyatı ${current_price:.2f} — hedef ${alert_value:.2f} altına düştü!"

            elif alert_type == "price_above" and current_price > alert_value:
                triggered = True
                message = f"🔔 {ticker} fiyatı ${current_price:.2f} — hedef ${alert_value:.2f} üzerine çıktı!"

            elif alert_type == "rsi_below":
                # Calculate RSI
                closes = hist["Close"].values
                if len(closes) >= 15:
                    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
                    gains = [d for d in deltas[-14:] if d > 0]
                    losses = [-d for d in deltas[-14:] if d < 0]
                    avg_gain = sum(gains) / 14 if gains else 0
                    avg_loss = sum(losses) / 14 if losses else 0.001
                    rsi = 100 - (100 / (1 + avg_gain / avg_loss))
                    actual_value = round(rsi, 1)
                    if rsi < alert_value:
                        triggered = True
                        message = f"📊 {ticker} RSI: {rsi:.1f} — hedef {alert_value} altına düştü (oversold!)"

            elif alert_type == "rsi_above":
                closes = hist["Close"].values
                if len(closes) >= 15:
                    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
                    gains = [d for d in deltas[-14:] if d > 0]
                    losses = [-d for d in deltas[-14:] if d < 0]
                    avg_gain = sum(gains) / 14 if gains else 0
                    avg_loss = sum(losses) / 14 if losses else 0.001
                    rsi = 100 - (100 / (1 + avg_gain / avg_loss))
                    actual_value = round(rsi, 1)
                    if rsi > alert_value:
                        triggered = True
                        message = f"📊 {ticker} RSI: {rsi:.1f} — hedef {alert_value} üzerine çıktı (overbought!)"

            elif alert_type == "vix_above":
                try:
                    vix_hist = yf.Ticker("^VIX").history(period="2d")
                    vix_val = float(vix_hist["Close"].iloc[-1])
                    actual_value = round(vix_val, 2)
                    if vix_val > alert_value:
                        triggered = True
                        message = f"🚨 VIX: {vix_val:.1f} — hedef {alert_value} üzerine çıktı! Panik modu!"
                except Exception:
                    pass

            if triggered:
                channel = item.get("notification_channel", "telegram")
                config = json.loads(item.get("notification_config", "{}"))

                # Send notification
                send_notification(channel, config, message)

                # Log
                save_alert_history(
                    watchlist_id=item["id"],
                    ticker=ticker,
                    alert_type=alert_type,
                    alert_value=alert_value,
                    actual_value=actual_value,
                    message=message,
                    channel=channel,
                )
                logger.info(f"Alert triggered: {message}")

        except Exception as e:
            logger.error(f"Watchlist check error for {item.get('ticker')}: {e}")


def send_notification(channel: str, config: dict, message: str):
    """Send notification via the specified channel."""
    if channel == "telegram":
        send_telegram(config.get("bot_token", os.environ.get("TELEGRAM_BOT_TOKEN", "")),
                       config.get("chat_id", os.environ.get("TELEGRAM_CHAT_ID", "")),
                       message)
    elif channel == "email":
        send_email(config.get("smtp_server", "smtp.gmail.com"),
                    config.get("smtp_port", 587),
                    config.get("email_from", ""),
                    config.get("email_password", ""),
                    config.get("email_to", ""),
                    "Trade Bot Alert",
                    message)
    elif channel == "discord":
        send_discord(config.get("webhook_url", os.environ.get("DISCORD_WEBHOOK_URL", "")),
                      message)


def send_telegram(bot_token: str, chat_id: str, message: str):
    """Send Telegram message."""
    if not bot_token or not chat_id:
        logger.warning("Telegram config missing — bot_token or chat_id")
        return
    import requests
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        if not resp.ok:
            logger.error(f"Telegram send failed: {resp.text}")
    except Exception as e:
        logger.error(f"Telegram error: {e}")


def send_email(smtp_server: str, smtp_port: int, email_from: str,
                email_password: str, email_to: str, subject: str, body: str):
    """Send email via SMTP."""
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

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_from, email_password)
            server.send_message(msg)
        logger.info(f"Email sent to {email_to}")
    except Exception as e:
        logger.error(f"Email error: {e}")


def send_discord(webhook_url: str, message: str):
    """Send Discord webhook message."""
    if not webhook_url:
        logger.warning("Discord webhook URL missing")
        return
    import requests
    try:
        resp = requests.post(
            webhook_url,
            json={"content": message},
            timeout=10,
        )
        if not resp.ok:
            logger.error(f"Discord send failed: {resp.text}")
    except Exception as e:
        logger.error(f"Discord error: {e}")


def start_scheduler():
    """
    Start APScheduler with background jobs:
    1. Watchlist scan every 15 minutes
    2. Update actual prices daily at 00:00
    """
    try:
        from apscheduler.schedulers.background import BackgroundScheduler

        scheduler = BackgroundScheduler()

        # Watchlist scan every 15 minutes
        scheduler.add_job(check_watchlist, 'interval', minutes=15,
                          id='watchlist_scan', replace_existing=True)

        # Update actual prices daily
        from db import update_actual_prices
        scheduler.add_job(update_actual_prices, 'cron', hour=0, minute=30,
                          id='update_prices', replace_existing=True)

        scheduler.start()
        logger.info("Scheduler started: watchlist scan (15min), price update (daily)")
        return scheduler

    except ImportError:
        logger.warning("APScheduler not installed — scheduled jobs disabled")
        return None
    except Exception as e:
        logger.error(f"Scheduler start error: {e}")
        return None
