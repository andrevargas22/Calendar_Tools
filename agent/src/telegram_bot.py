"""
Telegram webhook receiver for Calendar Agent.

No Telegram SDK — same "plain loop, call the HTTP API directly" philosophy as
agent.py's DeepSeek integration: `requests` against Telegram's Bot API for
sending replies, and Flask for receiving updates. Single authorized user
(the repo owner), single Cloud Run instance (see Dockerfile), in-memory
per-chat session state — no database, no message queue.

Processing happens synchronously inside the webhook request: DeepSeek +
Google Calendar calls finish in a few seconds, comfortably inside both
Telegram's webhook timeout (~60s) and Cloud Run's default request timeout.
"""

import hmac
import logging

import requests
from flask import Flask, request

from agent.src.agent import CONFIRM_YES, CalendarAgent
from agent.src.config import get_config
from agent.src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

app = Flask(__name__)

_agent = None  # lazily constructed: CalendarAgent() touches MLflow/network, so
               # the process can start (and pass Cloud Run's health check)
               # before paying that cost on the first real message
_sessions: dict = {}  # chat_id -> {"messages": list, "pending": PendingAction | None}

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


def _get_agent() -> CalendarAgent:
    global _agent
    if _agent is None:
        _agent = CalendarAgent()
    return _agent


def _send_message(token: str, chat_id: int, text: str) -> None:
    try:
        requests.post(_TELEGRAM_API.format(token=token), json={"chat_id": chat_id, "text": text}, timeout=15)
    except Exception:
        logger.exception("Failed to send Telegram reply")


def _handle_incoming(chat_id: int, text: str):
    session = _sessions.setdefault(chat_id, {"messages": [], "pending": None})

    agent = _get_agent()
    if session["pending"] is not None:
        confirmed = text.strip().lower() in CONFIRM_YES
        result = agent.confirm_pending_action(session["messages"], session["pending"], confirmed)
    else:
        session["messages"].append({"role": "user", "content": text})
        result = agent.handle_turn(session["messages"])

    session["pending"] = result.pending
    return result.text


@app.get("/health")
def health():
    return "ok", 200


@app.post("/webhook")
def webhook():
    config = get_config()
    expected_secret = config.get_telegram_webhook_secret()
    got_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")

    if not expected_secret or not hmac.compare_digest(got_secret, expected_secret):
        logger.warning("Rejected webhook call: missing or invalid secret token")
        return "forbidden", 403

    update = request.get_json(silent=True) or {}
    message = update.get("message") or update.get("edited_message")
    if not message:
        return "ok", 200

    chat_id = message.get("chat", {}).get("id")
    from_id = message.get("from", {}).get("id")
    text = message.get("text", "")
    allowed_id = config.get_telegram_allowed_user_id()

    if allowed_id is None or from_id != allowed_id or chat_id != allowed_id or not text:
        logger.warning(f"Ignored message from unauthorized or empty sender: {from_id}")
        return "ok", 200

    try:
        reply = _handle_incoming(chat_id, text)
    except Exception:
        logger.exception("Unhandled error processing Telegram update")
        reply = "Ocorreu um erro interno ao processar sua mensagem."

    _send_message(config.get_telegram_bot_token(), chat_id, reply)
    return "ok", 200
