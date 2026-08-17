"""
One-off script: point Telegram's webhook at the deployed Cloud Run URL.

Usage:
    python -m agent.scripts.set_webhook https://<cloud-run-url>

Run once after the first deploy, and again only if the Cloud Run URL
changes. Reads TELEGRAM_BOT_TOKEN / TELEGRAM_WEBHOOK_SECRET from the same
.env used by the rest of the agent.
"""

import sys

import requests

from agent.src.config import get_config


def main():
    if len(sys.argv) != 2:
        print("Usage: python -m agent.scripts.set_webhook <https://cloud-run-url>")
        sys.exit(1)

    config = get_config()
    token = config.get_telegram_bot_token()
    secret = config.get_telegram_webhook_secret()
    if not token or not secret:
        print("TELEGRAM_BOT_TOKEN and TELEGRAM_WEBHOOK_SECRET must be set (see .env.example).")
        sys.exit(1)

    url = sys.argv[1].rstrip("/") + "/webhook"
    response = requests.post(
        f"https://api.telegram.org/bot{token}/setWebhook",
        json={"url": url, "secret_token": secret},
        timeout=10,
    )
    response.raise_for_status()
    print(response.json())


if __name__ == "__main__":
    main()
