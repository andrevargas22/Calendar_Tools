# Calendar Tools

Personal repository with tools to manage my Google Calendar.

## Tools

### `sync/`
Automatically syncs my work calendar (Microsoft Teams) with Google Calendar. Runs twice a day via GitHub Actions.

**Stack:** Python, Google Calendar API, GitHub Actions.

### `agent/`
Natural language agent to create, search, and cancel events on my personal Google Calendar via chat. Deployed as a Telegram bot (webhook, single authorized user) on Google Cloud Run — see `agent/src/telegram_bot.py`, `agent/scripts/set_webhook.py`, and `.github/workflows/deploy-bot.yml`. Also runnable locally as a terminal REPL via `make agent-run`.

**Stack:** Python, DeepSeek (function calling), Google Calendar API, MLflow, Flask (Telegram webhook), Cloud Run.

## Structure

- `common/`: code shared between the two tools (Google Calendar integration).
- `sync/` and `agent/`: each tool, self-contained.

Both use the same Google service account, but different calendars configured.
