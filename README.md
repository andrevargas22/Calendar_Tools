# Calendar Tools

Personal repository with tools to manage my Google Calendar.

## Tools

### `sync/`
Automatically syncs my work calendar (Microsoft Teams) with Google Calendar. Runs twice a day via GitHub Actions.

**Stack:** Python, Google Calendar API, GitHub Actions.

### `agent/`
Natural language agent to create, search, and cancel events on my personal Google Calendar via chat.

**Stack:** Python, DeepSeek (function calling), Google Calendar API, MLflow.

## Structure

- `common/`: code shared between the two tools (Google Calendar integration).
- `sync/` and `agent/`: each tool, self-contained.

Both use the same Google service account, but different calendars configured.
