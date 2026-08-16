"""
Google Calendar functions, shared by sync/ and agent/.

Originally ported from Calendar_Sync/src/google_calendar.py to Calendar_Agent
(production-tested auth, retry, list/create/delete logic), then promoted here
so both tools use one implementation instead of two copies. Two deliberate
differences from the original Calendar_Sync module:

1. calendar_id (and the service account key) are explicit parameters on every
   function instead of module-level constants imported at build time — makes
   the module testable and usable with two different calendars (sync's work
   calendar, agent's personal one) from the same process.
2. get_calendar_service() re-raises on unexpected auth errors instead of
   silently logging and returning None (a gap in the original) — a caller
   that keeps running without a valid service is worse than a clear crash.

Datetimes are kept timezone-naive in local time (same convention as the
original Calendar_Sync), not because it's ideal but because it's a
consistent, already-proven convention.

Title masking (mask_titles=True) reproduces Calendar_Sync's LOG_MASK_TITLES
behavior: event titles are hashed before hitting the logs, since sync runs
unattended on GitHub Actions where log output isn't fully private. The agent
runs locally in a REPL, so it defaults mask_titles=False.
"""

import hashlib
import json
import logging
import random
import time
from datetime import datetime, timedelta

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def _mask_title(title: str) -> str:
    """Hash a title for privacy-masked logging (see module docstring)."""
    if not title:
        return title
    h = hashlib.sha256(title.encode("utf-8")).hexdigest()[:12]
    return f"EVENT[{h}]"


def get_calendar_service(service_account_key: str):
    """
    Initialize the Calendar API using a service account.

    Args:
        service_account_key: raw JSON string of the service account credentials.

    Returns:
        Google Calendar API service.

    Raises:
        ValueError: If credentials are missing or invalid.
        json.JSONDecodeError: If credentials JSON is malformed.
        Exception: If authentication fails for any other reason.
    """
    try:
        logger.info("Initializing Google Calendar API service...")
        if not service_account_key:
            raise ValueError("GOOGLE_SERVICE_ACCOUNT_KEY is not set")

        credentials_info = json.loads(service_account_key)
        required_keys = {"type", "project_id", "private_key", "client_email"}
        missing = [k for k in required_keys if not credentials_info.get(k)]
        if missing:
            raise ValueError(f"Service account credentials missing required fields: {missing}")

        credentials = service_account.Credentials.from_service_account_info(
            credentials_info, scopes=SCOPES
        )

        service = build("calendar", "v3", credentials=credentials)
        logger.info("Google Calendar API service initialized successfully")
        return service

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in service account credentials: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during authentication: {e}")
        raise


def _retry(callable_fn, *, max_attempts=4, base_delay=0.5, op_name="api_call"):
    """Simple exponential backoff retry for transient Google API errors."""
    attempt = 0
    while True:
        try:
            return callable_fn()
        except HttpError as e:
            status = getattr(e.resp, "status", None)
            if status in {429, 500, 502, 503, 504} and attempt < max_attempts - 1:
                delay = base_delay * (2**attempt) * (1 + random.random() * 0.25)
                logger.warning(
                    f"Transient error {status} during {op_name}, retrying in "
                    f"{delay:.2f}s (attempt {attempt + 1}/{max_attempts})"
                )
                time.sleep(delay)
                attempt += 1
                continue
            raise
        except Exception:
            if attempt < max_attempts - 1:
                delay = base_delay * (2**attempt) * (1 + random.random() * 0.25)
                logger.warning(
                    f"Error during {op_name}, retrying in {delay:.2f}s "
                    f"(attempt {attempt + 1}/{max_attempts})"
                )
                time.sleep(delay)
                attempt += 1
                continue
            raise


def get_google_events(svc, calendar_id: str, start: datetime, end: datetime) -> list[dict]:
    """
    Fetch events from Google Calendar in the given period.

    Args:
        svc: Google Calendar API service.
        calendar_id: target calendar id.
        start: start datetime (naive, local time).
        end: end datetime (naive, local time).

    Returns:
        List of dicts: {'id', 'titulo', 'inicio': datetime, 'fim': datetime}.

    Raises:
        HttpError: If the Google Calendar API request fails.
        Exception: If data processing fails.
    """
    try:
        extended_end = end + timedelta(days=1)
        logger.info(f"Fetching Google Calendar events from {start.date()} to {extended_end.date()}")

        def _list_call():
            return (
                svc.events()
                .list(
                    calendarId=calendar_id,
                    timeMin=start.isoformat() + "Z",
                    timeMax=extended_end.isoformat() + "Z",
                    singleEvents=True,
                    orderBy="startTime",
                    maxResults=2500,
                )
                .execute()
            )

        evs = _retry(_list_call, op_name="events.list").get("items", [])
        logger.info(f"Retrieved {len(evs)} raw events from Google Calendar")

        out = []
        for ev in evs:
            try:
                s = ev["start"].get("dateTime") or ev["start"].get("date")
                f = ev["end"].get("dateTime") or ev["end"].get("date")

                if not s or not f:
                    logger.warning(
                        f"Event with ID {ev.get('id', 'unknown')[:8]}... has missing "
                        "start/end times, skipping"
                    )
                    continue

                s = datetime.fromisoformat(s.replace("Z", ""))
                f = datetime.fromisoformat(f.replace("Z", ""))

                if s.tzinfo is not None:
                    s = s.astimezone().replace(tzinfo=None)
                if f.tzinfo is not None:
                    f = f.astimezone().replace(tzinfo=None)

                out.append(
                    {
                        "id": ev.get("id"),
                        "titulo": ev.get("summary", "Untitled Event"),
                        "inicio": s,
                        "fim": f,
                    }
                )
            except Exception as e:
                event_id_partial = ev.get("id", "unknown")[:8] if ev.get("id") else "unknown"
                logger.warning(f"Error processing event {event_id_partial}...: {e}")
                continue

        logger.info(f"Successfully processed {len(out)} events")
        return out

    except HttpError as e:
        logger.error(f"Google Calendar API error: {e}")
        if e.resp.status == 403:
            logger.error("Permission denied. Check if the service account has calendar access")
        elif e.resp.status == 404:
            logger.error("Calendar not found. Check your GOOGLE_CALENDAR_ID configuration")
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching events: {e}")
        raise


def criar_evento_google(svc, calendar_id: str, ev: dict, timezone: str, mask_titles: bool = False) -> dict:
    """Create an event in Google Calendar.

    Args:
        svc: Google Calendar API service.
        calendar_id: target calendar id.
        ev: dict with keys 'titulo', 'inicio', 'fim' (inicio/fim as ISO strings).
        timezone: IANA timezone name for the event (e.g. "America/Sao_Paulo").
        mask_titles: if True, hash the title before logging it (see module docstring).

    Returns:
        The created event resource (as returned by the Google Calendar API).
    """
    required_fields = ["titulo", "inicio", "fim"]
    missing_fields = [field for field in required_fields if not ev.get(field)]
    if missing_fields:
        logger.error(f"Invalid event data (missing fields: {missing_fields})")
        raise ValueError(f"Event missing required fields: {missing_fields}")

    body = {
        "summary": ev["titulo"],
        "start": {"dateTime": ev["inicio"], "timeZone": timezone},
        "end": {"dateTime": ev["fim"], "timeZone": timezone},
    }
    if ev.get("descricao"):
        body["description"] = ev["descricao"]

    logged_title = _mask_title(ev["titulo"]) if mask_titles else ev["titulo"]
    logger.debug(f"Creating event: {logged_title}")

    try:
        def _insert_call():
            return svc.events().insert(calendarId=calendar_id, body=body).execute()

        result = _retry(_insert_call, op_name="events.insert")
        event_id = result.get("id", "unknown")
        logger.info(f"Created event in Google Calendar: {logged_title} (ID: {event_id[:8]}...)")
        return result
    except HttpError as e:
        logger.error(f"Google Calendar API error creating event '{logged_title}': {e}")
        if e.resp.status == 403:
            logger.error("Permission denied. Check if the service account can create events")
        elif e.resp.status == 404:
            logger.error("Calendar not found. Check your GOOGLE_CALENDAR_ID configuration")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating event: {e}")
        raise


def remover_evento_google_by_id(
    svc, calendar_id: str, event_id: str, event_title: str = "", mask_titles: bool = False
) -> bool:
    """
    Remove an event from Google Calendar by ID.

    Args:
        svc: Google Calendar API service.
        calendar_id: target calendar id.
        event_id: Google Calendar event ID.
        event_title: event title, for logging only.
        mask_titles: if True, hash the title before logging it (see module docstring).

    Returns:
        bool: True if deleted (or already gone), False if it failed.
    """
    logged_title = _mask_title(event_title) if mask_titles else event_title

    if not event_id:
        logger.warning(f"Cannot delete event '{logged_title}': missing event ID")
        return False

    logger.debug(f"Deleting event: {logged_title} (ID: {event_id[:8]}...)")

    try:
        def _delete_call():
            return svc.events().delete(calendarId=calendar_id, eventId=event_id).execute()

        _retry(_delete_call, op_name="events.delete")
        logger.info(f"Deleted event from Google Calendar: {logged_title}")
        return True
    except HttpError as e:
        if e.resp.status == 404:
            logger.warning(f"Event '{logged_title}' not found (already deleted?)")
            return True
        if e.resp.status == 403:
            logger.error(f"Permission denied deleting event '{logged_title}'")
        else:
            logger.error(f"Google Calendar API error deleting event '{logged_title}': {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error deleting event '{logged_title}': {e}")
        return False
