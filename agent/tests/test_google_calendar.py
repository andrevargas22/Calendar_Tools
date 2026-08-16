"""
Unit tests for the ported Google Calendar module, with the Google API client
mocked. This is the piece with the highest blast radius (real writes to a
real calendar), so we verify the port didn't silently break anything from
the production-tested original in Calendar_Sync.
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from googleapiclient.errors import HttpError

from common.google_calendar import (
    _retry,
    criar_evento_google,
    get_calendar_service,
    get_google_events,
    remover_evento_google_by_id,
)

VALID_KEY = json.dumps(
    {
        "type": "service_account",
        "project_id": "test-project",
        "private_key": "fake-key",
        "client_email": "agent@test-project.iam.gserviceaccount.com",
    }
)


def _http_error(status: int) -> HttpError:
    resp = MagicMock()
    resp.status = status
    return HttpError(resp, b"error body")


class TestGetCalendarService:
    def test_missing_key_raises_value_error(self):
        with pytest.raises(ValueError):
            get_calendar_service("")

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            get_calendar_service("{not valid json")

    def test_missing_required_fields_raises_value_error(self):
        incomplete = json.dumps({"type": "service_account"})
        with pytest.raises(ValueError):
            get_calendar_service(incomplete)

    @patch("common.google_calendar.build")
    @patch("common.google_calendar.service_account.Credentials.from_service_account_info")
    def test_valid_key_builds_service(self, mock_from_info, mock_build):
        mock_build.return_value = "fake-service"
        result = get_calendar_service(VALID_KEY)
        assert result == "fake-service"
        mock_from_info.assert_called_once()


class TestRetry:
    @patch("common.google_calendar.time.sleep", return_value=None)
    def test_retries_transient_error_then_succeeds(self, _mock_sleep):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise _http_error(503)
            return "ok"

        assert _retry(flaky, op_name="test") == "ok"
        assert calls["n"] == 3

    @patch("common.google_calendar.time.sleep", return_value=None)
    def test_gives_up_after_max_attempts(self, _mock_sleep):
        def always_fails():
            raise _http_error(500)

        with pytest.raises(HttpError):
            _retry(always_fails, max_attempts=3, op_name="test")

    @patch("common.google_calendar.time.sleep", return_value=None)
    def test_non_transient_error_raises_immediately(self, mock_sleep):
        def permission_denied():
            raise _http_error(403)

        with pytest.raises(HttpError):
            _retry(permission_denied, max_attempts=4, op_name="test")
        mock_sleep.assert_not_called()


class TestGetGoogleEvents:
    def test_parses_events_into_naive_datetimes(self):
        svc = MagicMock()
        svc.events.return_value.list.return_value.execute.return_value = {
            "items": [
                {
                    "id": "evt1",
                    "summary": "Reunião",
                    "start": {"dateTime": "2026-08-14T10:00:00-03:00"},
                    "end": {"dateTime": "2026-08-14T11:00:00-03:00"},
                }
            ]
        }

        events = get_google_events(svc, "primary", datetime(2026, 8, 14), datetime(2026, 8, 15))

        assert len(events) == 1
        ev = events[0]
        assert ev["id"] == "evt1"
        assert ev["titulo"] == "Reunião"
        assert ev["inicio"].tzinfo is None
        assert ev["fim"].tzinfo is None

    def test_skips_events_with_missing_start_or_end(self):
        svc = MagicMock()
        svc.events.return_value.list.return_value.execute.return_value = {
            "items": [{"id": "evt1", "summary": "Broken", "start": {}, "end": {}}]
        }

        events = get_google_events(svc, "primary", datetime(2026, 8, 14), datetime(2026, 8, 15))
        assert events == []


class TestCriarEventoGoogle:
    def test_missing_fields_raises_value_error(self):
        svc = MagicMock()
        with pytest.raises(ValueError):
            criar_evento_google(svc, "primary", {"titulo": "Reunião"}, "America/Sao_Paulo")

    def test_builds_correct_body_and_returns_result(self):
        svc = MagicMock()
        svc.events.return_value.insert.return_value.execute.return_value = {"id": "new-evt-id"}

        ev = {
            "titulo": "Reunião",
            "inicio": "2026-08-14T10:00:00",
            "fim": "2026-08-14T11:00:00",
        }
        result = criar_evento_google(svc, "primary", ev, "America/Sao_Paulo")

        assert result["id"] == "new-evt-id"
        _, kwargs = svc.events.return_value.insert.call_args
        assert kwargs["calendarId"] == "primary"
        body = kwargs["body"]
        assert body["summary"] == "Reunião"
        assert body["start"] == {"dateTime": "2026-08-14T10:00:00", "timeZone": "America/Sao_Paulo"}
        assert body["end"] == {"dateTime": "2026-08-14T11:00:00", "timeZone": "America/Sao_Paulo"}


class TestRemoverEventoGoogle:
    def test_missing_event_id_returns_false(self):
        svc = MagicMock()
        assert remover_evento_google_by_id(svc, "primary", "", "Reunião") is False
        svc.events.return_value.delete.assert_not_called()

    def test_successful_delete_returns_true(self):
        svc = MagicMock()
        svc.events.return_value.delete.return_value.execute.return_value = {}

        result = remover_evento_google_by_id(svc, "primary", "evt123", "Reunião")

        assert result is True
        _, kwargs = svc.events.return_value.delete.call_args
        assert kwargs["calendarId"] == "primary"
        assert kwargs["eventId"] == "evt123"

    def test_already_deleted_404_returns_true(self):
        svc = MagicMock()
        svc.events.return_value.delete.return_value.execute.side_effect = _http_error(404)

        result = remover_evento_google_by_id(svc, "primary", "evt123", "Reunião")
        assert result is True

    def test_permission_denied_403_returns_false(self):
        svc = MagicMock()
        svc.events.return_value.delete.return_value.execute.side_effect = _http_error(403)

        result = remover_evento_google_by_id(svc, "primary", "evt123", "Reunião")
        assert result is False
