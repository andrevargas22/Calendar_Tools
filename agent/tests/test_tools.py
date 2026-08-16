"""
Unit tests for find_matching_events, the shared lookup used by both
create_event's idempotency check and delete_event's guardrail — an event is
a "match" only if title (normalized), start and end are all exactly equal.
"""

from datetime import datetime
from unittest.mock import MagicMock

from agent.src.tools import find_matching_events

START = datetime(2026, 8, 17, 8, 0)
END = datetime(2026, 8, 17, 9, 0)


def _svc_returning(items):
    svc = MagicMock()
    svc.events.return_value.list.return_value.execute.return_value = {"items": items}
    return svc


def _event(summary, start_iso, end_iso, event_id="evt1"):
    return {
        "id": event_id,
        "summary": summary,
        "start": {"dateTime": start_iso},
        "end": {"dateTime": end_iso},
    }


class TestFindMatchingEvents:
    def test_no_events_returns_empty(self):
        svc = _svc_returning([])
        assert find_matching_events(svc, "primary", "Vistoria", START, END) == []

    def test_exact_title_and_time_match(self):
        svc = _svc_returning([_event("Vistoria", "2026-08-17T08:00:00", "2026-08-17T09:00:00")])
        matches = find_matching_events(svc, "primary", "Vistoria", START, END)
        assert len(matches) == 1
        assert matches[0]["titulo"] == "Vistoria"

    def test_title_match_is_case_and_whitespace_insensitive(self):
        svc = _svc_returning([_event("  VISTORIA  ", "2026-08-17T08:00:00", "2026-08-17T09:00:00")])
        matches = find_matching_events(svc, "primary", "vistoria", START, END)
        assert len(matches) == 1

    def test_different_title_does_not_match(self):
        svc = _svc_returning([_event("Dentista", "2026-08-17T08:00:00", "2026-08-17T09:00:00")])
        assert find_matching_events(svc, "primary", "Vistoria", START, END) == []

    def test_different_time_does_not_match(self):
        svc = _svc_returning([_event("Vistoria", "2026-08-17T10:00:00", "2026-08-17T11:00:00")])
        assert find_matching_events(svc, "primary", "Vistoria", START, END) == []

    def test_multiple_identical_events_all_returned(self):
        svc = _svc_returning(
            [
                _event("Vistoria", "2026-08-17T08:00:00", "2026-08-17T09:00:00", event_id="evt1"),
                _event("Vistoria", "2026-08-17T08:00:00", "2026-08-17T09:00:00", event_id="evt2"),
            ]
        )
        matches = find_matching_events(svc, "primary", "Vistoria", START, END)
        assert len(matches) == 2
