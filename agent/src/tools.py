"""
Tool schemas (OpenAI-compatible function-calling format, used by DeepSeek)
and the execution function for the read-only tool.

create_event and delete_event have no execution function here on purpose:
deciding whether to actually write to the calendar involves a human
confirmation step (and, for create, an idempotency check), both of which are
about the conversation/session, not about the tool call itself — that
orchestration lives in src/agent.py.
"""

from datetime import datetime

from common.google_calendar import get_google_events

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_events",
            "description": "Busca eventos no Google Calendar dentro de um intervalo de datas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "string",
                        "description": "Início do intervalo em ISO 8601, ex: 2026-08-14T00:00:00",
                    },
                    "end": {
                        "type": "string",
                        "description": "Fim do intervalo em ISO 8601, ex: 2026-08-14T23:59:59",
                    },
                },
                "required": ["start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_event",
            "description": (
                "Propõe a criação de um evento no Google Calendar. Não escreve na agenda "
                "imediatamente — a criação real depende de confirmação explícita do "
                "usuário, tratada fora do modelo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Título do evento"},
                    "start": {"type": "string", "description": "Início em ISO 8601"},
                    "end": {"type": "string", "description": "Fim em ISO 8601"},
                    "description": {"type": "string", "description": "Descrição opcional do evento"},
                },
                "required": ["title", "start", "end"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_event",
            "description": (
                "Propõe a remoção de um evento existente da agenda. Não apaga "
                "imediatamente — a remoção real depende de confirmação explícita do "
                "usuário, tratada fora do modelo. Use título, início e fim exatos "
                "(de uma busca recente com search_events, se necessário) para "
                "identificar o evento certo."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Título exato do evento a remover"},
                    "start": {"type": "string", "description": "Início exato em ISO 8601"},
                    "end": {"type": "string", "description": "Fim exato em ISO 8601"},
                },
                "required": ["title", "start", "end"],
            },
        },
    },
]


def find_matching_events(svc, calendar_id: str, title: str, start, end) -> list[dict]:
    """
    Look up events in [start, end] whose normalized title and exact
    start/end match the given values. Shared by create_event's idempotency
    check and delete_event's lookup — both need "does an event exactly like
    this already exist" against the calendar itself, not a local store.
    """
    normalized_title = title.strip().lower()
    existing = get_google_events(svc, calendar_id, start, end)
    return [
        ev
        for ev in existing
        if ev["titulo"].strip().lower() == normalized_title and ev["inicio"] == start and ev["fim"] == end
    ]


def run_search_events(svc, calendar_id: str, args: dict) -> str:
    """Execute search_events and format the result as a tool_result string."""
    start = datetime.fromisoformat(args["start"])
    end = datetime.fromisoformat(args["end"])

    events = get_google_events(svc, calendar_id, start, end)

    if not events:
        return "Nenhum evento encontrado nesse intervalo."

    lines = [
        f"- {ev['titulo']}: {ev['inicio'].isoformat()} até {ev['fim'].isoformat()}"
        for ev in events
    ]
    return "Eventos encontrados:\n" + "\n".join(lines)
