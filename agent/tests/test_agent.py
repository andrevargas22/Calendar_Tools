"""
Unit tests for CalendarAgent's confirmation state machine — the pending
create_event/delete_event proposal that a UI-agnostic caller (REPL, Telegram
bot) resolves via confirm_pending_action() instead of the agent blocking on
input() itself.

The DeepSeek client and the Google Calendar service are both mocked; only
the state machine (handle_turn / confirm_pending_action / _resolve_pending)
is under test.
"""

import json
from unittest.mock import MagicMock, patch

from agent.src.agent import _PENDING_PLACEHOLDER, CalendarAgent, PendingAction


def _mock_config(dry_run=False):
    config = MagicMock()
    config.get_deepseek_api_key.return_value = "test-key"
    config.get_deepseek_base_url.return_value = "https://api.deepseek.com"
    config.get_model.return_value = "deepseek-chat"
    config.get_max_tokens.return_value = 1024
    config.get_timezone.return_value = "America/Sao_Paulo"
    config.get_google_calendar_id.return_value = "primary"
    config.load_system_prompt.return_value = "system prompt"
    config.setup_mlflow.return_value = False
    config.is_dry_run.return_value = dry_run
    config.get_temperature.return_value = 0
    return config


def _model_response(content=None, tool_calls=None, input_tokens=10, output_tokens=5):
    response = MagicMock()
    response.usage.prompt_tokens = input_tokens
    response.usage.completion_tokens = output_tokens
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    response.choices = [MagicMock(message=message)]
    return response


def _tool_call(call_id, name, arguments):
    tc = MagicMock()
    tc.id = call_id
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


def _events_response(items):
    return {"items": items}


def _event(summary, start_iso, end_iso, event_id="evt1"):
    return {"id": event_id, "summary": summary, "start": {"dateTime": start_iso}, "end": {"dateTime": end_iso}}


def _seed_pending_create(args=None):
    args = args or {"title": "Dentista", "start": "2026-08-20T10:00:00", "end": "2026-08-20T11:00:00"}
    messages = [
        {"role": "user", "content": "criar evento"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call1", "type": "function", "function": {"name": "create_event", "arguments": json.dumps(args)}}
            ],
        },
        {"role": "tool", "tool_call_id": "call1", "content": _PENDING_PLACEHOLDER},
    ]
    pending = PendingAction(kind="create", prompt="Confirmar criação? [s/N]", message_index=2, args=args)
    return messages, pending


def _seed_pending_delete(event=None):
    event = event or {"id": "evt1", "titulo": "Dentista", "inicio": "2026-08-20T10:00:00", "fim": "2026-08-20T11:00:00"}
    args = {"title": event["titulo"], "start": "2026-08-20T10:00:00", "end": "2026-08-20T11:00:00"}
    messages = [
        {"role": "user", "content": "remover evento"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call1", "type": "function", "function": {"name": "delete_event", "arguments": json.dumps(args)}}
            ],
        },
        {"role": "tool", "tool_call_id": "call1", "content": _PENDING_PLACEHOLDER},
    ]
    pending = PendingAction(kind="delete", prompt="Confirmar remoção? [s/N]", message_index=2, event=event)
    return messages, pending


@patch("agent.src.agent.get_calendar_service")
@patch("agent.src.agent.OpenAI")
@patch("agent.src.agent.get_config")
class TestToolDispatchFailureKeepsHistoryValid:
    """
    Regression test for a real production incident: a transient Google
    Calendar SSL error raised out of _handle_create_event, which propagated
    past the tool-call loop and left the assistant's tool_calls message with
    no matching tool-role reply. Because `messages` is persisted for the
    whole chat session (see telegram_bot.py), every subsequent turn in that
    chat then failed outright with DeepSeek's 400 "insufficient tool
    messages following tool_calls message" — the conversation was
    permanently wedged until the process restarted.
    """

    def test_dispatch_exception_becomes_tool_result_not_a_dangling_tool_call(
        self, mock_get_config, mock_openai_cls, mock_get_service
    ):
        mock_get_config.return_value = _mock_config(dry_run=False)
        svc = MagicMock()
        svc.events.return_value.list.return_value.execute.side_effect = OSError(
            "[SSL: UNEXPECTED_EOF_WHILE_READING] unexpected eof while reading"
        )
        mock_get_service.return_value = svc

        args = {"title": "Dentista", "start": "2026-08-20T10:00:00", "end": "2026-08-20T11:00:00"}
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _model_response(tool_calls=[_tool_call("call1", "create_event", args)]),
            _model_response(content="Deu um erro ao consultar sua agenda, tente de novo em instantes."),
        ]
        mock_openai_cls.return_value = client

        agent = CalendarAgent()
        messages = [{"role": "user", "content": "cria dentista amanha"}]
        result = agent.handle_turn(messages)

        assert result.pending is None
        assert client.chat.completions.create.call_count == 2

        # every tool_calls message must be immediately followed by a matching
        # tool-role reply, or the next model call in this same chat session
        # (this `messages` list is reused turn after turn) is rejected outright
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                expected_ids = {tc["id"] for tc in msg["tool_calls"]}
                got_ids = set()
                j = i + 1
                while j < len(messages) and messages[j]["role"] == "tool":
                    got_ids.add(messages[j]["tool_call_id"])
                    j += 1
                assert got_ids == expected_ids, f"message {i} has unanswered tool_calls: {expected_ids - got_ids}"

        # a follow-up turn in the same session must still be able to call the
        # model at all (i.e. nothing about the corrupted history blocks it)
        messages.append({"role": "user", "content": "tenta de novo"})
        client.chat.completions.create.side_effect = [_model_response(content="Ok, tentando de novo.")]
        follow_up = agent.handle_turn(messages)
        assert follow_up.text == "Ok, tentando de novo."


@patch("agent.src.agent.get_calendar_service")
@patch("agent.src.agent.OpenAI")
@patch("agent.src.agent.get_config")
class TestHandleTurnCreatesPending:
    def test_create_with_no_existing_match_returns_pending(self, mock_get_config, mock_openai_cls, mock_get_service):
        mock_get_config.return_value = _mock_config(dry_run=False)
        svc = MagicMock()
        svc.events.return_value.list.return_value.execute.return_value = _events_response([])
        mock_get_service.return_value = svc

        args = {"title": "Dentista", "start": "2026-08-20T10:00:00", "end": "2026-08-20T11:00:00"}
        client = MagicMock()
        client.chat.completions.create.return_value = _model_response(
            tool_calls=[_tool_call("call1", "create_event", args)]
        )
        mock_openai_cls.return_value = client

        agent = CalendarAgent()
        messages = [{"role": "user", "content": "cria dentista amanha"}]
        result = agent.handle_turn(messages)

        assert result.pending is not None
        assert result.pending.kind == "create"
        assert result.pending.args == args
        assert "Confirmar criação" in result.text
        assert client.chat.completions.create.call_count == 1

        # history stays API-valid: the tool_calls message is immediately
        # followed by a matching tool response (the placeholder).
        assert messages[-2]["tool_calls"][0]["id"] == "call1"
        assert messages[-1] == {"role": "tool", "tool_call_id": "call1", "content": _PENDING_PLACEHOLDER}
        assert result.pending.message_index == len(messages) - 1

    def test_create_with_existing_match_is_immediate_noop(self, mock_get_config, mock_openai_cls, mock_get_service):
        mock_get_config.return_value = _mock_config(dry_run=False)
        svc = MagicMock()
        svc.events.return_value.list.return_value.execute.return_value = _events_response(
            [_event("Dentista", "2026-08-20T10:00:00", "2026-08-20T11:00:00")]
        )
        mock_get_service.return_value = svc

        args = {"title": "Dentista", "start": "2026-08-20T10:00:00", "end": "2026-08-20T11:00:00"}
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _model_response(tool_calls=[_tool_call("call1", "create_event", args)]),
            _model_response(content="Já existe, não criei de novo."),
        ]
        mock_openai_cls.return_value = client

        agent = CalendarAgent()
        result = agent.handle_turn([{"role": "user", "content": "cria dentista amanha"}])

        assert result.pending is None
        assert client.chat.completions.create.call_count == 2

    def test_delete_zero_matches_is_immediate_noop(self, mock_get_config, mock_openai_cls, mock_get_service):
        mock_get_config.return_value = _mock_config(dry_run=False)
        svc = MagicMock()
        svc.events.return_value.list.return_value.execute.return_value = _events_response([])
        mock_get_service.return_value = svc

        args = {"title": "Dentista", "start": "2026-08-20T10:00:00", "end": "2026-08-20T11:00:00"}
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _model_response(tool_calls=[_tool_call("call1", "delete_event", args)]),
            _model_response(content="Não encontrei nada pra remover."),
        ]
        mock_openai_cls.return_value = client

        agent = CalendarAgent()
        messages = [{"role": "user", "content": "remove dentista"}]
        result = agent.handle_turn(messages)

        assert result.pending is None
        assert "nada foi removido" in messages[2]["content"]

    def test_delete_multiple_matches_is_immediate_refusal(self, mock_get_config, mock_openai_cls, mock_get_service):
        mock_get_config.return_value = _mock_config(dry_run=False)
        svc = MagicMock()
        svc.events.return_value.list.return_value.execute.return_value = _events_response(
            [
                _event("Dentista", "2026-08-20T10:00:00", "2026-08-20T11:00:00", event_id="evt1"),
                _event("Dentista", "2026-08-20T10:00:00", "2026-08-20T11:00:00", event_id="evt2"),
            ]
        )
        mock_get_service.return_value = svc

        args = {"title": "Dentista", "start": "2026-08-20T10:00:00", "end": "2026-08-20T11:00:00"}
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _model_response(tool_calls=[_tool_call("call1", "delete_event", args)]),
            _model_response(content="Tem mais de um, não vou remover."),
        ]
        mock_openai_cls.return_value = client

        agent = CalendarAgent()
        messages = [{"role": "user", "content": "remove dentista"}]
        result = agent.handle_turn(messages)

        assert result.pending is None
        assert "Encontrei 2 eventos" in messages[2]["content"]

    def test_delete_single_match_returns_pending_with_event(self, mock_get_config, mock_openai_cls, mock_get_service):
        mock_get_config.return_value = _mock_config(dry_run=False)
        svc = MagicMock()
        svc.events.return_value.list.return_value.execute.return_value = _events_response(
            [_event("Dentista", "2026-08-20T10:00:00", "2026-08-20T11:00:00", event_id="evt1")]
        )
        mock_get_service.return_value = svc

        args = {"title": "Dentista", "start": "2026-08-20T10:00:00", "end": "2026-08-20T11:00:00"}
        client = MagicMock()
        client.chat.completions.create.return_value = _model_response(
            tool_calls=[_tool_call("call1", "delete_event", args)]
        )
        mock_openai_cls.return_value = client

        agent = CalendarAgent()
        result = agent.handle_turn([{"role": "user", "content": "remove dentista"}])

        assert result.pending is not None
        assert result.pending.kind == "delete"
        assert result.pending.event["id"] == "evt1"


@patch("agent.src.agent.remover_evento_google_by_id")
@patch("agent.src.agent.criar_evento_google")
@patch("agent.src.agent.get_calendar_service")
@patch("agent.src.agent.OpenAI")
@patch("agent.src.agent.get_config")
class TestConfirmPendingAction:
    def _agent(self, mock_get_config, mock_openai_cls, mock_get_service, dry_run, final_content="Ok."):
        mock_get_config.return_value = _mock_config(dry_run=dry_run)
        mock_get_service.return_value = MagicMock()
        client = MagicMock()
        client.chat.completions.create.return_value = _model_response(content=final_content)
        mock_openai_cls.return_value = client
        return CalendarAgent(), client

    def test_confirm_create_dry_run_does_not_write(
        self, mock_get_config, mock_openai_cls, mock_get_service, mock_criar, mock_remover
    ):
        agent, client = self._agent(mock_get_config, mock_openai_cls, mock_get_service, dry_run=True)
        messages, pending = _seed_pending_create()

        result = agent.confirm_pending_action(messages, pending, confirmed=True)

        mock_criar.assert_not_called()
        assert "[DRY RUN]" in messages[pending.message_index]["content"]
        assert result.pending is None
        assert client.chat.completions.create.call_count == 1

    def test_confirm_create_writes_when_confirmed_and_not_dry_run(
        self, mock_get_config, mock_openai_cls, mock_get_service, mock_criar, mock_remover
    ):
        mock_criar.return_value = {"id": "evt-999"}
        agent, _ = self._agent(mock_get_config, mock_openai_cls, mock_get_service, dry_run=False)
        messages, pending = _seed_pending_create()

        result = agent.confirm_pending_action(messages, pending, confirmed=True)

        mock_criar.assert_called_once()
        _, kwargs = mock_criar.call_args
        assert result.pending is None
        assert "criado com sucesso" in messages[pending.message_index]["content"]

    def test_confirm_create_cancelled_does_not_write(
        self, mock_get_config, mock_openai_cls, mock_get_service, mock_criar, mock_remover
    ):
        agent, _ = self._agent(mock_get_config, mock_openai_cls, mock_get_service, dry_run=False)
        messages, pending = _seed_pending_create()

        agent.confirm_pending_action(messages, pending, confirmed=False)

        mock_criar.assert_not_called()
        assert "cancelou a criação" in messages[pending.message_index]["content"]

    def test_confirm_delete_writes_correct_event_id(
        self, mock_get_config, mock_openai_cls, mock_get_service, mock_criar, mock_remover
    ):
        mock_remover.return_value = True
        agent, _ = self._agent(mock_get_config, mock_openai_cls, mock_get_service, dry_run=False)
        messages, pending = _seed_pending_delete()

        agent.confirm_pending_action(messages, pending, confirmed=True)

        mock_remover.assert_called_once()
        args, _ = mock_remover.call_args
        assert args[2] == "evt1"
        assert "removido com sucesso" in messages[pending.message_index]["content"]

    def test_confirm_delete_dry_run_does_not_write(
        self, mock_get_config, mock_openai_cls, mock_get_service, mock_criar, mock_remover
    ):
        agent, _ = self._agent(mock_get_config, mock_openai_cls, mock_get_service, dry_run=True)
        messages, pending = _seed_pending_delete()

        agent.confirm_pending_action(messages, pending, confirmed=True)

        mock_remover.assert_not_called()
        assert "[DRY RUN]" in messages[pending.message_index]["content"]
