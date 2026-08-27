"""
Unit tests for CalendarAgent's confirmation state machine (handle_turn /
confirm_pending_action / _resolve_pending). DeepSeek client and the Google
Calendar service are both mocked.
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
    config.get_google_calendar_id_pets.return_value = "pets-calendar"
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


def _seed_pending_update(event=None, changes=None):
    event = event or {"id": "evt1", "titulo": "Dentista", "inicio": "2026-08-20T10:00:00", "fim": "2026-08-20T11:00:00"}
    changes = changes or {"inicio": "2026-08-20T15:00:00", "fim": "2026-08-20T16:00:00"}
    args = {
        "title": event["titulo"],
        "start": "2026-08-20T10:00:00",
        "end": "2026-08-20T11:00:00",
        "new_start": changes.get("inicio"),
        "new_end": changes.get("fim"),
    }
    messages = [
        {"role": "user", "content": "editar evento"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call1", "type": "function", "function": {"name": "update_event", "arguments": json.dumps(args)}}
            ],
        },
        {"role": "tool", "tool_call_id": "call1", "content": _PENDING_PLACEHOLDER},
    ]
    pending = PendingAction(kind="update", prompt="Confirmar edição? [s/N]", message_index=2, event=event, changes=changes)
    return messages, pending


@patch("agent.src.agent.get_calendar_service")
@patch("agent.src.agent.OpenAI")
@patch("agent.src.agent.get_config")
class TestToolDispatchFailureKeepsHistoryValid:
    """Regression: a tool dispatch exception must not leave a tool_calls message unanswered."""

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

        # every tool_calls message must have a matching tool-role reply
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                expected_ids = {tc["id"] for tc in msg["tool_calls"]}
                got_ids = set()
                j = i + 1
                while j < len(messages) and messages[j]["role"] == "tool":
                    got_ids.add(messages[j]["tool_call_id"])
                    j += 1
                assert got_ids == expected_ids, f"message {i} has unanswered tool_calls: {expected_ids - got_ids}"

        # a follow-up turn must still work
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

        # placeholder tool response is appended right after the tool_calls message
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

    def test_update_zero_matches_is_immediate_noop(self, mock_get_config, mock_openai_cls, mock_get_service):
        mock_get_config.return_value = _mock_config(dry_run=False)
        svc = MagicMock()
        svc.events.return_value.list.return_value.execute.return_value = _events_response([])
        mock_get_service.return_value = svc

        args = {
            "title": "Dentista",
            "start": "2026-08-20T10:00:00",
            "end": "2026-08-20T11:00:00",
            "new_start": "2026-08-20T15:00:00",
            "new_end": "2026-08-20T16:00:00",
        }
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _model_response(tool_calls=[_tool_call("call1", "update_event", args)]),
            _model_response(content="Não encontrei o evento."),
        ]
        mock_openai_cls.return_value = client

        agent = CalendarAgent()
        messages = [{"role": "user", "content": "muda horario do dentista"}]
        result = agent.handle_turn(messages)

        assert result.pending is None
        assert "nada foi alterado" in messages[2]["content"]

    def test_update_multiple_matches_is_immediate_refusal(self, mock_get_config, mock_openai_cls, mock_get_service):
        mock_get_config.return_value = _mock_config(dry_run=False)
        svc = MagicMock()
        svc.events.return_value.list.return_value.execute.return_value = _events_response(
            [
                _event("Dentista", "2026-08-20T10:00:00", "2026-08-20T11:00:00", event_id="evt1"),
                _event("Dentista", "2026-08-20T10:00:00", "2026-08-20T11:00:00", event_id="evt2"),
            ]
        )
        mock_get_service.return_value = svc

        args = {
            "title": "Dentista",
            "start": "2026-08-20T10:00:00",
            "end": "2026-08-20T11:00:00",
            "new_start": "2026-08-20T15:00:00",
            "new_end": "2026-08-20T16:00:00",
        }
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _model_response(tool_calls=[_tool_call("call1", "update_event", args)]),
            _model_response(content="Tem mais de um, não vou editar."),
        ]
        mock_openai_cls.return_value = client

        agent = CalendarAgent()
        messages = [{"role": "user", "content": "muda horario do dentista"}]
        result = agent.handle_turn(messages)

        assert result.pending is None
        assert "Encontrei 2 eventos" in messages[2]["content"]

    def test_update_with_no_new_fields_is_immediate_noop(self, mock_get_config, mock_openai_cls, mock_get_service):
        mock_get_config.return_value = _mock_config(dry_run=False)
        svc = MagicMock()
        svc.events.return_value.list.return_value.execute.return_value = _events_response(
            [_event("Dentista", "2026-08-20T10:00:00", "2026-08-20T11:00:00", event_id="evt1")]
        )
        mock_get_service.return_value = svc

        args = {"title": "Dentista", "start": "2026-08-20T10:00:00", "end": "2026-08-20T11:00:00"}
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            _model_response(tool_calls=[_tool_call("call1", "update_event", args)]),
            _model_response(content="Não sei o que mudar."),
        ]
        mock_openai_cls.return_value = client

        agent = CalendarAgent()
        messages = [{"role": "user", "content": "muda o dentista"}]
        result = agent.handle_turn(messages)

        assert result.pending is None
        assert "Nenhum campo novo" in messages[2]["content"]

    def test_update_single_match_with_changes_returns_pending(
        self, mock_get_config, mock_openai_cls, mock_get_service
    ):
        mock_get_config.return_value = _mock_config(dry_run=False)
        svc = MagicMock()
        svc.events.return_value.list.return_value.execute.return_value = _events_response(
            [_event("Dentista", "2026-08-20T10:00:00", "2026-08-20T11:00:00", event_id="evt1")]
        )
        mock_get_service.return_value = svc

        args = {
            "title": "Dentista",
            "start": "2026-08-20T10:00:00",
            "end": "2026-08-20T11:00:00",
            "new_start": "2026-08-20T15:00:00",
            "new_end": "2026-08-20T16:00:00",
        }
        client = MagicMock()
        client.chat.completions.create.return_value = _model_response(
            tool_calls=[_tool_call("call1", "update_event", args)]
        )
        mock_openai_cls.return_value = client

        agent = CalendarAgent()
        result = agent.handle_turn([{"role": "user", "content": "muda horario do dentista"}])

        assert result.pending is not None
        assert result.pending.kind == "update"
        assert result.pending.event["id"] == "evt1"
        assert result.pending.changes == {"inicio": "2026-08-20T15:00:00", "fim": "2026-08-20T16:00:00"}


class TestCalendarRouting:
    @patch("agent.src.agent.get_calendar_service")
    @patch("agent.src.agent.OpenAI")
    @patch("agent.src.agent.get_config")
    def test_defaults_to_pessoal_when_calendar_arg_missing(self, mock_get_config, mock_openai_cls, mock_get_service):
        mock_get_config.return_value = _mock_config(dry_run=False)
        mock_get_service.return_value = MagicMock()
        mock_openai_cls.return_value = MagicMock()

        agent = CalendarAgent()
        assert agent._resolve_calendar_id({}) == "primary"
        assert agent._resolve_calendar_id({"calendar": "unknown"}) == "primary"

    @patch("agent.src.agent.get_calendar_service")
    @patch("agent.src.agent.OpenAI")
    @patch("agent.src.agent.get_config")
    def test_resolves_pets_calendar(self, mock_get_config, mock_openai_cls, mock_get_service):
        mock_get_config.return_value = _mock_config(dry_run=False)
        mock_get_service.return_value = MagicMock()
        mock_openai_cls.return_value = MagicMock()

        agent = CalendarAgent()
        assert agent._resolve_calendar_id({"calendar": "pets"}) == "pets-calendar"

    @patch("agent.src.agent.get_calendar_service")
    @patch("agent.src.agent.OpenAI")
    @patch("agent.src.agent.get_config")
    def test_create_event_uses_requested_calendar(self, mock_get_config, mock_openai_cls, mock_get_service):
        mock_get_config.return_value = _mock_config(dry_run=False)
        svc = MagicMock()
        svc.events.return_value.list.return_value.execute.return_value = _events_response([])
        mock_get_service.return_value = svc

        args = {"title": "Vacina Quico", "start": "2026-08-20T10:00:00", "end": "2026-08-20T11:00:00", "calendar": "pets"}
        client = MagicMock()
        client.chat.completions.create.return_value = _model_response(
            tool_calls=[_tool_call("call1", "create_event", args)]
        )
        mock_openai_cls.return_value = client

        agent = CalendarAgent()
        result = agent.handle_turn([{"role": "user", "content": "vacina do Quico"}])

        assert result.pending.calendar_id == "pets-calendar"
        svc.events.return_value.list.assert_called_with(
            calendarId="pets-calendar",
            timeMin="2026-08-20T10:00:00Z",
            timeMax="2026-08-21T11:00:00Z",
            singleEvents=True,
            orderBy="startTime",
            maxResults=2500,
        )


@patch("agent.src.agent.atualizar_evento_google")
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
        self, mock_get_config, mock_openai_cls, mock_get_service, mock_criar, mock_remover, mock_atualizar
    ):
        agent, client = self._agent(mock_get_config, mock_openai_cls, mock_get_service, dry_run=True)
        messages, pending = _seed_pending_create()

        result = agent.confirm_pending_action(messages, pending, confirmed=True)

        mock_criar.assert_not_called()
        assert "[DRY RUN]" in messages[pending.message_index]["content"]
        assert result.pending is None
        assert client.chat.completions.create.call_count == 1

    def test_confirm_create_writes_when_confirmed_and_not_dry_run(
        self, mock_get_config, mock_openai_cls, mock_get_service, mock_criar, mock_remover, mock_atualizar
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
        self, mock_get_config, mock_openai_cls, mock_get_service, mock_criar, mock_remover, mock_atualizar
    ):
        agent, _ = self._agent(mock_get_config, mock_openai_cls, mock_get_service, dry_run=False)
        messages, pending = _seed_pending_create()

        agent.confirm_pending_action(messages, pending, confirmed=False)

        mock_criar.assert_not_called()
        assert "cancelou a criação" in messages[pending.message_index]["content"]

    def test_confirm_delete_writes_correct_event_id(
        self, mock_get_config, mock_openai_cls, mock_get_service, mock_criar, mock_remover, mock_atualizar
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
        self, mock_get_config, mock_openai_cls, mock_get_service, mock_criar, mock_remover, mock_atualizar
    ):
        agent, _ = self._agent(mock_get_config, mock_openai_cls, mock_get_service, dry_run=True)
        messages, pending = _seed_pending_delete()

        agent.confirm_pending_action(messages, pending, confirmed=True)

        mock_remover.assert_not_called()
        assert "[DRY RUN]" in messages[pending.message_index]["content"]

    def test_confirm_update_writes_correct_changes(
        self, mock_get_config, mock_openai_cls, mock_get_service, mock_criar, mock_remover, mock_atualizar
    ):
        agent, _ = self._agent(mock_get_config, mock_openai_cls, mock_get_service, dry_run=False)
        messages, pending = _seed_pending_update()

        agent.confirm_pending_action(messages, pending, confirmed=True)

        mock_atualizar.assert_called_once()
        call_args = mock_atualizar.call_args[0]
        assert call_args[2] == "evt1"
        assert call_args[3] == {"inicio": "2026-08-20T15:00:00", "fim": "2026-08-20T16:00:00"}
        assert "atualizado com sucesso" in messages[pending.message_index]["content"]

    def test_confirm_update_dry_run_does_not_write(
        self, mock_get_config, mock_openai_cls, mock_get_service, mock_criar, mock_remover, mock_atualizar
    ):
        agent, _ = self._agent(mock_get_config, mock_openai_cls, mock_get_service, dry_run=True)
        messages, pending = _seed_pending_update()

        agent.confirm_pending_action(messages, pending, confirmed=True)

        mock_atualizar.assert_not_called()
        assert "[DRY RUN]" in messages[pending.message_index]["content"]

    def test_confirm_update_cancelled_does_not_write(
        self, mock_get_config, mock_openai_cls, mock_get_service, mock_criar, mock_remover, mock_atualizar
    ):
        agent, _ = self._agent(mock_get_config, mock_openai_cls, mock_get_service, dry_run=False)
        messages, pending = _seed_pending_update()

        agent.confirm_pending_action(messages, pending, confirmed=False)

        mock_atualizar.assert_not_called()
        assert "cancelou a edição" in messages[pending.message_index]["content"]
