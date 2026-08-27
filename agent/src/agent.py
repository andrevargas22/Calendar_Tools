"""
Native tool-calling agent loop for Calendar Agent, on top of DeepSeek's
OpenAI-compatible chat completions API. No agent framework — a plain loop:
call the model, dispatch any tool calls, repeat until it replies in plain text.
Each turn is traced to MLflow if reachable; tracing failures never break the turn.
"""

import json
import logging
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
from zoneinfo import ZoneInfo

from openai import OpenAI

from agent.src.config import get_config
from common.google_calendar import (
    atualizar_evento_google,
    criar_evento_google,
    get_calendar_service,
    remover_evento_google_by_id,
)
from agent.src.tools import TOOL_SCHEMAS, find_matching_events, run_search_events

logger = logging.getLogger(__name__)

CONFIRM_YES = {"s", "sim", "y", "yes"}

_PENDING_PLACEHOLDER = "[Aguardando confirmação do usuário.]"


@dataclass
class PendingAction:
    """A create/update/delete proposal awaiting human confirmation."""

    kind: str  # "create" | "update" | "delete"
    prompt: str
    message_index: int = -1
    calendar_id: Optional[str] = None
    args: Optional[dict] = None  # kind == "create"
    event: Optional[dict] = None  # kind == "update" | "delete"
    changes: Optional[dict] = None  # kind == "update"


@dataclass
class TurnResult:
    text: str
    pending: Optional[PendingAction] = None


class CalendarAgent:
    def __init__(self):
        self.config = get_config()
        self.client = OpenAI(
            api_key=self.config.get_deepseek_api_key(),
            base_url=self.config.get_deepseek_base_url(),
        )
        self.model = self.config.get_model()
        self.max_tokens = self.config.get_max_tokens()
        self.tz = ZoneInfo(self.config.get_timezone())
        self.calendars = {
            "pessoal": self.config.get_google_calendar_id(),
            "pets": self.config.get_google_calendar_id_pets(),
        }

        self.system_prompt = self.config.load_system_prompt()
        self._mlflow_ready = self.config.setup_mlflow()
        self._svc = None

    def _get_service(self):
        if self._svc is None:
            self._svc = get_calendar_service(self.config.get_google_service_account_key())
        return self._svc

    def _resolve_calendar_id(self, args: dict) -> str:
        key = args.get("calendar") or "pessoal"
        return self.calendars.get(key, self.calendars["pessoal"])

    def _build_system(self) -> str:
        now = datetime.now(self.tz)
        context = (
            f"\n\nContexto atual: agora é {now.strftime('%A, %d de %B de %Y, %H:%M')} "
            f"({self.tz.key}). Use isso para resolver datas e horários relativos "
            "mencionados pelo usuário (ex: 'amanhã', 'essa semana', 'sexta-feira')."
        )
        return self.system_prompt + context

    def handle_turn(self, messages: list) -> TurnResult:
        """Run one turn, mutating `messages` in place. If the result's `pending` is
        set, resolve it via confirm_pending_action() before the next call."""
        user_input = messages[-1]["content"] if messages and messages[-1]["role"] == "user" else ""
        trace = {"user_input": user_input, "steps": []}
        return self._run_turn(messages, trace)

    def confirm_pending_action(self, messages: list, pending: PendingAction, confirmed: bool) -> TurnResult:
        """Execute (or cancel) a PendingAction, then resume the model loop for a natural-language reply."""
        result_text = self._resolve_pending(pending, confirmed)
        messages[pending.message_index]["content"] = result_text

        trace = {
            "user_input": f"[confirmação: {'sim' if confirmed else 'não'}]",
            "steps": [{"tool": f"{pending.kind}_event (confirmação)", "input": None, "result": result_text}],
        }
        return self._run_turn(messages, trace)

    def _run_turn(self, messages: list, trace: dict) -> TurnResult:
        """Model-call/tool-dispatch loop shared by handle_turn() and confirm_pending_action()."""
        system = self._build_system()
        start_time = time.perf_counter()
        total_input_tokens = 0
        total_output_tokens = 0
        num_model_calls = 0
        num_tool_calls = 0
        tools_called: list[str] = []
        status = "success"
        final_text = ""
        pending: Optional[PendingAction] = None

        try:
            while True:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.config.get_temperature(),
                    messages=[{"role": "system", "content": system}] + messages,
                    tools=TOOL_SCHEMAS,
                )
                num_model_calls += 1
                if response.usage:
                    total_input_tokens += response.usage.prompt_tokens
                    total_output_tokens += response.usage.completion_tokens

                choice = response.choices[0]
                message = choice.message

                assistant_msg = {"role": "assistant", "content": message.content or ""}
                if message.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in message.tool_calls
                    ]
                messages.append(assistant_msg)

                if not message.tool_calls:
                    final_text = message.content or ""
                    break

                for i, tc in enumerate(message.tool_calls):
                    num_tool_calls += 1
                    tools_called.append(tc.function.name)

                    try:
                        args = json.loads(tc.function.arguments or "{}")
                        result = self._dispatch_tool(tc.function.name, args)
                    except Exception as e:
                        # Every tool_calls message needs a matching tool reply, or the next
                        # model call in this session fails outright — never let this propagate.
                        logger.exception(f"Tool '{tc.function.name}' failed")
                        args = {}
                        result = f"Erro ao executar a ferramenta '{tc.function.name}': {e}"

                    if isinstance(result, PendingAction):
                        result.message_index = len(messages)
                        messages.append({"role": "tool", "tool_call_id": tc.id, "content": _PENDING_PLACEHOLDER})
                        trace["steps"].append(
                            {"tool": tc.function.name, "input": args, "result": "pending_confirmation"}
                        )

                        # remaining tool_calls in this response still need a matching reply
                        for rtc in message.tool_calls[i + 1 :]:
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": rtc.id,
                                    "content": "[Ação cancelada: havia uma confirmação pendente antes desta chamada.]",
                                }
                            )

                        final_text = result.prompt
                        pending = result
                        break

                    trace["steps"].append({"tool": tc.function.name, "input": args, "result": result})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

                if pending is not None:
                    break

        except Exception as e:
            status = "error"
            final_text = f"Ocorreu um erro ao processar seu pedido: {e}"
            pending = None
            logger.exception("Agent turn failed")

        if pending is not None:
            status = "pending_confirmation"

        trace["final_response"] = final_text
        latency_s = time.perf_counter() - start_time

        self._log_run(
            status=status,
            tools_called=tools_called,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            num_model_calls=num_model_calls,
            num_tool_calls=num_tool_calls,
            latency_s=latency_s,
            trace=trace,
        )
        return TurnResult(text=final_text, pending=pending)

    def _dispatch_tool(self, name: str, args: dict) -> Union[str, PendingAction]:
        if name == "search_events":
            return run_search_events(self._get_service(), self._resolve_calendar_id(args), args)
        if name == "create_event":
            return self._handle_create_event(args)
        if name == "delete_event":
            return self._handle_delete_event(args)
        if name == "update_event":
            return self._handle_update_event(args)
        return f"Ferramenta desconhecida: {name}"

    def _handle_create_event(self, args: dict) -> Union[str, PendingAction]:
        """Idempotency check, then return a PendingAction for the caller to confirm (dry_run applies later, in _resolve_pending)."""
        title = args["title"]
        start = datetime.fromisoformat(args["start"])
        end = datetime.fromisoformat(args["end"])
        description = args.get("description")
        calendar_id = self._resolve_calendar_id(args)

        matches = find_matching_events(self._get_service(), calendar_id, title, start, end)
        if matches:
            return (
                f"Já existe um evento idêntico ('{matches[0]['titulo']}') nesse horário — "
                "nada foi criado (idempotência)."
            )

        prompt_lines = [
            "O agente quer criar o seguinte evento:",
            f"  Título: {title}",
            f"  Início: {start.isoformat()}",
            f"  Fim:    {end.isoformat()}",
        ]
        if description:
            prompt_lines.append(f"  Descrição: {description}")
        prompt_lines.append("Confirmar criação? [s/N]")

        return PendingAction(kind="create", prompt="\n".join(prompt_lines), args=args, calendar_id=calendar_id)

    def _handle_delete_event(self, args: dict) -> Union[str, PendingAction]:
        """Look up the exact event by title+time; zero/multiple matches are an immediate no-op, one match returns a PendingAction."""
        title = args["title"]
        start = datetime.fromisoformat(args["start"])
        end = datetime.fromisoformat(args["end"])
        calendar_id = self._resolve_calendar_id(args)

        matches = find_matching_events(self._get_service(), calendar_id, title, start, end)
        if not matches:
            return "Não encontrei nenhum evento com esse título e horário exatos — nada foi removido."
        if len(matches) > 1:
            return (
                f"Encontrei {len(matches)} eventos idênticos ('{title}') nesse horário — "
                "não vou remover nenhum sem saber qual exatamente. Verifique a agenda diretamente."
            )

        event = matches[0]
        prompt = (
            "O agente quer remover o seguinte evento:\n"
            f"  Título: {event['titulo']}\n"
            f"  Início: {event['inicio'].isoformat()}\n"
            f"  Fim:    {event['fim'].isoformat()}\n"
            "Confirmar remoção? [s/N]"
        )
        return PendingAction(kind="delete", prompt=prompt, event=event, calendar_id=calendar_id)

    def _handle_update_event(self, args: dict) -> Union[str, PendingAction]:
        """Look up the exact event by title+time, then propose the given new_* fields as changes."""
        title = args["title"]
        start = datetime.fromisoformat(args["start"])
        end = datetime.fromisoformat(args["end"])
        calendar_id = self._resolve_calendar_id(args)

        matches = find_matching_events(self._get_service(), calendar_id, title, start, end)
        if not matches:
            return "Não encontrei nenhum evento com esse título e horário exatos — nada foi alterado."
        if len(matches) > 1:
            return (
                f"Encontrei {len(matches)} eventos idênticos ('{title}') nesse horário — "
                "não vou editar nenhum sem saber qual exatamente. Verifique a agenda diretamente."
            )

        changes = {}
        if args.get("new_title"):
            changes["titulo"] = args["new_title"]
        if args.get("new_start"):
            changes["inicio"] = args["new_start"]
        if args.get("new_end"):
            changes["fim"] = args["new_end"]
        if args.get("new_description"):
            changes["descricao"] = args["new_description"]

        if not changes:
            return "Nenhum campo novo foi informado para alterar — nada foi editado."

        event = matches[0]
        prompt_lines = [
            "O agente quer editar o seguinte evento:",
            f"  Título atual: {event['titulo']}",
            f"  Início atual: {event['inicio'].isoformat()}",
            f"  Fim atual:    {event['fim'].isoformat()}",
            "Para:",
        ]
        if "titulo" in changes:
            prompt_lines.append(f"  Novo título: {changes['titulo']}")
        if "inicio" in changes:
            prompt_lines.append(f"  Novo início: {changes['inicio']}")
        if "fim" in changes:
            prompt_lines.append(f"  Novo fim:    {changes['fim']}")
        if "descricao" in changes:
            prompt_lines.append(f"  Nova descrição: {changes['descricao']}")
        prompt_lines.append("Confirmar edição? [s/N]")

        return PendingAction(
            kind="update", prompt="\n".join(prompt_lines), event=event, changes=changes, calendar_id=calendar_id
        )

    def _resolve_pending(self, pending: PendingAction, confirmed: bool) -> str:
        if not confirmed:
            cancel_texts = {
                "create": "O usuário cancelou a criação do evento.",
                "update": "O usuário cancelou a edição do evento.",
                "delete": "O usuário cancelou a remoção do evento.",
            }
            return cancel_texts[pending.kind]

        calendar_id = pending.calendar_id or self.calendars["pessoal"]

        if pending.kind == "create":
            title = pending.args["title"]
            if self.config.is_dry_run():
                return f"[DRY RUN] Evento '{title}' seria criado, mas dry_run está ativado — nada foi escrito na agenda."
            result = criar_evento_google(
                self._get_service(),
                calendar_id,
                {
                    "titulo": title,
                    "inicio": pending.args["start"],
                    "fim": pending.args["end"],
                    "descricao": pending.args.get("description"),
                },
                self.config.get_timezone(),
            )
            return f"Evento '{title}' criado com sucesso (ID: {result.get('id', 'unknown')})."

        if pending.kind == "update":
            title = pending.event["titulo"]
            if self.config.is_dry_run():
                return f"[DRY RUN] Evento '{title}' seria editado, mas dry_run está ativado — nada foi alterado na agenda."
            atualizar_evento_google(
                self._get_service(), calendar_id, pending.event["id"], pending.changes, self.config.get_timezone()
            )
            return f"Evento '{title}' atualizado com sucesso."

        title = pending.event["titulo"]
        if self.config.is_dry_run():
            return f"[DRY RUN] Evento '{title}' seria removido, mas dry_run está ativado — nada foi apagado da agenda."
        success = remover_evento_google_by_id(self._get_service(), calendar_id, pending.event["id"], title)
        if success:
            return f"Evento '{title}' removido com sucesso."
        return f"Falha ao remover o evento '{title}' — veja os logs para detalhes."

    def _log_run(self, *, status, tools_called, input_tokens, output_tokens, num_model_calls, num_tool_calls, latency_s, trace):
        if not self._mlflow_ready:
            return
        try:
            import mlflow

            with mlflow.start_run(run_name=f"turn-{int(time.time())}"):
                mlflow.set_tags(
                    {
                        **self.config.get_default_tags(),
                        "tool_called": ",".join(sorted(set(tools_called))) or "none",
                        "status": status,
                    }
                )
                mlflow.log_params(
                    {
                        "model": self.model,
                        "temperature": self.config.get_temperature(),
                        "prompt_alias": self.config.get_active_alias(),
                        "prompt_version": self.config.get_prompt_version() or "unknown",
                        "dry_run": self.config.is_dry_run(),
                    }
                )
                mlflow.log_metrics(
                    {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                        "latency_s": latency_s,
                        "num_model_calls": num_model_calls,
                        "num_tool_calls": num_tool_calls,
                    }
                )
                with tempfile.TemporaryDirectory() as tmp_dir:
                    trace_path = Path(tmp_dir) / "trace.json"
                    trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False, default=str))
                    mlflow.log_artifact(str(trace_path), artifact_path="trace")
        except Exception as e:
            logger.warning(f"Failed to log MLflow run: {e}")
