"""Unit tests for the Telegram webhook receiver. CalendarAgent is replaced with a MagicMock via telegram_bot._agent."""

from unittest.mock import MagicMock, patch

import agent.src.telegram_bot as telegram_bot
from agent.src.agent import PendingAction, TurnResult
from agent.src.telegram_bot import _sessions, app

SECRET = "s3cr3t"
ALLOWED_ID = 12345
TOKEN = "bot-token"
HEADERS = {"X-Telegram-Bot-Api-Secret-Token": SECRET}


def _mock_config():
    config = MagicMock()
    config.get_telegram_webhook_secret.return_value = SECRET
    config.get_telegram_allowed_user_id.return_value = ALLOWED_ID
    config.get_telegram_bot_token.return_value = TOKEN
    return config


def _update(chat_id, from_id, text):
    return {"message": {"chat": {"id": chat_id}, "from": {"id": from_id}, "text": text}}


def _setup():
    _sessions.clear()
    telegram_bot._agent = MagicMock()
    return app.test_client()


@patch("agent.src.telegram_bot._send_message")
@patch("agent.src.telegram_bot.get_config")
class TestWebhookAuth:
    def test_missing_secret_header_is_rejected(self, mock_get_config, mock_send):
        mock_get_config.return_value = _mock_config()
        client = _setup()

        resp = client.post("/webhook", json=_update(ALLOWED_ID, ALLOWED_ID, "oi"))

        assert resp.status_code == 403
        telegram_bot._agent.handle_turn.assert_not_called()
        mock_send.assert_not_called()

    def test_wrong_secret_header_is_rejected(self, mock_get_config, mock_send):
        mock_get_config.return_value = _mock_config()
        client = _setup()

        resp = client.post(
            "/webhook", json=_update(ALLOWED_ID, ALLOWED_ID, "oi"), headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"}
        )

        assert resp.status_code == 403
        mock_send.assert_not_called()

    def test_unauthorized_sender_is_silently_ignored(self, mock_get_config, mock_send):
        mock_get_config.return_value = _mock_config()
        client = _setup()

        resp = client.post("/webhook", json=_update(99999, 99999, "oi"), headers=HEADERS)

        assert resp.status_code == 200
        telegram_bot._agent.handle_turn.assert_not_called()
        mock_send.assert_not_called()

    def test_non_message_update_is_ignored(self, mock_get_config, mock_send):
        mock_get_config.return_value = _mock_config()
        client = _setup()

        resp = client.post("/webhook", json={"some_other_update": {}}, headers=HEADERS)

        assert resp.status_code == 200
        telegram_bot._agent.handle_turn.assert_not_called()
        mock_send.assert_not_called()


@patch("agent.src.telegram_bot._send_message")
@patch("agent.src.telegram_bot.get_config")
class TestWebhookConversationFlow:
    def test_create_then_confirm_clears_pending(self, mock_get_config, mock_send):
        mock_get_config.return_value = _mock_config()
        client = _setup()

        pending = PendingAction(
            kind="create",
            prompt="Confirmar criação? [s/N]",
            message_index=2,
            args={"title": "Dentista", "start": "2026-08-20T10:00:00", "end": "2026-08-20T11:00:00"},
        )
        telegram_bot._agent.handle_turn.return_value = TurnResult(text=pending.prompt, pending=pending)

        resp = client.post("/webhook", json=_update(ALLOWED_ID, ALLOWED_ID, "cria dentista amanha"), headers=HEADERS)

        assert resp.status_code == 200
        assert _sessions[ALLOWED_ID]["pending"] is pending
        mock_send.assert_called_with(TOKEN, ALLOWED_ID, pending.prompt)

        telegram_bot._agent.confirm_pending_action.return_value = TurnResult(text="Evento criado!", pending=None)

        resp2 = client.post("/webhook", json=_update(ALLOWED_ID, ALLOWED_ID, "sim"), headers=HEADERS)

        assert resp2.status_code == 200
        telegram_bot._agent.confirm_pending_action.assert_called_once_with(
            _sessions[ALLOWED_ID]["messages"], pending, True
        )
        assert _sessions[ALLOWED_ID]["pending"] is None
        mock_send.assert_called_with(TOKEN, ALLOWED_ID, "Evento criado!")

    def test_pending_reply_interpreted_as_no_cancels(self, mock_get_config, mock_send):
        mock_get_config.return_value = _mock_config()
        client = _setup()
        pending = PendingAction(kind="delete", prompt="Confirmar remoção? [s/N]", message_index=0)
        _sessions[ALLOWED_ID] = {"messages": [], "pending": pending}
        telegram_bot._agent.confirm_pending_action.return_value = TurnResult(text="Cancelado.", pending=None)

        resp = client.post("/webhook", json=_update(ALLOWED_ID, ALLOWED_ID, "não"), headers=HEADERS)

        assert resp.status_code == 200
        telegram_bot._agent.confirm_pending_action.assert_called_once_with([], pending, False)
        assert _sessions[ALLOWED_ID]["pending"] is None
