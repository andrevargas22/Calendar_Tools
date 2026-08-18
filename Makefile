.PHONY: sync-run agent-lint agent-test agent-eval agent-register-prompt agent-promote agent-run agent-serve agent-set-webhook

# --- sync ---

sync-run:
	python -m sync.calendar_sync

# --- agent ---

agent-lint:
	python -m py_compile agent/src/*.py agent/scripts/*.py agent/eval/*.py

agent-test:
	pytest agent/tests/ -v

agent-eval:
	python -m agent.eval.run_eval

agent-register-prompt:
	python -m agent.src.register_prompts

agent-promote:
	python -m agent.src.promote_prompt

agent-run:
	python -m agent.scripts.chat

# local dev server; tunnel it (e.g. ngrok) + agent-set-webhook to test real Telegram messages
agent-serve:
	FLASK_APP=agent.src.telegram_bot:app flask run --port 8080

agent-set-webhook:
	python -m agent.scripts.set_webhook $(URL)
