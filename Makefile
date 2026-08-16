.PHONY: sync-run agent-lint agent-test agent-eval agent-register-prompt agent-promote agent-run

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
