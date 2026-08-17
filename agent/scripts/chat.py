"""
Local end-to-end validation REPL for Calendar Agent.

Usage:
    python -m scripts.chat

No deploy, no UI — this is the "validação ponta a ponta local" from the MVP
scope. Type "sair" or Ctrl-C to exit.
"""

import sys

from agent.src.agent import CalendarAgent, CONFIRM_YES
from agent.src.logging_config import setup_logging

setup_logging()


def main():
    agent = CalendarAgent()
    messages = []

    print("Calendar Agent — digite 'sair' para encerrar.\n")
    while True:
        try:
            user_input = input("você> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"sair", "exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_input})
        result = agent.handle_turn(messages)

        while result.pending is not None:
            try:
                answer = input(f"\n{result.text}\n").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            result = agent.confirm_pending_action(messages, result.pending, answer in CONFIRM_YES)

        print(f"agent> {result.text}\n")


if __name__ == "__main__":
    sys.exit(main())
