"""
Minimal tool-selection evaluation harness.

For each case in eval/cases.yaml, sends a single message to the model with
the two tool schemas and checks which tool (if any) it decided to call. Tools
are never executed here, so this never touches the calendar — safe to run
against the real DeepSeek API as a pre-flight check before considering the
MVP done.

Usage:
    python -m eval.run_eval
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from openai import OpenAI

from agent.src.config import get_config
from agent.src.logging_config import setup_logging
from agent.src.tools import TOOL_SCHEMAS

setup_logging()
logger = logging.getLogger(__name__)

CASES_PATH = Path(__file__).parent / "cases.yaml"


def _build_system(config) -> str:
    tz = ZoneInfo(config.get_timezone())
    now = datetime.now(tz)
    system_prompt = config.load_system_prompt()
    return system_prompt + (
        f"\n\nContexto atual: agora é {now.strftime('%A, %d de %B de %Y, %H:%M')} ({tz.key})."
    )


def _selected_tool(response) -> str:
    tool_calls = response.choices[0].message.tool_calls
    if tool_calls:
        return tool_calls[0].function.name
    return "none"


def run_eval():
    config = get_config()
    client = OpenAI(api_key=config.get_deepseek_api_key(), base_url=config.get_deepseek_base_url())
    system = _build_system(config)

    cases = yaml.safe_load(CASES_PATH.read_text())

    results = []
    for case in cases:
        response = client.chat.completions.create(
            model=config.get_model(),
            max_tokens=config.get_max_tokens(),
            temperature=config.get_temperature(),
            tools=TOOL_SCHEMAS,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": case["input"]},
            ],
        )
        selected = _selected_tool(response)
        passed = selected == case["expected_tool"]
        results.append({**case, "selected_tool": selected, "passed": passed})
        status = "OK" if passed else "FAIL"
        logger.info(f"[{status}] '{case['input']}' -> expected={case['expected_tool']} got={selected}")

    accuracy = sum(r["passed"] for r in results) / len(results)
    passed_count = sum(r["passed"] for r in results)
    logger.info(f"Tool selection accuracy: {accuracy:.0%} ({passed_count}/{len(results)})")

    if config.setup_mlflow():
        try:
            import mlflow

            with mlflow.start_run(run_name=f"eval-{int(time.time())}"):
                mlflow.set_tags({**config.get_default_tags(), "eval": "true"})
                mlflow.log_param("model", config.get_model())
                mlflow.log_param("num_cases", len(results))
                mlflow.log_metric("tool_selection_accuracy", accuracy)
        except Exception as e:
            logger.warning(f"Failed to log eval run to MLflow: {e}")

    return results, accuracy


def main():
    results, _ = run_eval()
    failures = [r for r in results if not r["passed"]]
    if failures:
        logger.error(f"{len(failures)} case(s) failed tool selection.")
        raise SystemExit(1)
    logger.info("All tool-selection eval cases passed.")


if __name__ == "__main__":
    main()
