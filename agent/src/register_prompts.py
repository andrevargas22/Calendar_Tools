"""
Register the system prompt from config/prompts.yaml into the MLflow Prompt
Registry, versioning it and pointing the "challenger" alias at the new
version. Mirrors Colorado_IA/src/register_prompts.py.

Usage:
    python -m src.register_prompts

Promoting a version to production (the "champion" alias, used when
agent.production_mode is true) is a deliberate manual step, not part of this
script — see src/promote_prompt.py / `make promote`.
"""

import logging
import sys

import mlflow

from agent.src.config import get_config
from agent.src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def register_prompt(config) -> bool:
    logger.info("Registering system prompt to MLflow...")
    try:
        system_prompt = config.get_system_prompt()
        prompt_name = config.get_prompt_name()

        client = mlflow.tracking.MlflowClient()

        try:
            client.create_prompt(name=prompt_name, description="Calendar Agent system prompt")
            logger.info(f"Created new prompt '{prompt_name}'")
        except Exception:
            logger.info(f"Prompt '{prompt_name}' already exists")

        prompt_version = client.create_prompt_version(
            name=prompt_name, template=system_prompt, description="Updated system prompt"
        )

        dev_alias = config.get_dev_prompt_alias()
        client.set_prompt_alias(name=prompt_name, alias=dev_alias, version=prompt_version.version)

        logger.info(f"Prompt registered as '{prompt_name}', version {prompt_version.version}, alias '{dev_alias}'")
        logger.info(
            f"Promote to '{config.get_prod_prompt_alias()}' manually once you've reviewed it "
            "(see module docstring)."
        )
        return True
    except Exception as e:
        logger.exception(f"Failed to register prompt: {e}")
        return False


def main():
    logger.info("Prompt registration started")
    config = get_config()

    if not config.load_prompts_from_yaml():
        logger.error("Failed to load config/prompts.yaml. Exiting.")
        sys.exit(1)

    if not config.setup_mlflow():
        logger.error("Failed to set up MLflow (check MLFLOW_TRACKING_URI). Exiting.")
        sys.exit(1)

    if not register_prompt(config):
        sys.exit(1)

    logger.info("Prompt registration completed successfully")


if __name__ == "__main__":
    main()
