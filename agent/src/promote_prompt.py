"""
Promote the current "challenger" prompt version to "champion".

Usage:
    python -m src.promote_prompt

Reads whichever version the challenger alias currently points to and points
champion at that same version. Kept as a separate, deliberate step from
register_prompts.py — promoting to production should never be an accidental
side effect of registering an edit; it's something you run yourself after
validating the challenger (e.g. with `production_mode: false` + `make eval`).
"""

import logging
import sys

import mlflow

from agent.src.config import get_config
from agent.src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def main():
    config = get_config()

    if not config.setup_mlflow():
        logger.error("Failed to set up MLflow (check MLFLOW_TRACKING_URI). Exiting.")
        sys.exit(1)

    prompt_name = config.get_prompt_name()
    dev_alias = config.get_dev_prompt_alias()
    prod_alias = config.get_prod_prompt_alias()

    try:
        from mlflow import genai as mlflow_genai

        challenger_prompt = mlflow_genai.load_prompt(f"prompts:/{prompt_name}@{dev_alias}")
    except Exception as e:
        logger.error(f"Could not load '{dev_alias}' version of '{prompt_name}': {e}")
        sys.exit(1)

    version = challenger_prompt.version
    client = mlflow.tracking.MlflowClient()
    client.set_prompt_alias(name=prompt_name, alias=prod_alias, version=version)
    logger.info(f"Promoted '{prompt_name}' version {version} from '{dev_alias}' to '{prod_alias}'")


if __name__ == "__main__":
    main()
