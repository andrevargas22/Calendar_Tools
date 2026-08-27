"""
Configuration manager for Calendar Agent.

Same singleton pattern as Colorado_IA/src/core/config.py: YAML files under
config/ are the source of truth for non-secret settings, .env/.env.local
provide secrets and environment-specific overrides, and MLflow's Prompt
Registry is the versioned runtime source for the system prompt (config/
prompts.yaml is only the editable pre-registration source).
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

logger = logging.getLogger(__name__)


class ConfigManager:
    """Singleton configuration manager for Calendar Agent."""

    _instance: Optional["ConfigManager"] = None
    _agent: Dict[str, Any] = {}
    _mlflow: Dict[str, Any] = {}
    _prompt_registry: Dict[str, Any] = {}
    _prompts: Dict[str, Any] = {}
    _prompt_version: Optional[str] = None
    _config_dir: Path = Path(__file__).parent.parent / "config"
    _project_root: Path = Path(__file__).parent.parent

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_env()
            cls._instance._load_config()
        return cls._instance

    def _load_env(self) -> None:
        """Load .env.local (if present, wins) or .env, matching Colorado_IA.

        The .env lives at the repo root (Calendar_Tools/.env), one level above
        this project's own root, since it's shared with sync/ (same Google
        service account, distinct calendar id per tool).
        """
        if load_dotenv is None:
            logger.debug("python-dotenv not installed, skipping .env loading")
            return

        repo_root = self._project_root.parent

        env_local = repo_root / ".env.local"
        if env_local.exists():
            load_dotenv(env_local, override=True)
            logger.debug(f"Loaded environment from {env_local}")
            return

        env_file = repo_root / ".env"
        if env_file.exists():
            load_dotenv(env_file, override=True)
            logger.debug(f"Loaded environment from {env_file}")
        else:
            logger.debug("No .env file found, using system environment variables only")

    def _load_config(self) -> None:
        agent_path = self._config_dir / "agent.yaml"
        if not agent_path.exists():
            raise FileNotFoundError(f"Missing configuration file: {agent_path}")

        with open(agent_path, "r", encoding="utf-8") as f:
            self._agent = (yaml.safe_load(f) or {}).get("agent", {})

        mlflow_path = self._config_dir / "mlflow.yaml"
        if mlflow_path.exists():
            with open(mlflow_path, "r", encoding="utf-8") as f:
                raw_mlflow = yaml.safe_load(f) or {}

            tracking = raw_mlflow.get("tracking", {})
            uri = tracking.get("tracking_uri")
            if uri and uri.startswith("${") and uri.endswith("}"):
                env_var = uri[2:-1]
                tracking["tracking_uri"] = os.getenv(env_var, "")
            raw_mlflow["tracking"] = tracking

            self._mlflow = raw_mlflow
            self._prompt_registry = raw_mlflow.get("prompt_registry", {})
        else:
            logger.warning(f"MLflow config not found at {mlflow_path}")
            self._mlflow = {}
            self._prompt_registry = {}

    # -- agent settings ----------------------------------------------------

    def get_model(self) -> str:
        return self._agent.get("model", "deepseek-chat")

    def get_max_tokens(self) -> int:
        return int(self._agent.get("max_tokens", 1024))

    def get_timezone(self) -> str:
        return self._agent.get("timezone", "America/Sao_Paulo")

    def get_temperature(self) -> float:
        return float(self._agent.get("temperature", 0))

    def is_production_mode(self) -> bool:
        return bool(self._agent.get("production_mode", True))

    def is_dry_run(self) -> bool:
        return bool(self._agent.get("dry_run", True))

    # -- secrets / env -------------------------------------------------------

    def get_deepseek_api_key(self) -> Optional[str]:
        return os.getenv("DEEPSEEK_API_KEY")

    def get_deepseek_base_url(self) -> str:
        return self._agent.get("base_url", "https://api.deepseek.com")

    def get_google_service_account_key(self) -> Optional[str]:
        value = os.getenv("GOOGLE_SERVICE_ACCOUNT_KEY")
        return value.strip() if value else value

    def get_google_calendar_id(self) -> Optional[str]:
        value = os.getenv("GOOGLE_CALENDAR_ID_PERSONAL")
        return value.strip() if value else value

    def get_google_calendar_id_pets(self) -> Optional[str]:
        # A stray trailing newline in the GitHub secret once made this an
        # invalid calendarId, so the Calendar API returned 404.
        value = os.getenv("GOOGLE_CALENDAR_ID_PETS")
        return value.strip() if value else value

    # -- telegram bot ---------------------------------------------------------

    def get_telegram_bot_token(self) -> Optional[str]:
        return os.getenv("TELEGRAM_BOT_TOKEN")

    def get_telegram_allowed_user_id(self) -> Optional[int]:
        value = os.getenv("TELEGRAM_ALLOWED_USER_ID")
        return int(value) if value else None

    def get_telegram_webhook_secret(self) -> Optional[str]:
        return os.getenv("TELEGRAM_WEBHOOK_SECRET")

    # -- MLflow tracking -----------------------------------------------------

    def get_mlflow_tracking_uri(self) -> Optional[str]:
        uri = self._mlflow.get("tracking", {}).get("tracking_uri")
        return uri or None

    def get_experiment_name(self) -> str:
        return self._mlflow.get("experiment_name", "calendar-agent")

    def get_default_tags(self) -> Dict[str, str]:
        return dict(self._mlflow.get("default_tags", {}))

    def setup_mlflow(self) -> bool:
        """Configure MLflow tracking URI and experiment. Never raises."""
        try:
            import mlflow
        except ImportError:
            logger.warning("mlflow not installed, tracking disabled")
            return False

        tracking_uri = self.get_mlflow_tracking_uri()
        if not tracking_uri:
            logger.warning("MLFLOW_TRACKING_URI not set, tracking disabled")
            return False

        try:
            mlflow.set_tracking_uri(tracking_uri)
            experiment_name = self.get_experiment_name()
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
                logger.info(f"Created MLflow experiment: {experiment_name}")
            elif experiment.lifecycle_stage == "deleted":
                mlflow.tracking.MlflowClient().restore_experiment(experiment.experiment_id)
                logger.info(f"Restored MLflow experiment: {experiment_name}")
            mlflow.set_experiment(experiment_name)
            return True
        except Exception as e:
            logger.warning(f"Failed to set up MLflow tracking: {e}")
            return False

    # -- prompt registry -------------------------------------------------------

    def get_prompt_name(self) -> str:
        return self._prompt_registry.get("name", "calendar-agent-system-prompt")

    def get_dev_prompt_alias(self) -> str:
        return self._prompt_registry.get("dev_alias", "challenger")

    def get_prod_prompt_alias(self) -> str:
        return self._prompt_registry.get("prod_alias", "champion")

    def get_active_alias(self) -> str:
        return self.get_prod_prompt_alias() if self.is_production_mode() else self.get_dev_prompt_alias()

    def get_prompt_version(self) -> Optional[str]:
        return self._prompt_version

    def load_prompts_from_yaml(self) -> bool:
        """Load config/prompts.yaml. Used only by register_prompts.py."""
        prompts_path = self._config_dir / "prompts.yaml"
        if not prompts_path.exists():
            logger.error(f"Prompts config not found at {prompts_path}")
            return False
        try:
            with open(prompts_path, "r", encoding="utf-8") as f:
                self._prompts = yaml.safe_load(f) or {}
            return True
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in {prompts_path}: {e}")
            return False

    def get_system_prompt(self) -> str:
        if not self._prompts:
            raise ValueError("Prompts not loaded. Call load_prompts_from_yaml() or load_system_prompt() first.")
        return self._prompts.get("system_prompt", "").strip()

    def load_system_prompt(self, alias: Optional[str] = None, fallback_alias: Optional[str] = None) -> str:
        """
        Load the system prompt from the MLflow Prompt Registry.

        Tries `alias` first, then `fallback_alias`, then the latest version.
        Falls back to config/prompts.yaml on disk if MLflow is unreachable, so
        the agent can still run locally without a registered prompt.
        """
        alias = alias or self.get_active_alias()
        fallback_alias = fallback_alias or (
            self.get_dev_prompt_alias() if alias == self.get_prod_prompt_alias() else self.get_prod_prompt_alias()
        )

        try:
            from mlflow import genai as mlflow_genai
        except ImportError:
            logger.warning("mlflow not installed, loading system prompt from config/prompts.yaml instead")
            self.load_prompts_from_yaml()
            return self.get_system_prompt()

        prompt_name = self.get_prompt_name()
        for candidate_alias, label in ((alias, alias), (fallback_alias, f"{fallback_alias} (fallback)")):
            if not candidate_alias:
                continue
            try:
                prompt_uri = f"prompts:/{prompt_name}@{candidate_alias}"
                loaded = mlflow_genai.load_prompt(prompt_uri)
                self._prompt_version = getattr(loaded, "version", None)
                logger.info(f"Loaded system prompt '{prompt_name}' from alias '{label}' (version {self._prompt_version})")
                return loaded.template.strip()
            except Exception as e:
                logger.warning(f"Failed to load prompt alias '{candidate_alias}': {e}")

        logger.warning(
            f"Could not load prompt '{prompt_name}' from MLflow, falling back to config/prompts.yaml"
        )
        self.load_prompts_from_yaml()
        return self.get_system_prompt()


def get_config() -> ConfigManager:
    """Get the singleton ConfigManager instance."""
    return ConfigManager()
