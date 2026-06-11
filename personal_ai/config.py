"""Configuration and local data paths.

All settings come from environment variables with sensible defaults so the
assistant runs out of the box. Data is stored locally under PERSONAL_AI_HOME
(default: ~/.personal_ai).
"""

import os
from pathlib import Path


def _bool_env(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class Config:
    """Central configuration, populated from the environment."""

    def __init__(self) -> None:
        # Where all local data lives.
        self.home = Path(
            os.environ.get("PERSONAL_AI_HOME", str(Path.home() / ".personal_ai"))
        ).expanduser()
        self.db_path = self.home / "personal_ai.db"
        # A sandboxed workspace the file tools are allowed to touch.
        self.workspace = Path(
            os.environ.get("PERSONAL_AI_WORKSPACE", str(self.home / "workspace"))
        ).expanduser()

        # Backend preference: "auto" (local first, then online), "local", "online".
        self.backend = os.environ.get("PERSONAL_AI_BACKEND", "auto").strip().lower()

        # Local (Ollama) settings.
        self.ollama_host = os.environ.get(
            "OLLAMA_HOST", "http://localhost:11434"
        ).rstrip("/")
        self.local_model = os.environ.get("PERSONAL_AI_LOCAL_MODEL", "llama3.2")
        self.local_embed_model = os.environ.get(
            "PERSONAL_AI_LOCAL_EMBED_MODEL", "nomic-embed-text"
        )
        self.local_vision_model = os.environ.get(
            "PERSONAL_AI_LOCAL_VISION_MODEL", "llama3.2-vision"
        )

        # Online (OpenAI-compatible) settings.
        self.online_base_url = os.environ.get(
            "PERSONAL_AI_ONLINE_BASE_URL", "https://api.openai.com/v1"
        ).rstrip("/")
        self.online_api_key = os.environ.get("PERSONAL_AI_ONLINE_API_KEY") or os.environ.get(
            "OPENAI_API_KEY"
        )
        self.online_model = os.environ.get("PERSONAL_AI_ONLINE_MODEL", "gpt-4o-mini")
        self.online_embed_model = os.environ.get(
            "PERSONAL_AI_ONLINE_EMBED_MODEL", "text-embedding-3-small"
        )
        self.online_vision_model = os.environ.get(
            "PERSONAL_AI_ONLINE_VISION_MODEL", "gpt-4o-mini"
        )

        # Behavior.
        self.max_steps = int(os.environ.get("PERSONAL_AI_MAX_STEPS", "8"))
        self.history_turns = int(os.environ.get("PERSONAL_AI_HISTORY_TURNS", "12"))
        self.memory_top_k = int(os.environ.get("PERSONAL_AI_MEMORY_TOP_K", "4"))
        self.context_char_budget = int(
            os.environ.get("PERSONAL_AI_CONTEXT_CHAR_BUDGET", "12000")
        )
        self.consolidate_every = int(os.environ.get("PERSONAL_AI_CONSOLIDATE_EVERY", "25"))
        self.auto_remember = _bool_env("PERSONAL_AI_AUTO_REMEMBER", True)
        self.allow_web = _bool_env("PERSONAL_AI_ALLOW_WEB", True)
        self.allow_code = _bool_env("PERSONAL_AI_ALLOW_CODE", True)
        self.allow_shell = _bool_env("PERSONAL_AI_ALLOW_SHELL", True)
        self.allow_self_modify = _bool_env("PERSONAL_AI_ALLOW_SELF_MODIFY", True)
        # Optional encryption-at-rest passphrase (needs the cryptography package).
        self.passphrase = os.environ.get("PERSONAL_AI_PASSPHRASE")
        self.request_timeout = int(os.environ.get("PERSONAL_AI_REQUEST_TIMEOUT", "120"))

    def ensure_dirs(self) -> None:
        """Create the local data directories if they do not exist."""
        self.home.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)


# A module-level default instance for convenience.
config = Config()
