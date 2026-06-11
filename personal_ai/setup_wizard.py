"""First-run setup wizard and environment doctor.

doctor() reports whether everything needed to run is present. run_wizard()
walks you through choosing a backend and writes a .env file you can source.
Both are safe to run anytime.
"""

import os
from typing import List

from .config import Config
from .connectivity import is_online, ollama_available
from .http_util import HttpError, get_json
from .platform_util import summary as platform_summary


def installed_ollama_models(host: str) -> List[str]:
    try:
        data = get_json(f"{host.rstrip('/')}/api/tags", timeout=3)
        return [m.get("name", "") for m in data.get("models", [])]
    except (HttpError, Exception):
        return []


def doctor(config: Config) -> str:
    """Return a readiness report for the current environment."""
    lines = ["Personal AI doctor", "=================="]
    lines.append(f"platform: {platform_summary()}")
    lines.append(f"data dir: {config.home}")
    online = is_online()
    lines.append(f"internet: {'yes' if online else 'no'}")

    local = ollama_available(config.ollama_host)
    lines.append(f"ollama running: {'yes' if local else 'no'} ({config.ollama_host})")
    if local:
        models = installed_ollama_models(config.ollama_host)
        have_chat = any(config.local_model in m for m in models)
        have_embed = any(config.local_embed_model in m for m in models)
        have_vision = any(config.local_vision_model in m for m in models)
        lines.append(f"  chat model '{config.local_model}': {'yes' if have_chat else 'MISSING'}")
        lines.append(f"  embed model '{config.local_embed_model}': {'yes' if have_embed else 'MISSING'}")
        lines.append(f"  vision model '{config.local_vision_model}': {'yes' if have_vision else 'optional, missing'}")

    has_key = bool(config.online_api_key)
    lines.append(f"online API key set: {'yes' if has_key else 'no'}")

    # Verdict.
    ready_local = local and any(config.local_model in m for m in installed_ollama_models(config.ollama_host))
    ready_online = has_key and online
    if ready_local or ready_online:
        lines.append("\nverdict: READY to chat.")
    else:
        lines.append("\nverdict: NOT READY. Do one of:")
        lines.append(f"  - Local: install Ollama, run `ollama serve`, then "
                     f"`ollama pull {config.local_model}` and `ollama pull {config.local_embed_model}`.")
        lines.append("  - Online: set PERSONAL_AI_ONLINE_API_KEY (or OPENAI_API_KEY) and connect to the internet.")
    return "\n".join(lines)


def run_wizard(config: Config) -> str:
    """Interactive setup. Writes a .env file in the data dir. Returns its path."""
    print("Personal AI setup")
    print("-----------------")
    print(doctor(config))
    print()

    env_lines: List[str] = []

    choice = _ask("Use local (Ollama) or online model? [local/online]", "local").lower()
    if choice.startswith("o"):
        env_lines.append("PERSONAL_AI_BACKEND=online")
        key = _ask("Paste your API key (PERSONAL_AI_ONLINE_API_KEY)", "").strip()
        if key:
            env_lines.append(f"PERSONAL_AI_ONLINE_API_KEY={key}")
        model = _ask("Online model name", config.online_model).strip()
        env_lines.append(f"PERSONAL_AI_ONLINE_MODEL={model}")
    else:
        env_lines.append("PERSONAL_AI_BACKEND=auto")
        model = _ask("Local model name", config.local_model).strip()
        env_lines.append(f"PERSONAL_AI_LOCAL_MODEL={model}")
        print(f"\nMake sure to run: ollama pull {model} && ollama pull {config.local_embed_model}")

    env_path = config.home / ".env"
    config.home.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    print(f"\nWrote {env_path}. Load it with:  set -a; . {env_path}; set +a")
    return str(env_path)


def _ask(prompt: str, default: str) -> str:
    try:
        ans = input(f"{prompt} [{default}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        return default
    return ans or default
