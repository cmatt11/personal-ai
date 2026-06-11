"""Plugin loader: add your own tools without touching the core.

Drop a .py file in the plugins folder (default: ~/.personal_ai/plugins/).
Each plugin may define either of these top-level functions:

    def get_tools(config, memory):
        # return a list of Tool instances
        return [MyTool()]

    def register(registry, config, memory):
        # register tools directly
        registry.register(MyTool())

Plugins are optional. A broken plugin is skipped with a warning, never crashes
the assistant.
"""

import importlib.util
import sys
from pathlib import Path
from typing import List

from .config import Config
from .memory import Memory
from .tools.base import ToolRegistry


def plugins_dir(config: Config) -> Path:
    d = config.home / "plugins"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_plugins(registry: ToolRegistry, config: Config, memory: Memory) -> List[str]:
    """Load all plugins, registering their tools. Returns loaded plugin names."""
    loaded: List[str] = []
    directory = plugins_dir(config)
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            module = _import_file(path)
        except Exception as exc:
            sys.stderr.write(f"[plugin] failed to load {path.name}: {exc}\n")
            continue

        try:
            if hasattr(module, "register"):
                module.register(registry, config, memory)
            if hasattr(module, "get_tools"):
                for tool in module.get_tools(config, memory) or []:
                    registry.register(tool)
            loaded.append(path.stem)
        except Exception as exc:
            sys.stderr.write(f"[plugin] error initializing {path.name}: {exc}\n")
    return loaded


def _import_file(path: Path):
    spec = importlib.util.spec_from_file_location(f"personal_ai_plugin_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



def import_one(path: Path):
    """Import a single plugin file, raising on error. Returns the module."""
    return _import_file(path)


def register_module(module, registry: ToolRegistry, config: Config, memory: Memory) -> None:
    """Register the tools a loaded plugin module provides."""
    if hasattr(module, "register"):
        module.register(registry, config, memory)
    if hasattr(module, "get_tools"):
        for tool in module.get_tools(config, memory) or []:
            registry.register(tool)
