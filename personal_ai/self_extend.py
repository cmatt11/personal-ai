"""Self-extension: let the assistant program and reprogram itself.

The assistant adds new abilities by writing plugin files into its plugins
folder, then hot-loading them at runtime. It removes abilities by deleting
those files. All changes are validated and backed up so a bad edit can never
break the running app:

- New code is syntax-checked before it is saved.
- It is then imported in isolation to catch import-time errors.
- Existing files are backed up before being overwritten or removed.
- Only plugin-provided tools can be changed; the core stays protected.

This is real self-modification, kept reversible and contained.
"""

import re
import shutil
import time
from pathlib import Path
from typing import List, Optional, Set

from .config import Config
from .memory import Memory
from . import plugins
from .tools.base import ToolRegistry

PLUGIN_TEMPLATE_HINT = '''A plugin is a Python file that defines tools. Example:

from personal_ai.tools.base import Tool

class GreetTool(Tool):
    name = "greet"
    description = "Greet someone by name."
    parameters = {"who": "The name to greet."}
    def run(self, args):
        return f"Hello, {args.get('who', 'friend')}!"

def get_tools(config, memory):
    return [GreetTool()]
'''


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", name.strip().lower()).strip("_")
    return s or "plugin"


class SelfExtender:
    """Manages the assistant's self-written plugins."""

    def __init__(self, registry: ToolRegistry, config: Config, memory: Memory) -> None:
        self.registry = registry
        self.config = config
        self.memory = memory
        self.core_names: Set[str] = set()  # set via snapshot_core()

    def snapshot_core(self) -> None:
        """Record the built-in tools so reload knows what not to remove."""
        self.core_names = set(self.registry.names())

    def dir(self) -> Path:
        return plugins.plugins_dir(self.config)

    def backups_dir(self) -> Path:
        d = self.dir() / ".backups"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ----- inspection -----

    def list_capabilities(self) -> str:
        core = sorted(n for n in self.registry.names() if n in self.core_names)
        added = sorted(n for n in self.registry.names() if n not in self.core_names)
        files = sorted(p.stem for p in self.dir().glob("*.py") if not p.name.startswith("_"))
        out = [f"Built-in tools ({len(core)}): " + ", ".join(core)]
        out.append(f"Self-added tools ({len(added)}): " + (", ".join(added) or "(none)"))
        out.append(f"Plugin files: " + (", ".join(files) or "(none)"))
        return "\n".join(out)

    def read_capability(self, name: str) -> str:
        path = self.dir() / (_slug(name) + ".py")
        if not path.is_file():
            return f"error: no plugin named '{name}'"
        return path.read_text(encoding="utf-8")

    # ----- reload -----

    def reload(self) -> List[str]:
        """Remove all plugin tools, then re-load every plugin file."""
        for n in list(self.registry.names()):
            if n not in self.core_names:
                self.registry.unregister(n)
        return plugins.load_plugins(self.registry, self.config, self.memory)

    # ----- modification -----

    def create_capability(self, name: str, code: str) -> str:
        """Write a new plugin (or replace one), validate it, and hot-load it."""
        slug = _slug(name)
        # 1. Syntax check before touching disk.
        try:
            compile(code, f"<{slug}>", "exec")
        except SyntaxError as exc:
            return f"error: syntax error on line {exc.lineno}: {exc.msg}\n\n{PLUGIN_TEMPLATE_HINT}"

        if "get_tools" not in code and "def register" not in code:
            return (
                "error: a plugin must define get_tools(config, memory) or "
                "register(registry, config, memory).\n\n" + PLUGIN_TEMPLATE_HINT
            )

        path = self.dir() / (slug + ".py")
        backup = None
        if path.exists():
            backup = self.backups_dir() / f"{slug}.{int(time.time())}.py"
            shutil.copy2(path, backup)

        before = set(self.registry.names())
        path.write_text(code, encoding="utf-8")

        # 2. Import in isolation to catch import-time errors before activating.
        try:
            module = plugins.import_one(path)
        except Exception as exc:
            # Roll back: restore backup or remove the broken file.
            if backup is not None:
                shutil.copy2(backup, path)
            else:
                path.unlink(missing_ok=True)
            self.reload()
            return f"error: the code failed to import and was not kept: {exc}"

        # 3. Activate via a full reload so removals/renames are clean.
        self.reload()
        after = set(self.registry.names())
        added = sorted(after - before)
        if not added:
            return (
                f"Saved plugin '{slug}', but it did not register any new tool. "
                "Make sure get_tools returns Tool instances."
            )
        return f"Added capability. New tool(s): {', '.join(added)}. Saved as plugin '{slug}'."

    def remove_capability(self, name: str) -> str:
        """Remove a plugin file (backed up first) and unload its tools."""
        slug = _slug(name)
        path = self.dir() / (slug + ".py")
        if not path.is_file():
            files = ", ".join(p.stem for p in self.dir().glob("*.py")) or "(none)"
            return f"error: no plugin named '{slug}'. Plugins: {files}"
        before = set(self.registry.names())
        backup = self.backups_dir() / f"{slug}.{int(time.time())}.removed.py"
        shutil.move(str(path), str(backup))
        self.reload()
        removed = sorted(before - set(self.registry.names()))
        return (
            f"Removed plugin '{slug}'"
            + (f" and tool(s): {', '.join(removed)}" if removed else "")
            + f". A backup is kept at {backup.name}."
        )

    def restore_backup(self, name: str) -> str:
        """Restore the most recent backup of a plugin."""
        slug = _slug(name)
        candidates = sorted(self.backups_dir().glob(f"{slug}.*.py"))
        if not candidates:
            return f"error: no backup found for '{slug}'"
        latest = candidates[-1]
        target = self.dir() / (slug + ".py")
        shutil.copy2(latest, target)
        self.reload()
        return f"Restored '{slug}' from {latest.name}."
