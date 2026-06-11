"""Tools that let the assistant program and reprogram itself at runtime."""

from typing import Any, Dict

from ..self_extend import PLUGIN_TEMPLATE_HINT, SelfExtender
from .base import Tool


class ListCapabilitiesTool(Tool):
    name = "list_capabilities"
    description = "List your built-in tools, your self-added tools, and plugin files."
    parameters = {}

    def __init__(self, extender: SelfExtender) -> None:
        self.extender = extender

    def run(self, args: Dict[str, Any]) -> str:
        return self.extender.list_capabilities()


class ReadCapabilityTool(Tool):
    name = "read_capability"
    description = "Read the source code of one of your self-added plugins."
    parameters = {"name": "The plugin name."}

    def __init__(self, extender: SelfExtender) -> None:
        self.extender = extender

    def run(self, args: Dict[str, Any]) -> str:
        return self.extender.read_capability(str(args.get("name", "")))


class CreateCapabilityTool(Tool):
    name = "create_capability"
    description = (
        "Add a NEW ability to yourself, or replace one, by writing a plugin and "
        "loading it live. Provide a plugin name and Python code. The code must "
        "define get_tools(config, memory) returning Tool instances. "
        + PLUGIN_TEMPLATE_HINT
    )
    parameters = {
        "name": "A short name for the plugin (becomes the file name).",
        "code": "The full Python source of the plugin.",
    }

    def __init__(self, extender: SelfExtender) -> None:
        self.extender = extender

    def run(self, args: Dict[str, Any]) -> str:
        name = str(args.get("name", "")).strip()
        code = str(args.get("code", ""))
        if not name or not code.strip():
            return "error: both 'name' and 'code' are required"
        return self.extender.create_capability(name, code)


class RemoveCapabilityTool(Tool):
    name = "remove_capability"
    description = "Remove one of your self-added plugins (its code is backed up first)."
    parameters = {"name": "The plugin name to remove."}

    def __init__(self, extender: SelfExtender) -> None:
        self.extender = extender

    def run(self, args: Dict[str, Any]) -> str:
        name = str(args.get("name", "")).strip()
        if not name:
            return "error: 'name' is required"
        return self.extender.remove_capability(name)


class RestoreCapabilityTool(Tool):
    name = "restore_capability"
    description = "Restore the most recent backup of a self-added plugin."
    parameters = {"name": "The plugin name to restore."}

    def __init__(self, extender: SelfExtender) -> None:
        self.extender = extender

    def run(self, args: Dict[str, Any]) -> str:
        name = str(args.get("name", "")).strip()
        if not name:
            return "error: 'name' is required"
        return self.extender.restore_backup(name)
