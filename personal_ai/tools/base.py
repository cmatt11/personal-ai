"""Tool interface and registry.

A Tool exposes a name, a description, an argument schema (for the model to read),
and a run() method. The agent reads the schemas, decides which tool to call,
and dispatches here.
"""

from typing import Any, Dict, List


class Tool:
    """Base class for all tools."""

    name: str = "tool"
    description: str = ""
    # parameters: {arg_name: "description"} kept simple and model-friendly.
    parameters: Dict[str, str] = {}
    # Set True for tools that need the internet.
    requires_internet: bool = False

    def run(self, args: Dict[str, Any]) -> str:
        raise NotImplementedError

    def spec(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "requires_internet": self.requires_internet,
        }


class ToolRegistry:
    """Holds the set of tools available to the agent."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(name)
        return self._tools[name]

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def all(self) -> List[Tool]:
        return list(self._tools.values())

    def specs(self) -> List[Dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]
