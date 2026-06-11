"""Tool package: builds the registry of tools available to the agent."""

from typing import Optional

from ..config import Config
from ..integrations import build_integrations
from ..llm import LLMBackend
from ..memory import Memory
from .base import Tool, ToolRegistry
from .calculator import CalculatorTool
from .code_exec import RunPythonTool
from .datetime_tool import DateTimeTool
from .files import DeleteFileTool, ListFilesTool, ReadFileTool, WriteFileTool
from .integrations_tool import IntegrationsPanelTool, UseIntegrationTool
from .knowledge import IngestFileTool
from .memory_tool import RecallTool, RememberTool
from .profile_tool import ForgetProfileTool, RememberAboutMeTool, ViewProfileTool
from .search import FindFilesTool, SearchFilesTool
from .self_tool import (
    CreateCapabilityTool,
    ListCapabilitiesTool,
    ReadCapabilityTool,
    RemoveCapabilityTool,
    RestoreCapabilityTool,
)
from .shell import RunShellTool
from .tasks import AddTaskTool, CompleteTaskTool, ListTasksTool
from .vision import AnalyzeImageTool
from .web import WebFetchTool, WebSearchTool


def build_registry(
    config: Config, memory: Memory, vision_backend: Optional[LLMBackend] = None
) -> ToolRegistry:
    """Assemble all tools. Web, code, and shell tools depend on config flags."""
    registry = ToolRegistry()

    # Always-on, offline tools.
    registry.register(CalculatorTool())
    registry.register(DateTimeTool())
    registry.register(ReadFileTool(config.workspace))
    registry.register(WriteFileTool(config.workspace))
    registry.register(ListFilesTool(config.workspace))
    registry.register(DeleteFileTool(config.workspace))
    registry.register(FindFilesTool(config.workspace))
    registry.register(SearchFilesTool(config.workspace))
    registry.register(RememberTool(memory))
    registry.register(RecallTool(memory, top_k=config.memory_top_k))
    registry.register(IngestFileTool(config.workspace, memory))
    registry.register(AddTaskTool(memory))
    registry.register(ListTasksTool(memory))
    registry.register(CompleteTaskTool(memory))

    # Profile (auto-memory) management.
    registry.register(RememberAboutMeTool(memory))
    registry.register(ViewProfileTool(memory))
    registry.register(ForgetProfileTool(memory))

    # Image analysis (needs a vision-capable model).
    registry.register(AnalyzeImageTool(config.workspace, vision_backend))

    # Integrations panel.
    integrations = build_integrations(config)
    registry.register(IntegrationsPanelTool(integrations))
    registry.register(UseIntegrationTool(integrations))

    # Gated tools.
    if config.allow_code:
        registry.register(RunPythonTool(config.workspace))
    if config.allow_shell:
        registry.register(RunShellTool(config.workspace))
    if config.allow_web:
        registry.register(WebSearchTool())
        registry.register(WebFetchTool())

    # Self-extension: let the assistant program/reprogram itself via plugins.
    if config.allow_self_modify:
        from ..self_extend import SelfExtender

        extender = SelfExtender(registry, config, memory)
        registry.register(ListCapabilitiesTool(extender))
        registry.register(ReadCapabilityTool(extender))
        registry.register(CreateCapabilityTool(extender))
        registry.register(RemoveCapabilityTool(extender))
        registry.register(RestoreCapabilityTool(extender))
        # Snapshot the built-in tools so reloads never remove core capabilities.
        extender.snapshot_core()
        registry.self_extender = extender  # exposed for the CLI/web layers

    return registry


__all__ = [
    "Tool",
    "ToolRegistry",
    "build_registry",
]
