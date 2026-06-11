"""Tools that expose the integrations panel to the agent."""

import json
from typing import Any, Dict

from ..integrations.base import IntegrationRegistry
from .base import Tool


class IntegrationsPanelTool(Tool):
    name = "integrations_panel"
    description = (
        "Show the integrations panel: every connected service (Google, "
        "Microsoft, social, dev tools, notes), its status, and how to connect it."
    )
    parameters = {}

    def __init__(self, registry: IntegrationRegistry) -> None:
        self.registry = registry

    def run(self, args: Dict[str, Any]) -> str:
        return self.registry.panel()


class UseIntegrationTool(Tool):
    name = "use_integration"
    description = (
        "Call an action on an integration. First check integrations_panel for "
        "ids and actions. Example: integration='github', action='list_repos'."
    )
    parameters = {
        "integration": "The integration id, e.g. 'github', 'gmail', 'notes'.",
        "action": "The action name to run.",
        "params": "Optional JSON object of action parameters.",
    }

    def __init__(self, registry: IntegrationRegistry) -> None:
        self.registry = registry

    def run(self, args: Dict[str, Any]) -> str:
        integration_id = str(args.get("integration", "")).strip()
        action = str(args.get("action", "")).strip()
        if not integration_id or not action:
            return "error: 'integration' and 'action' are required"
        integration = self.registry.get(integration_id)
        if integration is None:
            ids = ", ".join(i.id for i in self.registry.all())
            return f"error: unknown integration '{integration_id}'. Available: {ids}"
        params = args.get("params", {})
        if isinstance(params, str):
            try:
                params = json.loads(params) if params.strip() else {}
            except json.JSONDecodeError:
                return "error: 'params' must be a JSON object"
        if not isinstance(params, dict):
            params = {}
        return integration.guarded_call(action, params)
