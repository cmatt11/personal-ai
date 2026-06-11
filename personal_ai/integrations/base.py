"""Integration base class and registry."""

import os
from typing import Any, Dict, List, Optional

from ..config import Config
from ..connectivity import is_online

STATUS_CONNECTED = "connected"
STATUS_NEEDS_SETUP = "needs setup"
STATUS_OFFLINE = "offline"


class Integration:
    """Base class for a service integration.

    Subclasses set id/name/category/required_env/docs_url and implement
    actions() and call().
    """

    id: str = "integration"
    name: str = "Integration"
    category: str = "Other"
    required_env: List[str] = []
    docs_url: str = ""
    requires_internet: bool = True
    local: bool = False  # True for offline-capable integrations (e.g. local notes)

    def __init__(self, config: Config) -> None:
        self.config = config

    def env(self, key: str) -> Optional[str]:
        return os.environ.get(key)

    def is_configured(self) -> bool:
        return all(os.environ.get(k) for k in self.required_env)

    def status(self) -> str:
        if self.local:
            return STATUS_CONNECTED
        if not self.is_configured():
            return STATUS_NEEDS_SETUP
        if self.requires_internet and not is_online():
            return STATUS_OFFLINE
        return STATUS_CONNECTED

    def setup_hint(self) -> str:
        if self.is_configured() or self.local:
            return ""
        envs = ", ".join(self.required_env)
        hint = f"set {envs}"
        if self.docs_url:
            hint += f" (see {self.docs_url})"
        return hint

    def actions(self) -> Dict[str, str]:
        """Map of action name -> description."""
        return {}

    def call(self, action: str, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def guarded_call(self, action: str, params: Dict[str, Any]) -> str:
        """Check status before calling, with friendly errors."""
        st = self.status()
        if st == STATUS_NEEDS_SETUP:
            return f"{self.name} is not connected. To connect: {self.setup_hint()}."
        if st == STATUS_OFFLINE:
            return f"{self.name} needs an internet connection, and you are offline."
        if action not in self.actions():
            avail = ", ".join(self.actions().keys()) or "(none)"
            return f"Unknown action '{action}' for {self.name}. Available: {avail}"
        try:
            return self.call(action, params)
        except Exception as exc:
            return f"error calling {self.name}.{action}: {exc}"


class IntegrationRegistry:
    def __init__(self) -> None:
        self._items: Dict[str, Integration] = {}

    def register(self, integration: Integration) -> None:
        self._items[integration.id] = integration

    def get(self, integration_id: str) -> Optional[Integration]:
        return self._items.get(integration_id)

    def all(self) -> List[Integration]:
        return list(self._items.values())

    def panel(self) -> str:
        """Render the integrations panel grouped by category."""
        by_cat: Dict[str, List[Integration]] = {}
        for it in self._items.values():
            by_cat.setdefault(it.category, []).append(it)
        lines = ["Integrations panel:"]
        marks = {STATUS_CONNECTED: "[on]", STATUS_NEEDS_SETUP: "[--]", STATUS_OFFLINE: "[zz]"}
        for cat in sorted(by_cat):
            lines.append(f"\n{cat}:")
            for it in by_cat[cat]:
                st = it.status()
                line = f"  {marks.get(st, '[? ]')} {it.name} ({it.id}) - {st}"
                hint = it.setup_hint()
                if hint:
                    line += f"\n        connect: {hint}"
                lines.append(line)
        return "\n".join(lines)
