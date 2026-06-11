"""Local Notes integration. Works fully offline.

Stores plain-text notes as files under the workspace 'notes' folder, so they
are private, portable, and require no external service.
"""

import re
import time
from pathlib import Path
from typing import Any, Dict

from .base import STATUS_CONNECTED, Integration


def _slug(title: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", title.strip().lower()).strip("-")
    return s or time.strftime("note-%Y%m%d-%H%M%S")


class LocalNotesIntegration(Integration):
    id = "notes"
    name = "Notes (local)"
    category = "Notes"
    required_env = []
    requires_internet = False
    local = True

    def __init__(self, config) -> None:
        super().__init__(config)
        self.dir = Path(config.workspace) / "notes"
        self.dir.mkdir(parents=True, exist_ok=True)

    def status(self) -> str:
        return STATUS_CONNECTED

    def actions(self) -> Dict[str, str]:
        return {
            "add": "Create a note. params: title, body.",
            "list": "List all notes.",
            "read": "Read a note. params: title.",
            "delete": "Delete a note. params: title.",
        }

    def call(self, action: str, params: Dict[str, Any]) -> str:
        if action == "add":
            title = str(params.get("title", "")).strip()
            if not title:
                return "error: title is required"
            path = self.dir / (_slug(title) + ".md")
            path.write_text(f"# {title}\n\n{params.get('body', '')}\n", encoding="utf-8")
            return f"Saved note '{title}'."
        if action == "list":
            titles = []
            for p in sorted(self.dir.glob("*.md")):
                first = p.read_text(encoding="utf-8", errors="replace").splitlines()[:1]
                title = first[0].lstrip("# ").strip() if first else p.stem
                titles.append(title)
            return "\n".join(f"- {t}" for t in titles) or "(no notes)"
        if action == "read":
            title = str(params.get("title", "")).strip()
            path = self.dir / (_slug(title) + ".md")
            if not path.is_file():
                return f"error: no note '{title}'"
            return path.read_text(encoding="utf-8")
        if action == "delete":
            title = str(params.get("title", "")).strip()
            path = self.dir / (_slug(title) + ".md")
            if not path.is_file():
                return f"error: no note '{title}'"
            path.unlink()
            return f"Deleted note '{title}'."
        return "unsupported action"
