"""File tools, sandboxed to a single workspace directory.

The assistant can read, write, and list files only inside the configured
workspace. Paths are resolved and checked so it cannot escape the sandbox.
Works fully offline.
"""

from pathlib import Path
from typing import Any, Dict

from .base import Tool

MAX_READ_CHARS = 20000


class _Sandboxed:
    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _resolve(self, rel: str) -> Path:
        target = (self.workspace / rel).resolve()
        if self.workspace not in target.parents and target != self.workspace:
            raise ValueError("path escapes the workspace sandbox")
        return target


class ReadFileTool(_Sandboxed, Tool):
    name = "read_file"
    description = "Read a text file from the local workspace."
    parameters = {"path": "Path relative to the workspace."}

    def run(self, args: Dict[str, Any]) -> str:
        rel = str(args.get("path", "")).strip()
        if not rel:
            return "error: no path provided"
        try:
            target = self._resolve(rel)
            if not target.is_file():
                return f"error: no such file: {rel}"
            text = target.read_text(encoding="utf-8", errors="replace")
            if len(text) > MAX_READ_CHARS:
                text = text[:MAX_READ_CHARS] + "\n... [truncated]"
            return text
        except Exception as exc:
            return f"error: {exc}"


class WriteFileTool(_Sandboxed, Tool):
    name = "write_file"
    description = "Create or overwrite a text file in the local workspace."
    parameters = {
        "path": "Path relative to the workspace.",
        "content": "The full text content to write.",
    }

    def run(self, args: Dict[str, Any]) -> str:
        rel = str(args.get("path", "")).strip()
        content = str(args.get("content", ""))
        if not rel:
            return "error: no path provided"
        try:
            target = self._resolve(rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return f"Wrote {len(content)} chars to {rel}"
        except Exception as exc:
            return f"error: {exc}"


class ListFilesTool(_Sandboxed, Tool):
    name = "list_files"
    description = "List files and folders in the local workspace."
    parameters = {"path": "Optional subfolder relative to the workspace (default root)."}

    def run(self, args: Dict[str, Any]) -> str:
        rel = str(args.get("path", "") or ".").strip()
        try:
            target = self._resolve(rel)
            if not target.exists():
                return f"error: no such path: {rel}"
            if target.is_file():
                return rel
            entries = []
            for p in sorted(target.iterdir()):
                marker = "/" if p.is_dir() else ""
                entries.append(p.name + marker)
            return "\n".join(entries) if entries else "(empty)"
        except Exception as exc:
            return f"error: {exc}"


class DeleteFileTool(_Sandboxed, Tool):
    name = "delete_file"
    description = "Delete a file from the local workspace."
    parameters = {"path": "Path relative to the workspace."}

    def run(self, args: Dict[str, Any]) -> str:
        rel = str(args.get("path", "")).strip()
        if not rel:
            return "error: no path provided"
        try:
            target = self._resolve(rel)
            if not target.exists():
                return f"error: no such file: {rel}"
            if target.is_dir():
                return "error: refusing to delete a directory"
            target.unlink()
            return f"Deleted {rel}"
        except Exception as exc:
            return f"error: {exc}"
