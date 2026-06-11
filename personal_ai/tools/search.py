"""Search tools: find files by name (glob) and search file contents (grep).

Both are sandboxed to the workspace and work fully offline.
"""

import re
from pathlib import Path
from typing import Any, Dict

from .base import Tool

MAX_MATCHES = 100
MAX_FILE_BYTES = 2_000_000  # skip files larger than ~2 MB when grepping


class _Sandboxed:
    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)


class FindFilesTool(_Sandboxed, Tool):
    name = "find_files"
    description = "Find files in the workspace by glob pattern, e.g. '**/*.py'."
    parameters = {"pattern": "A glob pattern like '*.txt' or '**/*.py'."}

    def run(self, args: Dict[str, Any]) -> str:
        pattern = str(args.get("pattern", "")).strip() or "*"
        try:
            matches = []
            for p in self.workspace.glob(pattern):
                rel = p.relative_to(self.workspace)
                matches.append(str(rel) + ("/" if p.is_dir() else ""))
                if len(matches) >= MAX_MATCHES:
                    break
            if not matches:
                return f"no files match '{pattern}'"
            return "\n".join(sorted(matches))
        except Exception as exc:
            return f"error: {exc}"


class SearchFilesTool(_Sandboxed, Tool):
    name = "search_files"
    description = (
        "Search file contents in the workspace with a regular expression and "
        "return matching lines with their file and line number (like grep)."
    )
    parameters = {
        "query": "A regular expression to search for.",
        "glob": "Optional file glob to limit the search (default '**/*').",
    }

    def run(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return "error: no query provided"
        glob = str(args.get("glob", "") or "**/*").strip()
        try:
            regex = re.compile(query)
        except re.error as exc:
            return f"error: invalid regex: {exc}"

        results = []
        try:
            for p in self.workspace.glob(glob):
                if not p.is_file():
                    continue
                try:
                    if p.stat().st_size > MAX_FILE_BYTES:
                        continue
                    text = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
                rel = p.relative_to(self.workspace)
                for i, line in enumerate(text.splitlines(), start=1):
                    if regex.search(line):
                        snippet = line.strip()
                        if len(snippet) > 200:
                            snippet = snippet[:200] + " ..."
                        results.append(f"{rel}:{i}: {snippet}")
                        if len(results) >= MAX_MATCHES:
                            results.append("... [more matches truncated]")
                            return "\n".join(results)
        except Exception as exc:
            return f"error: {exc}"
        return "\n".join(results) if results else f"no matches for '{query}'"
