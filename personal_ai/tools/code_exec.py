"""Code execution tools.

run_python actually executes Python code and returns its output, so the
assistant can compute, test logic, and verify programs it writes instead of
guessing. Code runs in a separate process with a timeout, inside the local
workspace folder.

Security note: this runs real code on your machine. It is meant for a personal
assistant on a machine you control. Disable it with PERSONAL_AI_ALLOW_CODE=false
if you do not want it.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from .base import Tool

MAX_OUTPUT_CHARS = 8000


class RunPythonTool(Tool):
    name = "run_python"
    description = (
        "Execute Python 3 code and return its stdout and stderr. Use print() to "
        "show results. Good for math, data work, testing logic, and writing and "
        "running real programs. Runs in the local workspace."
    )
    parameters = {
        "code": "The Python source code to run.",
        "timeout": "Optional max seconds to run (default 15).",
    }

    def __init__(self, workspace: Path, default_timeout: int = 15) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.default_timeout = default_timeout

    def run(self, args: Dict[str, Any]) -> str:
        code = str(args.get("code", ""))
        if not code.strip():
            return "error: no code provided"
        try:
            timeout = int(args.get("timeout", self.default_timeout))
        except (TypeError, ValueError):
            timeout = self.default_timeout
        timeout = max(1, min(timeout, 60))

        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace),
            )
        except subprocess.TimeoutExpired:
            return f"error: code timed out after {timeout}s"
        except Exception as exc:
            return f"error: {exc}"

        parts = []
        if proc.stdout:
            parts.append("stdout:\n" + proc.stdout.rstrip())
        if proc.stderr:
            parts.append("stderr:\n" + proc.stderr.rstrip())
        if proc.returncode != 0:
            parts.append(f"exit code: {proc.returncode}")
        out = "\n".join(parts) if parts else "(no output; remember to print() results)"
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + "\n... [truncated]"
        return out
