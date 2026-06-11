"""Shell command tool.

run_shell executes a shell command and returns its output. This is the most
powerful "do stuff" tool: it can run git, build tools, scripts, system
commands, anything you could type in a terminal.

Security note: this runs real commands on your machine, inside the workspace
folder. It is meant for a personal assistant on a machine you control. It is
gated by PERSONAL_AI_ALLOW_SHELL and you can turn it off entirely.
"""

import subprocess
from pathlib import Path
from typing import Any, Dict

from ..platform_util import shell_command
from .base import Tool

MAX_OUTPUT_CHARS = 8000


class RunShellTool(Tool):
    name = "run_shell"
    description = (
        "Run a shell command and return stdout/stderr. Use for git, listing "
        "processes, running scripts, system tasks. Runs in the local workspace."
    )
    parameters = {
        "command": "The shell command to run.",
        "timeout": "Optional max seconds (default 30).",
    }

    def __init__(self, workspace: Path, default_timeout: int = 30) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.default_timeout = default_timeout

    def run(self, args: Dict[str, Any]) -> str:
        command = str(args.get("command", "")).strip()
        if not command:
            return "error: no command provided"
        try:
            timeout = int(args.get("timeout", self.default_timeout))
        except (TypeError, ValueError):
            timeout = self.default_timeout
        timeout = max(1, min(timeout, 120))

        try:
            proc = subprocess.run(
                shell_command(command),
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.workspace),
            )
        except subprocess.TimeoutExpired:
            return f"error: command timed out after {timeout}s"
        except Exception as exc:
            return f"error: {exc}"

        parts = []
        if proc.stdout:
            parts.append(proc.stdout.rstrip())
        if proc.stderr:
            parts.append("stderr:\n" + proc.stderr.rstrip())
        if proc.returncode != 0:
            parts.append(f"exit code: {proc.returncode}")
        out = "\n".join(parts) if parts else "(command produced no output)"
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + "\n... [truncated]"
        return out
