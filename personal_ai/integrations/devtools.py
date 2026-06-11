"""Developer tool integrations: GitHub, VS Code, Vercel.

GitHub and Vercel work with a personal access token (no OAuth dance). VS Code
is a local integration that opens files/folders via the `code` CLI.
"""

import shutil
import subprocess
from typing import Any, Dict

from .. import http_util
from .base import STATUS_CONNECTED, STATUS_NEEDS_SETUP, Integration


class GitHubIntegration(Integration):
    id = "github"
    name = "GitHub"
    category = "Dev tools"
    required_env = ["GITHUB_TOKEN"]
    docs_url = "https://github.com/settings/tokens"

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"token {self.env('GITHUB_TOKEN')}",
            "Accept": "application/vnd.github+json",
        }

    def actions(self) -> Dict[str, str]:
        return {
            "whoami": "Show the authenticated GitHub user.",
            "list_repos": "List your repositories.",
            "list_issues": "List issues for a repo. params: repo='owner/name'.",
            "create_issue": "Create an issue. params: repo, title, body.",
        }

    def call(self, action: str, params: Dict[str, Any]) -> str:
        if action == "whoami":
            data = http_util.get_json("https://api.github.com/user", headers=self._headers())
            return f"{data.get('login')} ({data.get('name')}) - {data.get('html_url')}"
        if action == "list_repos":
            data = http_util.get_json(
                "https://api.github.com/user/repos?per_page=20&sort=updated",
                headers=self._headers(),
            )
            return "\n".join(f"- {r['full_name']}" for r in data) or "(no repos)"
        if action == "list_issues":
            repo = params.get("repo")
            if not repo:
                return "error: repo='owner/name' is required"
            data = http_util.get_json(
                f"https://api.github.com/repos/{repo}/issues?per_page=20",
                headers=self._headers(),
            )
            return "\n".join(f"#{i['number']} {i['title']}" for i in data) or "(no issues)"
        if action == "create_issue":
            repo = params.get("repo")
            title = params.get("title")
            if not repo or not title:
                return "error: repo and title are required"
            data = http_util.post_json(
                f"https://api.github.com/repos/{repo}/issues",
                {"title": title, "body": params.get("body", "")},
                headers=self._headers(),
            )
            return f"Created issue #{data.get('number')}: {data.get('html_url')}"
        return "unsupported action"


class VercelIntegration(Integration):
    id = "vercel"
    name = "Vercel"
    category = "Dev tools"
    required_env = ["VERCEL_TOKEN"]
    docs_url = "https://vercel.com/account/tokens"

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.env('VERCEL_TOKEN')}"}

    def actions(self) -> Dict[str, str]:
        return {
            "list_projects": "List your Vercel projects.",
            "list_deployments": "List recent deployments.",
        }

    def call(self, action: str, params: Dict[str, Any]) -> str:
        if action == "list_projects":
            data = http_util.get_json(
                "https://api.vercel.com/v9/projects?limit=20", headers=self._headers()
            )
            projects = data.get("projects", [])
            return "\n".join(f"- {p['name']}" for p in projects) or "(no projects)"
        if action == "list_deployments":
            data = http_util.get_json(
                "https://api.vercel.com/v6/deployments?limit=10", headers=self._headers()
            )
            deps = data.get("deployments", [])
            return "\n".join(f"- {d.get('name')} {d.get('url')}" for d in deps) or "(none)"
        return "unsupported action"


class VSCodeIntegration(Integration):
    id = "vscode"
    name = "VS Code"
    category = "Dev tools"
    required_env = []
    requires_internet = False
    docs_url = "https://code.visualstudio.com/docs/editor/command-line"

    def is_configured(self) -> bool:
        return shutil.which("code") is not None

    def status(self) -> str:
        return STATUS_CONNECTED if self.is_configured() else STATUS_NEEDS_SETUP

    def setup_hint(self) -> str:
        if self.is_configured():
            return ""
        return "install VS Code and enable the 'code' command in PATH"

    def actions(self) -> Dict[str, str]:
        return {"open": "Open a file or folder in VS Code. params: path."}

    def call(self, action: str, params: Dict[str, Any]) -> str:
        if action == "open":
            path = str(params.get("path", "")).strip()
            if not path:
                return "error: path is required"
            subprocess.Popen(["code", path])
            return f"Opened {path} in VS Code"
        return "unsupported action"
