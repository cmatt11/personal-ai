"""Microsoft integrations: Outlook, OneDrive, Word, Teams.

These use the Microsoft Graph API with an access token. Provide a token in
MS_GRAPH_TOKEN (from the Azure portal app registration / OAuth flow with the
right scopes). The real Graph calls activate once the token is present.
"""

from typing import Any, Dict

from .. import http_util
from .base import Integration

GRAPH = "https://graph.microsoft.com/v1.0"


class _MsBase(Integration):
    category = "Microsoft"
    required_env = ["MS_GRAPH_TOKEN"]
    docs_url = "https://learn.microsoft.com/graph/auth-v2-user"

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.env('MS_GRAPH_TOKEN')}"}


class OutlookIntegration(_MsBase):
    id = "outlook"
    name = "Outlook"

    def actions(self) -> Dict[str, str]:
        return {
            "list_messages": "List recent inbox message subjects.",
            "send": "Send an email. params: to, subject, body.",
        }

    def call(self, action: str, params: Dict[str, Any]) -> str:
        if action == "list_messages":
            data = http_util.get_json(
                f"{GRAPH}/me/messages?$top=10&$select=subject,from", headers=self._headers()
            )
            msgs = data.get("value", [])
            return "\n".join(f"- {m.get('subject')}" for m in msgs) or "(no messages)"
        if action == "send":
            to = params.get("to")
            if not to:
                return "error: 'to' is required"
            body = {
                "message": {
                    "subject": params.get("subject", ""),
                    "body": {"contentType": "Text", "content": params.get("body", "")},
                    "toRecipients": [{"emailAddress": {"address": to}}],
                }
            }
            http_util.post_json(f"{GRAPH}/me/sendMail", body, headers=self._headers())
            return f"Email sent to {to}."
        return "unsupported action"


class OneDriveIntegration(_MsBase):
    id = "onedrive"
    name = "OneDrive"

    def actions(self) -> Dict[str, str]:
        return {"list_files": "List files in your OneDrive root."}

    def call(self, action: str, params: Dict[str, Any]) -> str:
        if action == "list_files":
            data = http_util.get_json(
                f"{GRAPH}/me/drive/root/children?$select=name,folder", headers=self._headers()
            )
            items = data.get("value", [])
            return "\n".join(f"- {i['name']}" for i in items) or "(empty)"
        return "unsupported action"


class WordIntegration(_MsBase):
    id = "word"
    name = "Word"

    def actions(self) -> Dict[str, str]:
        return {"list_documents": "List Word (.docx) documents in OneDrive."}

    def call(self, action: str, params: Dict[str, Any]) -> str:
        if action == "list_documents":
            data = http_util.get_json(
                f"{GRAPH}/me/drive/root/search(q='.docx')?$select=name,webUrl",
                headers=self._headers(),
            )
            items = data.get("value", [])
            return "\n".join(f"- {i['name']}" for i in items) or "(no documents)"
        return "unsupported action"


class TeamsIntegration(_MsBase):
    id = "teams"
    name = "Teams"

    def actions(self) -> Dict[str, str]:
        return {"list_teams": "List the Teams you are a member of."}

    def call(self, action: str, params: Dict[str, Any]) -> str:
        if action == "list_teams":
            data = http_util.get_json(f"{GRAPH}/me/joinedTeams", headers=self._headers())
            items = data.get("value", [])
            return "\n".join(f"- {t['displayName']}" for t in items) or "(no teams)"
        return "unsupported action"
