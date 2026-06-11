"""Google integrations: Gmail, Drive, Docs, Meet.

These use Google's REST APIs with an OAuth access token. Provide a token in
GOOGLE_ACCESS_TOKEN (from the OAuth playground or your own OAuth flow with the
right scopes). The real API calls below activate once the token is present.
"""

from typing import Any, Dict

from .. import http_util
from .base import Integration


class _GoogleBase(Integration):
    category = "Google"
    required_env = ["GOOGLE_ACCESS_TOKEN"]
    docs_url = "https://developers.google.com/identity/protocols/oauth2"

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.env('GOOGLE_ACCESS_TOKEN')}"}


class GmailIntegration(_GoogleBase):
    id = "gmail"
    name = "Gmail"

    def actions(self) -> Dict[str, str]:
        return {
            "list_unread": "List recent unread message subjects.",
            "send": "Send an email. params: to, subject, body.",
        }

    def call(self, action: str, params: Dict[str, Any]) -> str:
        if action == "list_unread":
            data = http_util.get_json(
                "https://gmail.googleapis.com/gmail/v1/users/me/messages?q=is:unread&maxResults=10",
                headers=self._headers(),
            )
            msgs = data.get("messages", [])
            out = []
            for m in msgs[:10]:
                meta = http_util.get_json(
                    f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{m['id']}"
                    "?format=metadata&metadataHeaders=Subject",
                    headers=self._headers(),
                )
                headers = meta.get("payload", {}).get("headers", [])
                subj = next((h["value"] for h in headers if h["name"] == "Subject"), "(no subject)")
                out.append(f"- {subj}")
            return "\n".join(out) or "(no unread mail)"
        if action == "send":
            import base64

            to = params.get("to")
            subject = params.get("subject", "")
            body = params.get("body", "")
            if not to:
                return "error: 'to' is required"
            raw = f"To: {to}\r\nSubject: {subject}\r\n\r\n{body}"
            encoded = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")
            http_util.post_json(
                "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
                {"raw": encoded},
                headers=self._headers(),
            )
            return f"Email sent to {to}."
        return "unsupported action"


class GoogleDriveIntegration(_GoogleBase):
    id = "gdrive"
    name = "Google Drive"

    def actions(self) -> Dict[str, str]:
        return {"list_files": "List recent files in your Drive."}

    def call(self, action: str, params: Dict[str, Any]) -> str:
        if action == "list_files":
            data = http_util.get_json(
                "https://www.googleapis.com/drive/v3/files?pageSize=20"
                "&fields=files(name,mimeType)",
                headers=self._headers(),
            )
            files = data.get("files", [])
            return "\n".join(f"- {f['name']}" for f in files) or "(no files)"
        return "unsupported action"


class GoogleDocsIntegration(_GoogleBase):
    id = "gdocs"
    name = "Google Docs"

    def actions(self) -> Dict[str, str]:
        return {"read": "Read a doc's text. params: document_id."}

    def call(self, action: str, params: Dict[str, Any]) -> str:
        if action == "read":
            doc_id = params.get("document_id")
            if not doc_id:
                return "error: document_id is required"
            data = http_util.get_json(
                f"https://docs.googleapis.com/v1/documents/{doc_id}",
                headers=self._headers(),
            )
            parts = []
            for el in data.get("body", {}).get("content", []):
                para = el.get("paragraph")
                if not para:
                    continue
                for run in para.get("elements", []):
                    txt = run.get("textRun", {}).get("content")
                    if txt:
                        parts.append(txt)
            text = "".join(parts).strip()
            return text[:4000] or "(empty doc)"
        return "unsupported action"


class GoogleMeetIntegration(_GoogleBase):
    id = "gmeet"
    name = "Google Meet"

    def actions(self) -> Dict[str, str]:
        return {"create": "Create a Meet link via Calendar. params: title, start, end (RFC3339)."}

    def call(self, action: str, params: Dict[str, Any]) -> str:
        if action == "create":
            title = params.get("title", "Meeting")
            start = params.get("start")
            end = params.get("end")
            if not start or not end:
                return "error: start and end (RFC3339 datetimes) are required"
            body = {
                "summary": title,
                "start": {"dateTime": start},
                "end": {"dateTime": end},
                "conferenceData": {
                    "createRequest": {"requestId": str(abs(hash(title + start)))}
                },
            }
            data = http_util.post_json(
                "https://www.googleapis.com/calendar/v3/calendars/primary/events"
                "?conferenceDataVersion=1",
                body,
                headers=self._headers(),
            )
            link = data.get("hangoutLink") or data.get("htmlLink")
            return f"Meet created: {link}"
        return "unsupported action"
