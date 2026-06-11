"""Tiny HTTP helpers built only on the standard library (urllib).

This keeps the whole project dependency-free. Used for talking to Ollama,
online LLM APIs, and for the web tools.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Iterator, Optional


# Some API gateways (for example Cloudflare in front of Groq) reject the
# default Python-urllib signature with a 403/1010 error. A normal User-Agent
# avoids that block.
USER_AGENT = "Mozilla/5.0 (compatible; personal-ai/1.0; +https://github.com/cmatt11/personal-ai)"


class HttpError(Exception):
    """Raised when an HTTP request fails."""

    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


def post_json(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    """POST a JSON body and parse a JSON response."""
    data = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            pass
        raise HttpError(f"HTTP {exc.code} from {url}: {detail}", status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise HttpError(f"Could not reach {url}: {exc.reason}") from exc


def get_text(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> str:
    """GET a URL and return the raw text body."""
    req_headers = {"User-Agent": USER_AGENT}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            return resp.read().decode(charset, errors="replace")
    except urllib.error.HTTPError as exc:
        raise HttpError(f"HTTP {exc.code} from {url}", status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise HttpError(f"Could not reach {url}: {exc.reason}") from exc


def get_json(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
) -> Dict[str, Any]:
    """GET a URL and parse a JSON response."""
    text = get_text(url, headers=headers, timeout=timeout)
    return json.loads(text) if text else {}


def post_stream(
    url: str,
    payload: Dict[str, Any],
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 300,
) -> Iterator[str]:
    """POST a JSON body and yield response lines as they arrive (for streaming)."""
    data = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line:
                    yield line
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            pass
        raise HttpError(f"HTTP {exc.code} from {url}: {detail}", status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise HttpError(f"Could not reach {url}: {exc.reason}") from exc
