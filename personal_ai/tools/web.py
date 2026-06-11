"""Web tools: search and page fetch.

These need the internet. When offline they return a clear message instead of
crashing, so the assistant degrades gracefully and keeps working with local
tools and memory.
"""

import html
import json
import re
import urllib.parse
from typing import Any, Dict

from .. import http_util
from ..connectivity import is_online
from .base import Tool


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web for current information (needs internet)."
    parameters = {"query": "The search query."}
    requires_internet = True

    def run(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return "error: no query provided"
        if not is_online():
            return "offline: web search is unavailable without an internet connection."
        # DuckDuckGo Instant Answer API (no key required).
        url = (
            "https://api.duckduckgo.com/?q="
            + urllib.parse.quote(query)
            + "&format=json&no_redirect=1&no_html=1"
        )
        try:
            data = http_util.get_json(url, timeout=15)
        except http_util.HttpError as exc:
            return f"error: {exc}"

        parts = []
        if data.get("AbstractText"):
            parts.append(data["AbstractText"])
        for topic in data.get("RelatedTopics", [])[:5]:
            if isinstance(topic, dict) and topic.get("Text"):
                parts.append("- " + topic["Text"])
        if not parts:
            return f"No instant answer found for '{query}'. Try web_fetch on a known URL."
        return "\n".join(parts)


class WebFetchTool(Tool):
    name = "web_fetch"
    description = "Fetch a web page and return its readable text (needs internet)."
    parameters = {"url": "The full URL to fetch (https preferred)."}
    requires_internet = True

    def run(self, args: Dict[str, Any]) -> str:
        url = str(args.get("url", "")).strip()
        if not url:
            return "error: no url provided"
        if url.startswith("http://"):
            url = "https://" + url[len("http://") :]
        if not url.startswith("https://"):
            url = "https://" + url
        if not is_online():
            return "offline: web fetch is unavailable without an internet connection."
        try:
            raw = http_util.get_text(url, timeout=20)
        except http_util.HttpError as exc:
            return f"error: {exc}"
        text = _strip_html(raw)
        if len(text) > 6000:
            text = text[:6000] + " ... [truncated]"
        return text or "(no readable text found)"
