"""Social integrations: Twitter/X, Instagram, Discord.

Discord works with an incoming webhook URL (simple, no app review). Twitter/X
and Instagram use their official APIs, which require an approved developer app
and an access token; the real API calls are implemented and activate once you
provide a token. No scraping is used.
"""

from typing import Any, Dict

from .. import http_util
from .base import Integration


class DiscordIntegration(Integration):
    id = "discord"
    name = "Discord"
    category = "Social"
    required_env = ["DISCORD_WEBHOOK_URL"]
    docs_url = "https://support.discord.com/hc/en-us/articles/228383668"

    def actions(self) -> Dict[str, str]:
        return {"send_message": "Post a message to a channel via webhook. params: content."}

    def call(self, action: str, params: Dict[str, Any]) -> str:
        if action == "send_message":
            content = str(params.get("content", "")).strip()
            if not content:
                return "error: content is required"
            http_util.post_json(self.env("DISCORD_WEBHOOK_URL"), {"content": content})
            return "Message sent to Discord."
        return "unsupported action"


class TwitterIntegration(Integration):
    id = "twitter"
    name = "Twitter/X"
    category = "Social"
    required_env = ["TWITTER_BEARER_TOKEN"]
    docs_url = "https://developer.twitter.com/en/portal/dashboard"

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.env('TWITTER_BEARER_TOKEN')}"}

    def actions(self) -> Dict[str, str]:
        return {
            "get_user": "Look up a user by handle. params: username.",
            "recent_tweets": "Recent tweets for a username. params: username.",
        }

    def call(self, action: str, params: Dict[str, Any]) -> str:
        username = str(params.get("username", "")).strip().lstrip("@")
        if not username:
            return "error: username is required"
        user = http_util.get_json(
            f"https://api.twitter.com/2/users/by/username/{username}",
            headers=self._headers(),
        )
        data = user.get("data", {})
        if action == "get_user":
            return f"@{data.get('username')} id={data.get('id')} name={data.get('name')}"
        if action == "recent_tweets":
            uid = data.get("id")
            tweets = http_util.get_json(
                f"https://api.twitter.com/2/users/{uid}/tweets?max_results=5",
                headers=self._headers(),
            )
            items = tweets.get("data", [])
            return "\n".join(f"- {t.get('text')}" for t in items) or "(no tweets)"
        return "unsupported action"


class InstagramIntegration(Integration):
    id = "instagram"
    name = "Instagram"
    category = "Social"
    required_env = ["INSTAGRAM_ACCESS_TOKEN"]
    docs_url = "https://developers.facebook.com/docs/instagram-api"

    def actions(self) -> Dict[str, str]:
        return {
            "profile": "Show your business/creator account info.",
            "recent_media": "List your recent media.",
        }

    def call(self, action: str, params: Dict[str, Any]) -> str:
        token = self.env("INSTAGRAM_ACCESS_TOKEN")
        if action == "profile":
            data = http_util.get_json(
                f"https://graph.instagram.com/me?fields=id,username&access_token={token}"
            )
            return f"@{data.get('username')} id={data.get('id')}"
        if action == "recent_media":
            data = http_util.get_json(
                f"https://graph.instagram.com/me/media?fields=caption,permalink&access_token={token}"
            )
            items = data.get("data", [])
            return "\n".join(f"- {m.get('permalink')}" for m in items) or "(no media)"
        return "unsupported action"
