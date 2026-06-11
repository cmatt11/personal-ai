"""Tools to view and manage the auto-learned user profile."""

from typing import Any, Dict

from ..memory import Memory
from .base import Tool

VALID_CATEGORIES = {"name", "location", "job", "mood", "like", "dislike", "goal"}


class RememberAboutMeTool(Tool):
    name = "remember_about_me"
    description = (
        "Manually save a personal detail to the profile. category is one of: "
        "name, location, job, mood, like, dislike, goal."
    )
    parameters = {"category": "One of name/location/job/mood/like/dislike/goal.", "value": "The value."}

    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    def run(self, args: Dict[str, Any]) -> str:
        category = str(args.get("category", "")).strip().lower()
        value = str(args.get("value", "")).strip()
        if category not in VALID_CATEGORIES:
            return f"error: category must be one of {', '.join(sorted(VALID_CATEGORIES))}"
        if not value:
            return "error: value is required"
        self.memory.set_profile(category, value)
        return f"Saved to profile: {category} = {value}"


class ViewProfileTool(Tool):
    name = "view_profile"
    description = "Show everything the assistant has learned about the user."
    parameters = {}

    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    def run(self, args: Dict[str, Any]) -> str:
        rows = self.memory.list_profile()
        if not rows:
            return "No profile data yet."
        return "\n".join(f"#{r['id']} [{r['category']}] {r['value']}" for r in rows)


class ForgetProfileTool(Tool):
    name = "forget_about_me"
    description = "Delete a profile entry by its id (see view_profile for ids)."
    parameters = {"id": "The profile entry id to delete."}

    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    def run(self, args: Dict[str, Any]) -> str:
        try:
            pid = int(args.get("id"))
        except (TypeError, ValueError):
            return "error: a numeric profile id is required"
        ok = self.memory.delete_profile(pid)
        return f"Deleted profile entry #{pid}" if ok else f"No profile entry #{pid}"
