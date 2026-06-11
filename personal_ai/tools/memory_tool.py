"""Memory tools: store durable facts and recall them later.

These wrap the local vector memory so the assistant can remember things across
sessions and search its own past, fully offline.
"""

from typing import Any, Dict

from ..memory import Memory
from .base import Tool


class RememberTool(Tool):
    name = "remember"
    description = "Save an important fact or preference to long-term memory."
    parameters = {"text": "The fact to remember, written as a clear standalone statement."}

    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    def run(self, args: Dict[str, Any]) -> str:
        text = str(args.get("text", "")).strip()
        if not text:
            return "error: nothing to remember"
        self.memory.add_memory(text, kind="fact", source="tool")
        return f"Saved to memory: {text}"


class RecallTool(Tool):
    name = "recall"
    description = "Search long-term memory for facts relevant to a query."
    parameters = {"query": "What to look up in memory."}

    def __init__(self, memory: Memory, top_k: int = 5) -> None:
        self.memory = memory
        self.top_k = top_k

    def run(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return "error: no query provided"
        hits = self.memory.search_memory(query, top_k=self.top_k)
        if not hits:
            return "No relevant memories found."
        lines = [f"- ({score:.2f}) {text}" for score, text, _kind in hits]
        return "Relevant memories:\n" + "\n".join(lines)
