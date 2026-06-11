"""Task and reminder tools, backed by local SQLite. Works offline.

Lets the assistant keep a to-do list for you: add tasks with an optional due
date, list open tasks, and mark them done.
"""

from typing import Any, Dict

from ..memory import Memory
from .base import Tool


class AddTaskTool(Tool):
    name = "add_task"
    description = "Add a task or reminder to the local to-do list."
    parameters = {
        "text": "What needs to be done.",
        "due": "Optional due date/time as text, e.g. '2026-06-15' or 'Friday 3pm'.",
    }

    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    def run(self, args: Dict[str, Any]) -> str:
        text = str(args.get("text", "")).strip()
        if not text:
            return "error: no task text provided"
        due = str(args.get("due", "")).strip() or None
        task_id = self.memory.add_task(text, due)
        due_str = f" (due {due})" if due else ""
        return f"Added task #{task_id}: {text}{due_str}"


class ListTasksTool(Tool):
    name = "list_tasks"
    description = "List open tasks. Set include_done=true to also show finished ones."
    parameters = {"include_done": "Optional 'true' to include completed tasks."}

    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    def run(self, args: Dict[str, Any]) -> str:
        include = str(args.get("include_done", "")).strip().lower() in ("1", "true", "yes")
        tasks = self.memory.list_tasks(include_done=include)
        if not tasks:
            return "No tasks."
        lines = []
        for t in tasks:
            box = "[x]" if t["done"] else "[ ]"
            due = f" (due {t['due']})" if t["due"] else ""
            lines.append(f"{box} #{t['id']} {t['text']}{due}")
        return "\n".join(lines)


class CompleteTaskTool(Tool):
    name = "complete_task"
    description = "Mark a task as done by its id."
    parameters = {"id": "The task id to complete."}

    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    def run(self, args: Dict[str, Any]) -> str:
        try:
            task_id = int(args.get("id"))
        except (TypeError, ValueError):
            return "error: a numeric task id is required"
        ok = self.memory.complete_task(task_id)
        return f"Completed task #{task_id}" if ok else f"No task with id #{task_id}"
