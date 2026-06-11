"""Proactive reminders for due tasks.

Surfaces open tasks that have a due date, flagging overdue and today's items.
Due dates are free text; ISO-like dates (YYYY-MM-DD) are parsed to detect
overdue. Everything works offline.
"""

import datetime
import re
from typing import Dict, List

from .memory import Memory

_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _parse_date(due: str):
    m = _ISO.search(due or "")
    if not m:
        return None
    try:
        return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def due_reminders(memory: Memory) -> List[str]:
    """Return reminder lines for open tasks that have a due date."""
    today = datetime.date.today()
    lines: List[str] = []
    for t in memory.list_tasks(include_done=False):
        due = t.get("due")
        if not due:
            continue
        d = _parse_date(str(due))
        if d is not None and d < today:
            flag = "OVERDUE"
        elif d is not None and d == today:
            flag = "TODAY"
        else:
            flag = "upcoming"
        lines.append(f"  [{flag}] #{t['id']} {t['text']} (due {due})")
    return lines


def reminder_banner(memory: Memory) -> str:
    lines = due_reminders(memory)
    if not lines:
        return ""
    return "Reminders:\n" + "\n".join(lines)
