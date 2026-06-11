"""Date and time tool. Works fully offline."""

import datetime
from typing import Any, Dict

from .base import Tool


class DateTimeTool(Tool):
    name = "datetime"
    description = "Get the current local date and time."
    parameters = {}

    def run(self, args: Dict[str, Any]) -> str:
        now = datetime.datetime.now()
        return now.strftime("%A, %Y-%m-%d %H:%M:%S")
