"""Automatic memory: learn facts about the user from what they say.

After each message, this scans for personal details (name, job, location, mood,
likes, dislikes, goals) and stores them in the profile. The profile is then
injected into every reply so the assistant feels personal.

Extraction is rule-based so it works fully offline and deterministically. When
an LLM is available it can be used for richer extraction, but the rules cover
the common cases reliably.
"""

import re
from typing import Dict, List, Tuple

from .memory import Memory

# Each rule: (category, compiled regex). The first capture group is the value.
_RULES: List[Tuple[str, "re.Pattern[str]"]] = [
    ("name", re.compile(r"\bmy name is\s+([A-Za-z][\w'\-]+)", re.I)),
    ("name", re.compile(r"\b(?:call me|i am called)\s+([A-Za-z][\w'\-]+)", re.I)),
    ("location", re.compile(r"\bi(?:'m| am)?\s*(?:live|living) in\s+([A-Za-z][\w .,'\-]+)", re.I)),
    ("location", re.compile(r"\bi(?:'m| am)\s+from\s+([A-Za-z][\w .,'\-]+)", re.I)),
    ("location", re.compile(r"\bi(?:'m| am)\s+based in\s+([A-Za-z][\w .,'\-]+)", re.I)),
    ("job", re.compile(r"\bi\s+work\s+as\s+(?:an?\s+)?([A-Za-z][\w .'\-]+)", re.I)),
    ("job", re.compile(r"\bi(?:'m| am)\s+(?:an?\s+)([A-Za-z][\w .'\-]*(?:developer|engineer|designer|teacher|doctor|nurse|writer|manager|student|artist|founder|lawyer|accountant|analyst|consultant|chef|driver)\b)", re.I)),
    ("job", re.compile(r"\bmy job is\s+(?:an?\s+)?([A-Za-z][\w .'\-]+)", re.I)),
    ("job", re.compile(r"\bi\s+work\s+at\s+([A-Za-z][\w .'\-]+)", re.I)),
    ("like", re.compile(r"\bi\s+(?:really\s+)?(?:like|love|enjoy|prefer)\s+([^.,;!?\n]+)", re.I)),
    ("dislike", re.compile(r"\bi\s+(?:dislike|hate|can't stand|cannot stand|don'?t like|do not like)\s+([^.,;!?\n]+)", re.I)),
    ("goal", re.compile(r"\bmy goal is\s+(?:to\s+)?([^.,;!?\n]+)", re.I)),
    ("goal", re.compile(r"\bi\s+want\s+to\s+([^.,;!?\n]+)", re.I)),
    ("goal", re.compile(r"\bi(?:'m| am)\s+trying\s+to\s+([^.,;!?\n]+)", re.I)),
    ("mood", re.compile(r"\bi(?:'m| am)?\s*feeling\s+([A-Za-z][\w\- ]{1,30})", re.I)),
    ("mood", re.compile(r"\bi(?:'m| am)\s+(happy|sad|tired|exhausted|excited|stressed|anxious|angry|frustrated|relaxed|motivated|bored|sick|great|good|okay|fine)\b", re.I)),
]

# Words that are not real names/values to avoid bad captures.
_STOPVALUES = {"not", "so", "very", "really", "just", "too", "the", "a", "an"}

MAX_VALUE_LEN = 80


def _clean(value: str) -> str:
    value = value.strip().strip(".,;:!?\"'")
    value = re.sub(r"\s+", " ", value)
    # Cut run-on sentences at common connectors so values stay focused.
    value = re.split(r"\s+(?:and|but|because|so|though|although|while)\s+", value, maxsplit=1, flags=re.I)[0]
    # Drop trailing time qualifiers (mostly for mood).
    value = re.sub(r"\s+(today|right now|lately|currently|at the moment|these days)$", "", value, flags=re.I)
    value = value.strip()
    if len(value) > MAX_VALUE_LEN:
        value = value[:MAX_VALUE_LEN].rsplit(" ", 1)[0]
    return value


class ProfileExtractor:
    """Pulls personal facts out of text and stores them in the profile."""

    def __init__(self, memory: Memory) -> None:
        self.memory = memory

    def observe(self, text: str) -> Dict[str, str]:
        """Scan one user message, store any found facts, return what was learned."""
        learned: Dict[str, str] = {}
        for category, pattern in _RULES:
            m = pattern.search(text)
            if not m:
                continue
            value = _clean(m.group(1))
            low = value.lower()
            if not value or low in _STOPVALUES or len(low) < 2:
                continue
            self.memory.set_profile(category, value)
            learned[category] = value
        return learned
