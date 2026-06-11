"""The agent loop.

This is the brain's control flow: think, optionally call a tool, observe the
result, repeat, until it produces a final answer.

It uses a model-agnostic JSON tool protocol so it works the same with a local
Ollama model or an online API. Before each turn it pulls relevant long-term
memories into context (retrieval-augmented generation).
"""

import json
import re
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from .config import Config
from .llm import LLMBackend, Message
from .memory import Memory
from .profile import ProfileExtractor
from .tools.base import ToolRegistry

SYSTEM_TEMPLATE = """You are a personal AI assistant that runs on the user's own machine.
You are helpful, concise, and honest. You can work offline or online.

You have access to tools. To use a tool, reply with ONLY a JSON object:
{{"tool": "<tool_name>", "args": {{ ... }}}}

When you have the final answer, reply with plain text (no JSON wrapper).

Rules:
- Use a tool only when it helps. For simple chat, answer directly in plain text.
- Use one tool per step. You will see its result, then continue.
- If a tool reports it is offline, adapt and answer with what you know.
- Save durable facts about the user with the "remember" tool when useful.
- Never invent tool results. Only act on what tools actually return.

Available tools:
{tools}
"""


def _format_tools(registry: ToolRegistry) -> str:
    lines = []
    for spec in registry.specs():
        params = spec["parameters"]
        if params:
            arg_str = ", ".join(f'"{k}": <{v}>' for k, v in params.items())
        else:
            arg_str = "(no arguments)"
        net = " [needs internet]" if spec["requires_internet"] else ""
        lines.append(f'- {spec["name"]}{net}: {spec["description"]}\n  args: {{{arg_str}}}')
    return "\n".join(lines)


def extract_json(text: str) -> Optional[dict]:
    """Pull the first balanced top-level JSON object out of model text.

    Local models sometimes wrap JSON in prose or code fences; this is lenient.
    """
    # Prefer fenced blocks if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = []
    if fence:
        candidates.append(fence.group(1))

    # Scan for balanced braces.
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    candidates.append(text[start : i + 1])

    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict) and ("tool" in obj or "answer" in obj):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def estimate_tokens(text: str) -> int:
    """Rough token estimate (about 4 characters per token for English)."""
    return max(1, len(text) // 4)


def trim_to_budget(messages: List[Message], char_budget: int) -> List[Message]:
    """Trim conversation history so the context fits within a character budget.

    System messages (prompt, profile, retrieved memory) are always kept. The
    oldest non-system turns are dropped first. A marker notes any trimming.
    """
    if char_budget <= 0:
        return messages
    total = sum(len(m.get("content", "")) for m in messages)
    if total <= char_budget:
        return messages

    system_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]
    system_chars = sum(len(m.get("content", "")) for m in system_msgs)
    remaining = max(0, char_budget - system_chars)

    # Keep the most recent turns that fit.
    kept_reversed: List[Message] = []
    used = 0
    dropped = 0
    for m in reversed(other_msgs):
        size = len(m.get("content", ""))
        if used + size <= remaining:
            kept_reversed.append(m)
            used += size
        else:
            dropped += 1
    kept = list(reversed(kept_reversed))

    result = list(system_msgs)
    if dropped:
        result.append(
            {
                "role": "system",
                "content": f"[note: {dropped} older message(s) trimmed to fit the context window]",
            }
        )
    result.extend(kept)
    return result


class Agent:
    """Runs the think-act-observe loop over an LLM, tools, and memory."""

    def __init__(
        self,
        llm: LLMBackend,
        registry: ToolRegistry,
        memory: Memory,
        config: Config,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.memory = memory
        self.config = config
        self.profile = ProfileExtractor(memory)

    def _system_prompt(self) -> str:
        return SYSTEM_TEMPLATE.format(tools=_format_tools(self.registry))

    def _retrieve_context(self, user_input: str) -> str:
        hits = self.memory.search_memory(user_input, top_k=self.config.memory_top_k)
        if not hits:
            return ""
        lines = [f"- {text}" for _score, text, _kind in hits]
        return "Relevant things you remember about the user:\n" + "\n".join(lines)

    def _call_model(
        self,
        messages: List[Message],
        on_token: Optional[Callable[[str], None]],
    ) -> Tuple[str, bool]:
        """Call the model. If on_token is set and the backend streams, stream
        prose answers token by token, but hold back tool-call JSON (which
        starts with '{') so the user never sees raw JSON.

        Returns (full_text, streamed_as_prose).
        """
        if on_token is None or not getattr(self.llm, "supports_streaming", False):
            return self.llm.chat(messages), False

        buffer: List[str] = []
        decided = False
        streaming_prose = False
        head = ""
        for chunk in self.llm.chat_stream(messages):
            buffer.append(chunk)
            if not decided:
                head += chunk
                stripped = head.lstrip()
                if stripped:
                    # JSON tool calls and code fences start with these.
                    streaming_prose = not (stripped[0] == "{" or stripped.startswith("```"))
                    decided = True
                    if streaming_prose:
                        on_token(stripped)
            elif streaming_prose:
                on_token(chunk)
        return "".join(buffer), streaming_prose

    def chat(
        self,
        session: str,
        user_input: str,
        on_step: Optional[Callable[[str, str], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Handle one user message and return the assistant's final answer.

        on_step(kind, detail) surfaces tool calls and observations.
        on_token(text) streams the final answer as it is generated.
        """
        self.memory.add_message(session, "user", user_input)

        # Automatically learn personal facts (name, job, likes, mood, ...).
        if self.config.auto_remember:
            try:
                learned = self.profile.observe(user_input)
                if learned and on_step:
                    summary = ", ".join(f"{k}={v}" for k, v in learned.items())
                    on_step("learned", summary)
            except Exception:
                pass

        # Build the working context: system prompt, profile, retrieved memory, history.
        messages: List[Message] = [{"role": "system", "content": self._system_prompt()}]
        profile_summary = self.memory.profile_summary()
        if profile_summary:
            messages.append(
                {
                    "role": "system",
                    "content": "What you know about the user (use it to personalize):\n"
                    + profile_summary,
                }
            )
        retrieved = self._retrieve_context(user_input)
        if retrieved:
            messages.append({"role": "system", "content": retrieved})
        history = self.memory.recent_messages(session, self.config.history_turns)
        messages.extend(history)
        # Keep the context within the model's budget.
        messages = trim_to_budget(messages, self.config.context_char_budget)

        final_answer = ""
        streamed_final = False
        for _step in range(self.config.max_steps):
            raw, streamed = self._call_model(messages, on_token)
            obj = extract_json(raw)

            # No structured output: treat the whole reply as the final answer.
            if obj is None:
                final_answer = raw.strip()
                streamed_final = streamed
                break

            if "answer" in obj:
                final_answer = str(obj["answer"]).strip()
                streamed_final = False
                break

            tool_name = str(obj.get("tool", "")).strip()
            tool_args = obj.get("args", {}) or {}
            if on_step:
                on_step("tool", f"{tool_name} {json.dumps(tool_args)}")

            if not self.registry.has(tool_name):
                observation = (
                    f"error: unknown tool '{tool_name}'. "
                    f"Available: {', '.join(t.name for t in self.registry.all())}"
                )
            else:
                try:
                    observation = self.registry.get(tool_name).run(tool_args)
                except Exception as exc:  # tools should not crash the loop
                    observation = f"error while running {tool_name}: {exc}"

            if on_step:
                on_step("observation", observation)

            # Record the model's action and the tool result, then loop.
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {"role": "user", "content": f"Tool result for {tool_name}:\n{observation}"}
            )
        else:
            final_answer = (
                final_answer
                or "I reached the step limit before finishing. Here is what I have so far."
            )

        # Deliver the final answer through on_token if it was not streamed live.
        if on_token and not streamed_final and final_answer:
            on_token(final_answer)

        self.memory.add_message(session, "assistant", final_answer)

        # Auto-remember the exchange so the assistant builds long-term memory.
        if self.config.auto_remember:
            try:
                self.memory.add_memory(
                    f"User said: {user_input}", kind="conversation", source=session
                )
            except Exception:
                pass

        return final_answer
