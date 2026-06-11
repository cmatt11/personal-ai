"""LLM backends and selection.

Three backends share one interface:
- OllamaBackend: a local model served by Ollama (works fully offline).
- OpenAIBackend: any OpenAI-compatible chat API (works online).
- MockBackend: scripted responses, used for tests and demos.

The selector prefers the local model so the assistant runs offline by default,
and falls back to the online backend when local is unavailable.
"""

import json
from typing import Callable, Dict, Iterator, List, Optional

from . import http_util
from .config import Config
from .connectivity import is_online, ollama_available

Message = Dict[str, str]  # {"role": "user"|"assistant"|"system", "content": "..."}


class LLMError(Exception):
    """Raised when no usable LLM backend can be found or a call fails."""


class LLMBackend:
    """Common interface for all chat backends."""

    name = "base"
    online = False
    supports_vision = False
    supports_streaming = False

    def chat(self, messages: List[Message], temperature: float = 0.7) -> str:
        raise NotImplementedError

    def chat_stream(
        self, messages: List[Message], temperature: float = 0.7
    ) -> Iterator[str]:
        """Yield text chunks as they are generated. Default: one chunk."""
        yield self.chat(messages, temperature)

    def vision(self, prompt: str, images_b64: List[str], mime: str = "image/png") -> str:
        """Analyze one or more images (base64-encoded) given a text prompt."""
        raise NotImplementedError("this backend does not support image analysis")


class OllamaBackend(LLMBackend):
    """Local model served by Ollama (https://ollama.com)."""

    online = False

    def __init__(
        self, host: str, model: str, timeout: int = 120, vision_model: str = ""
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.vision_model = vision_model or model
        self.timeout = timeout
        self.supports_vision = bool(vision_model)
        self.supports_streaming = True
        self.name = f"local:ollama/{model}"

    def chat(self, messages: List[Message], temperature: float = 0.7) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            data = http_util.post_json(
                f"{self.host}/api/chat", payload, timeout=self.timeout
            )
        except http_util.HttpError as exc:
            raise LLMError(str(exc)) from exc
        msg = data.get("message") or {}
        content = msg.get("content")
        if content is None:
            raise LLMError(f"Unexpected Ollama response: {data}")
        return content.strip()

    def chat_stream(
        self, messages: List[Message], temperature: float = 0.7
    ) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": temperature},
        }
        try:
            for line in http_util.post_stream(
                f"{self.host}/api/chat", payload, timeout=self.timeout
            ):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                chunk = (obj.get("message") or {}).get("content")
                if chunk:
                    yield chunk
                if obj.get("done"):
                    break
        except http_util.HttpError as exc:
            raise LLMError(str(exc)) from exc

    def vision(self, prompt: str, images_b64: List[str], mime: str = "image/png") -> str:
        # Ollama takes raw base64 strings in the message's "images" field.
        payload = {
            "model": self.vision_model,
            "messages": [{"role": "user", "content": prompt, "images": images_b64}],
            "stream": False,
        }
        try:
            data = http_util.post_json(
                f"{self.host}/api/chat", payload, timeout=self.timeout
            )
        except http_util.HttpError as exc:
            raise LLMError(str(exc)) from exc
        return ((data.get("message") or {}).get("content") or "").strip()


class OpenAIBackend(LLMBackend):
    """Any OpenAI-compatible chat completions API."""

    online = True

    def __init__(
        self, base_url: str, api_key: str, model: str, timeout: int = 120,
        vision_model: str = "",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.vision_model = vision_model or model
        self.timeout = timeout
        self.supports_vision = True
        self.supports_streaming = True
        self.name = f"online:{model}"

    def chat(self, messages: List[Message], temperature: float = 0.7) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            data = http_util.post_json(
                f"{self.base_url}/chat/completions",
                payload,
                headers=headers,
                timeout=self.timeout,
            )
        except http_util.HttpError as exc:
            raise LLMError(str(exc)) from exc
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"Unexpected API response: {data}")
        return (choices[0].get("message") or {}).get("content", "").strip()

    def chat_stream(
        self, messages: List[Message], temperature: float = 0.7
    ) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            for line in http_util.post_stream(
                f"{self.base_url}/chat/completions",
                payload,
                headers=headers,
                timeout=self.timeout,
            ):
                # Server-sent events: lines look like "data: {json}".
                if not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]":
                    break
                try:
                    obj = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0].get("delta") or {}).get("content")
                if delta:
                    yield delta
        except http_util.HttpError as exc:
            raise LLMError(str(exc)) from exc

    def vision(self, prompt: str, images_b64: List[str], mime: str = "image/png") -> str:
        content = [{"type": "text", "text": prompt}]
        for b64 in images_b64:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                }
            )
        payload = {
            "model": self.vision_model,
            "messages": [{"role": "user", "content": content}],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            data = http_util.post_json(
                f"{self.base_url}/chat/completions",
                payload,
                headers=headers,
                timeout=self.timeout,
            )
        except http_util.HttpError as exc:
            raise LLMError(str(exc)) from exc
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"Unexpected API response: {data}")
        return (choices[0].get("message") or {}).get("content", "").strip()


class MockBackend(LLMBackend):
    """Scripted backend for tests and offline demos.

    Accepts a callable that maps the message list to a string response, or a
    fixed list of responses returned in order.
    """

    online = False
    name = "mock"

    def __init__(
        self,
        responder: Optional[Callable[[List[Message]], str]] = None,
        scripted: Optional[List[str]] = None,
    ) -> None:
        self.responder = responder
        self.scripted = list(scripted or [])
        self._i = 0
        self.supports_streaming = True

    def chat(self, messages: List[Message], temperature: float = 0.7) -> str:
        if self.responder is not None:
            return self.responder(messages)
        if self._i < len(self.scripted):
            out = self.scripted[self._i]
            self._i += 1
            return out
        return '{"answer": "I have no scripted reply left."}'

    def chat_stream(
        self, messages: List[Message], temperature: float = 0.7
    ) -> Iterator[str]:
        text = self.chat(messages, temperature)
        # Emit word by word to simulate token streaming.
        parts = text.split(" ")
        for i, word in enumerate(parts):
            yield word if i == 0 else " " + word

    def vision(self, prompt: str, images_b64: List[str], mime: str = "image/png") -> str:
        return f"[mock vision] described {len(images_b64)} image(s) for prompt: {prompt}"


class HybridBackend(LLMBackend):
    """Online-first, local-fallback brain that auto-switches per message.

    When the internet is reachable it uses the online backend (faster, smarter).
    When offline, it falls back to the local backend so the assistant keeps
    working. The choice is made fresh on every call, so it adapts as your
    connection comes and goes.
    """

    def __init__(
        self,
        online_backend: Optional[LLMBackend],
        local_backend: Optional[LLMBackend],
        online_check: Callable[[], bool] = is_online,
    ) -> None:
        self.online_backend = online_backend
        self.local_backend = local_backend
        self._online_check = online_check
        self.online = bool(online_backend)
        self.supports_streaming = bool(
            (online_backend and online_backend.supports_streaming)
            or (local_backend and local_backend.supports_streaming)
        )
        self.supports_vision = bool(
            (online_backend and online_backend.supports_vision)
            or (local_backend and local_backend.supports_vision)
        )
        on = online_backend.name if online_backend else "none"
        loc = local_backend.name if local_backend else "none"
        self.name = f"hybrid[online={on} | offline={loc}]"

    def _pick(self) -> LLMBackend:
        """Pick the backend to use right now."""
        if self.online_backend is not None and self._online_check():
            return self.online_backend
        if self.local_backend is not None:
            return self.local_backend
        if self.online_backend is not None:
            # Online configured but we appear offline; try it anyway (it will
            # surface a clear error if it truly cannot connect).
            return self.online_backend
        raise LLMError("Hybrid backend has no usable brain configured.")

    def chat(self, messages: List[Message], temperature: float = 0.7) -> str:
        return self._pick().chat(messages, temperature)

    def chat_stream(
        self, messages: List[Message], temperature: float = 0.7
    ) -> Iterator[str]:
        backend = self._pick()
        if backend.supports_streaming:
            yield from backend.chat_stream(messages, temperature)
        else:
            yield backend.chat(messages, temperature)

    def vision(self, prompt: str, images_b64: List[str], mime: str = "image/png") -> str:
        # Prefer whichever picked backend can actually see.
        backend = self._pick()
        if not backend.supports_vision:
            other = (
                self.online_backend
                if backend is self.local_backend
                else self.local_backend
            )
            if other is not None and other.supports_vision:
                backend = other
        return backend.vision(prompt, images_b64, mime=mime)


def select_backend(config: Config) -> LLMBackend:
    """Choose a backend according to config and current connectivity."""
    pref = config.backend
    local_ready = ollama_available(config.ollama_host)
    online_ready = bool(config.online_api_key) and is_online()

    def make_local() -> LLMBackend:
        return OllamaBackend(
            config.ollama_host,
            config.local_model,
            timeout=config.request_timeout,
            vision_model=config.local_vision_model,
        )

    def make_online() -> LLMBackend:
        return OpenAIBackend(
            config.online_base_url,
            config.online_api_key,
            config.online_model,
            timeout=config.request_timeout,
            vision_model=config.online_vision_model,
        )

    if pref == "local":
        if not local_ready:
            raise LLMError(
                "Local backend requested but Ollama is not reachable at "
                f"{config.ollama_host}. Start it with `ollama serve` and pull a "
                f"model: `ollama pull {config.local_model}`."
            )
        return make_local()

    if pref == "online":
        if not config.online_api_key:
            raise LLMError(
                "Online backend requested but no API key set "
                "(PERSONAL_AI_ONLINE_API_KEY or OPENAI_API_KEY)."
            )
        if not is_online():
            raise LLMError("Online backend requested but the machine appears offline.")
        return make_online()

    if pref == "hybrid":
        online_b = make_online() if config.online_api_key else None
        local_b = make_local() if local_ready else None
        if online_b is None and local_b is None:
            raise LLMError(
                "Hybrid mode needs at least one brain.\n"
                "- Online: set PERSONAL_AI_ONLINE_API_KEY (or OPENAI_API_KEY).\n"
                f"- Offline: run Ollama and `ollama pull {config.local_model}`."
            )
        return HybridBackend(online_b, local_b)

    # auto: local first (offline-friendly), then online.
    if local_ready:
        return make_local()
    if online_ready:
        return make_online()

    raise LLMError(
        "No LLM backend available.\n"
        f"- Local: start Ollama (`ollama serve`) and `ollama pull {config.local_model}`.\n"
        "- Online: set PERSONAL_AI_ONLINE_API_KEY (or OPENAI_API_KEY) and connect to the internet."
    )
