"""Text embeddings for the vector memory.

Three backends share one interface:
- OllamaEmbeddings: local embeddings via Ollama (offline).
- OpenAIEmbeddings: online embeddings via an OpenAI-compatible API.
- HashingEmbeddings: a pure-Python fallback so memory ALWAYS works, even with
  no model available. Lower quality, but keeps the system functional.

The selector mirrors the LLM selector: local first, then online, then the
hashing fallback as a last resort.
"""

import hashlib
import math
import re
from typing import List

from . import http_util
from .config import Config
from .connectivity import is_online, ollama_available

Vector = List[float]


class EmbeddingBackend:
    """Common interface for embedding backends."""

    name = "base"
    dim = 0

    def embed(self, texts: List[str]) -> List[Vector]:
        raise NotImplementedError

    def embed_one(self, text: str) -> Vector:
        return self.embed([text])[0]


class OllamaEmbeddings(EmbeddingBackend):
    """Local embeddings via Ollama."""

    def __init__(self, host: str, model: str, timeout: int = 60) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.name = f"local:ollama/{model}"

    def embed(self, texts: List[str]) -> List[Vector]:
        out: List[Vector] = []
        for text in texts:
            data = http_util.post_json(
                f"{self.host}/api/embeddings",
                {"model": self.model, "prompt": text},
                timeout=self.timeout,
            )
            vec = data.get("embedding")
            if not vec:
                raise RuntimeError(f"No embedding returned by Ollama: {data}")
            out.append([float(x) for x in vec])
        if out:
            self.dim = len(out[0])
        return out


class OpenAIEmbeddings(EmbeddingBackend):
    """Online embeddings via an OpenAI-compatible API."""

    def __init__(
        self, base_url: str, api_key: str, model: str, timeout: int = 60
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.name = f"online:{model}"

    def embed(self, texts: List[str]) -> List[Vector]:
        data = http_util.post_json(
            f"{self.base_url}/embeddings",
            {"model": self.model, "input": texts},
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=self.timeout,
        )
        items = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
        out = [[float(x) for x in item["embedding"]] for item in items]
        if out:
            self.dim = len(out[0])
        return out


class HashingEmbeddings(EmbeddingBackend):
    """Deterministic, dependency-free fallback embedding.

    Hashes word and character trigrams into a fixed-size vector (the hashing
    trick), then L2-normalizes. Captures rough lexical similarity so the vector
    memory remains useful when no embedding model is present.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"fallback:hashing/{dim}"

    def _features(self, text: str) -> List[str]:
        text = text.lower()
        words = re.findall(r"[a-z0-9]+", text)
        feats: List[str] = list(words)
        # Character trigrams add robustness to typos and morphology.
        joined = " ".join(words)
        for i in range(len(joined) - 2):
            feats.append("#" + joined[i : i + 3])
        return feats

    def embed(self, texts: List[str]) -> List[Vector]:
        out: List[Vector] = []
        for text in texts:
            vec = [0.0] * self.dim
            for feat in self._features(text):
                h = hashlib.md5(feat.encode("utf-8")).digest()
                idx = int.from_bytes(h[:4], "big") % self.dim
                sign = 1.0 if h[4] & 1 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            out.append(vec)
        return out


def select_embeddings(config: Config) -> EmbeddingBackend:
    """Choose an embedding backend: local, then online, then hashing fallback."""
    pref = config.backend

    if pref != "online" and ollama_available(config.ollama_host):
        try:
            backend = OllamaEmbeddings(config.ollama_host, config.local_embed_model)
            backend.embed(["warmup"])  # verify the embed model is pulled
            return backend
        except Exception:
            pass

    if pref != "local" and config.online_api_key and is_online():
        try:
            backend = OpenAIEmbeddings(
                config.online_base_url, config.online_api_key, config.online_embed_model
            )
            backend.embed(["warmup"])
            return backend
        except Exception:
            pass

    return HashingEmbeddings()


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Cosine similarity between two vectors (0 if either is empty/zero)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
