"""Document ingestion tool.

ingest_file reads a text document from the workspace, splits it into chunks,
embeds each chunk, and stores them in long-term memory. After that, the
assistant can answer questions about the document using its normal memory
recall (retrieval-augmented generation over your own files). Works offline.
"""

from pathlib import Path
from typing import Any, Dict, List

from ..memory import Memory
from .base import Tool

CHUNK_CHARS = 800
CHUNK_OVERLAP = 100
MAX_DOC_CHARS = 500_000


def _chunk(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if overlap >= size:
        overlap = size // 4
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        # Try to break on a paragraph or sentence boundary for cleaner chunks.
        if end < n:
            window = text[start:end]
            for sep in ("\n\n", "\n", ". "):
                idx = window.rfind(sep)
                if idx > size // 2:
                    end = start + idx + len(sep)
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        # Advance with overlap, always making forward progress.
        start = max(end - overlap, start + 1)
    return chunks


class IngestFileTool(Tool):
    name = "ingest_file"
    description = (
        "Read a text document from the workspace and store it in long-term "
        "memory so you can answer questions about it later. Use 'recall' to "
        "retrieve from it afterward."
    )
    parameters = {"path": "Path to a text file relative to the workspace."}

    def __init__(self, workspace: Path, memory: Memory) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.memory = memory

    def _resolve(self, rel: str) -> Path:
        target = (self.workspace / rel).resolve()
        if self.workspace not in target.parents and target != self.workspace:
            raise ValueError("path escapes the workspace sandbox")
        return target

    def run(self, args: Dict[str, Any]) -> str:
        rel = str(args.get("path", "")).strip()
        if not rel:
            return "error: no path provided"
        try:
            target = self._resolve(rel)
            if not target.is_file():
                return f"error: no such file: {rel}"
            text = target.read_text(encoding="utf-8", errors="replace")
            if len(text) > MAX_DOC_CHARS:
                text = text[:MAX_DOC_CHARS]
            chunks = _chunk(text)
            if not chunks:
                return "error: file is empty"
            for chunk in chunks:
                self.memory.add_memory(chunk, kind="document", source=rel)
            return (
                f"Ingested '{rel}' into memory as {len(chunks)} chunks. "
                "Ask me about it and I will recall the relevant parts."
            )
        except Exception as exc:
            return f"error: {exc}"
