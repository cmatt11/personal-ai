"""Image analysis tool.

analyze_image reads an image from the workspace, encodes it, and sends it to a
vision-capable model (local llava / llama3.2-vision via Ollama, or an online
model like gpt-4o). Returns a description or answer to your question about it.

Needs a vision model. Offline: `ollama pull llama3.2-vision`. Online: any
vision-capable model.
"""

import base64
from pathlib import Path
from typing import Any, Dict, Optional

from ..llm import LLMBackend
from .base import Tool

MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}
MAX_IMAGE_BYTES = 12_000_000


class AnalyzeImageTool(Tool):
    name = "analyze_image"
    description = (
        "Look at an image file in the workspace and describe or answer a "
        "question about it. Use for screenshots, photos, diagrams, receipts."
    )
    parameters = {
        "path": "Path to an image file relative to the workspace.",
        "question": "What to ask about the image (optional; defaults to 'describe it').",
    }

    def __init__(self, workspace: Path, vision_backend: Optional[LLMBackend]) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.vision_backend = vision_backend

    def _resolve(self, rel: str) -> Path:
        target = (self.workspace / rel).resolve()
        if self.workspace not in target.parents and target != self.workspace:
            raise ValueError("path escapes the workspace sandbox")
        return target

    def run(self, args: Dict[str, Any]) -> str:
        if self.vision_backend is None or not self.vision_backend.supports_vision:
            return (
                "No vision model available. Offline: run `ollama pull "
                "llama3.2-vision`. Online: set an API key with a vision model."
            )
        rel = str(args.get("path", "")).strip()
        if not rel:
            return "error: no image path provided"
        question = str(args.get("question", "")).strip() or "Describe this image in detail."
        try:
            target = self._resolve(rel)
            if not target.is_file():
                return f"error: no such image: {rel}"
            if target.stat().st_size > MAX_IMAGE_BYTES:
                return "error: image too large"
            mime = MIME_BY_EXT.get(target.suffix.lower(), "image/png")
            data = target.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            return self.vision_backend.vision(question, [b64], mime=mime)
        except Exception as exc:
            return f"error: {exc}"
