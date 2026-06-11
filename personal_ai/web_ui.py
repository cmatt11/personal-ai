"""A local web chat UI built only on the standard library (http.server).

Open it in any browser on your machine (or your phone on the same network) at
http://localhost:8000. Replies stream live. No external dependencies.

Run:  python -m personal_ai.web_ui   or   python run.py --web [port]
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from .agent import Agent
from .config import Config
from .embeddings import select_embeddings
from .integrations import build_integrations
from .llm import LLMError, select_backend
from .memory import Memory
from .plugins import load_plugins
from .reminders import due_reminders
from .tools import build_registry

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Personal AI</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, system-ui, sans-serif; margin: 0; height: 100vh;
         display: flex; flex-direction: column; background: #0f1115; color: #e6e6e6; }
  header { padding: 12px 16px; background: #171a21; font-weight: 600; border-bottom: 1px solid #262b36; }
  #log { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 80%; padding: 10px 14px; border-radius: 14px; white-space: pre-wrap; line-height: 1.4; }
  .you { align-self: flex-end; background: #2563eb; color: #fff; }
  .ai { align-self: flex-start; background: #1f242e; }
  .meta { align-self: flex-start; font-size: 12px; color: #8b93a7; }
  form { display: flex; gap: 8px; padding: 12px; background: #171a21; border-top: 1px solid #262b36; }
  input { flex: 1; padding: 12px; border-radius: 10px; border: 1px solid #2b3340; background: #0f1115; color: #e6e6e6; font-size: 16px; }
  button { padding: 12px 18px; border: 0; border-radius: 10px; background: #2563eb; color: #fff; font-size: 16px; cursor: pointer; }
  button:disabled { opacity: .5; }
</style>
</head>
<body>
  <header>Personal AI <span id="status" style="font-weight:400;font-size:12px;color:#8b93a7"></span></header>
  <div id="log"></div>
  <form id="f">
    <input id="m" autocomplete="off" placeholder="Message your AI..." autofocus>
    <button id="send">Send</button>
  </form>
<script>
const log = document.getElementById('log');
const form = document.getElementById('f');
const input = document.getElementById('m');
const sendBtn = document.getElementById('send');
const session = 'web-' + Math.random().toString(36).slice(2, 8);

fetch('/api/status').then(r => r.json()).then(s => {
  document.getElementById('status').textContent =
    `- ${s.network} - brain: ${s.brain} - ${s.tools} tools`;
  if (s.reminders && s.reminders.length) addMeta('Reminders:\\n' + s.reminders.join('\\n'));
});

function addMsg(cls, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  return d;
}
function addMeta(text) {
  const d = document.createElement('div');
  d.className = 'meta';
  d.textContent = text;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  addMsg('you', text);
  input.value = '';
  sendBtn.disabled = true;
  const bubble = addMsg('ai', '');
  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, session })
    });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const parts = buf.split('\\n\\n');
      buf = parts.pop();
      for (const part of parts) {
        if (!part.startsWith('data:')) continue;
        const evt = JSON.parse(part.slice(5).trim());
        if (evt.token) bubble.textContent += evt.token;
        else if (evt.step) addMeta(evt.step);
        else if (evt.error) bubble.textContent += '[error] ' + evt.error;
        log.scrollTop = log.scrollHeight;
      }
    }
  } catch (err) {
    bubble.textContent += '[connection error]';
  }
  sendBtn.disabled = false;
  input.focus();
});
</script>
</body>
</html>"""


class _App:
    """Holds the shared agent and serializes access (SQLite + ordering)."""

    def __init__(self, config: Config) -> None:
        self.config = config
        config.ensure_dirs()
        self.embedder = select_embeddings(config)
        self.memory = Memory(config.db_path, self.embedder, same_thread=False)
        self.llm = select_backend(config)
        self.registry = build_registry(config, self.memory, vision_backend=self.llm)
        load_plugins(self.registry, config, self.memory)
        self.integrations = build_integrations(config)
        self.agent = Agent(self.llm, self.registry, self.memory, config)
        self.lock = threading.Lock()


def _make_handler(app: _App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"))
            elif self.path == "/api/status":
                body = {
                    "brain": app.llm.name,
                    "network": "online" if __import__("personal_ai.connectivity", fromlist=["is_online"]).is_online() else "offline",
                    "tools": len(app.registry.all()),
                    "reminders": due_reminders(app.memory),
                }
                self._send(200, "application/json", json.dumps(body).encode("utf-8"))
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self):
            if self.path != "/api/chat":
                self._send(404, "text/plain", b"not found")
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._send(400, "text/plain", b"bad json")
                return
            message = str(payload.get("message", "")).strip()
            session = str(payload.get("session", "web")).strip() or "web"
            if not message:
                self._send(400, "text/plain", b"empty message")
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def emit(obj):
                try:
                    self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def on_token(t):
                emit({"token": t})

            def on_step(kind, detail):
                if kind == "tool":
                    emit({"step": f"using tool: {detail}"})
                elif kind == "learned":
                    emit({"step": f"learned about you: {detail}"})

            try:
                with app.lock:
                    app.agent.chat(session, message, on_step=on_step, on_token=on_token)
            except LLMError as exc:
                emit({"error": str(exc)})
            except Exception as exc:  # never crash the server
                emit({"error": str(exc)})
            emit({"done": True})

        def _send(self, status, ctype, body):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def serve(port: int = 8000, config: Optional[Config] = None) -> None:
    config = config or Config()
    try:
        app = _App(config)
    except LLMError as exc:
        print("Could not start the AI brain:\n")
        print(str(exc))
        return
    handler = _make_handler(app)
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    print(f"Personal AI web UI running at http://localhost:{port}")
    print(f"brain: {app.llm.name} | tools: {len(app.registry.all())}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
        server.shutdown()
        app.memory.close()


def main() -> int:
    import sys

    port = 8000
    for arg in sys.argv[1:]:
        if arg.isdigit():
            port = int(arg)
    serve(port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
