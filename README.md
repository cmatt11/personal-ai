# Personal AI

A local-first, offline-capable personal AI assistant. It runs on your own machine, stores everything locally, uses the internet when it is available, and keeps working when it is not.

It has the pieces of a real AI system:

- A reasoning brain (a local or online large language model).
- A think, act, observe agent loop with tools.
- Long-term memory with semantic recall (retrieval-augmented generation).
- An automatic profile that learns about you and personalizes every reply.
- Streaming replies that appear as they generate.
- A local web UI you can open in a browser or on your phone.
- Image analysis with a vision model.
- An integrations panel for Google, Microsoft, social, and dev-tool services.
- A plugin folder for adding your own tools, and optional encryption at rest.
- Self-reprogramming: it can write, load, and remove its own tools on request.
- Local storage so your data never leaves your machine.

Runs on Windows, macOS, Linux, and Android (via Termux). Works online and offline.

## What it does

- Runs locally with [Ollama](https://ollama.com) so it works with no internet.
- Falls back to an online model (any OpenAI-compatible API) when you want more power.
- Stores conversations and facts locally in a single SQLite file.
- Remembers things across sessions and recalls them by meaning, not just keywords.
- Uses tools that actually do things:
  - **Math**: a calculator with arithmetic plus sqrt, trig, logs, factorial, and constants (pi, e).
  - **Programming**: a `run_python` tool that really executes code and returns the output, so it can compute and test, not guess.
  - **Shell**: a `run_shell` tool to run real commands (git, scripts, system tasks) in the workspace.
  - **Files**: read, write, list, and delete files inside a sandboxed workspace.
  - **Search**: find files by name (glob) and search file contents (grep).
  - **Tasks**: keep a to-do list with due dates; add, list, and complete tasks.
  - **Documents**: ingest your own text files into memory and answer questions about them.
  - **Images**: `analyze_image` looks at screenshots, photos, and diagrams with a vision model.
  - **Memory**: remember and recall facts across sessions.
  - **Profile**: automatically learns your name, job, location, mood, likes, dislikes, and goals from what you say, and uses them in every reply. View and edit it anytime.
  - **Integrations**: a panel for Google (Gmail, Drive, Docs, Meet), Microsoft (Outlook, OneDrive, Word, Teams), social (Twitter/X, Instagram, Discord), dev tools (GitHub, VS Code, Vercel), and local Notes.
  - **Web**: search and fetch pages when you are online.
- Web tools activate when you are online and degrade gracefully when you are offline.
- Has zero external Python dependencies. The whole thing runs on the standard library.

## How it works

```
You type a message
        |
        v
  [ Agent loop ]  <-- pulls relevant memories from local storage (RAG)
        |
   think with the LLM (local Ollama or online API)
        |
   need a tool?  --yes-->  run tool (calc / files / web / memory)  --> observe result --> loop
        |
        no
        v
   final answer  -->  saved to local SQLite memory
```

- **Brain**: a local model through Ollama by default, so it is private and offline. If you set an online API key, it can use that when local is unavailable.
- **Memory**: every message is stored locally. Important facts get an embedding vector so the assistant can find them later by meaning. Search uses cosine similarity computed in pure Python.
- **Tools**: the model requests a tool by emitting a small JSON object. The loop runs the tool and feeds the result back. This is the same agent pattern used by production AI systems.
- **Offline vs online**: the brain and memory work offline. The web tools need the internet and say so clearly when it is missing, instead of failing.

## Setup

### 1. Get the code

```bash
git clone <your-repo-url> personal-ai
cd personal-ai
```

You need Python 3.9 or newer. There are no packages to install.

### 2. Install a local brain (recommended, for offline use)

Install Ollama from https://ollama.com, then pull a chat model and an embedding model:

```bash
ollama pull llama3.2          # the reasoning model (try qwen2.5 or mistral too)
ollama pull nomic-embed-text  # for memory embeddings
ollama serve                  # start the local server (if not already running)
```

### 3. Run it

```bash
python run.py
# or
python -m personal_ai
```

You will see a status line showing your network state, which brain is active, and how many facts are stored. Then just chat.

### Web UI

Prefer a browser or your phone? Start the local web interface:

```bash
python run.py --web          # serves http://localhost:8000
python run.py --web 9000     # custom port
```

Open the address in any browser on the same machine or network. Replies stream live. Built on the standard library, no extra packages.

### Check your setup

```bash
python run.py --doctor       # reports what is ready and what is missing
python run.py --setup        # guided first-time setup, writes a .env file
```

## Optional: use an online model

If you want to use a hosted model when you have internet, set an API key:

```bash
export PERSONAL_AI_ONLINE_API_KEY=sk-your-key
export PERSONAL_AI_BACKEND=auto      # local first, online as fallback
python run.py
```

Any OpenAI-compatible endpoint works. Point `PERSONAL_AI_ONLINE_BASE_URL` at it and set the model name. See `.env.example` for all options.

### No model at all?

Memory and tools still load. The assistant needs at least one brain (local Ollama or an online key) to chat. Even without an embedding model, memory falls back to a built-in pure-Python embedding so recall keeps working at lower quality.

## Commands

Inside the chat:

| Command | What it does |
|---|---|
| `/help` | list commands |
| `/status` | show platform, network, brain, embedding backend, memory count, tools |
| `/doctor` | check environment readiness |
| `/integrations` | show the integrations panel and connection status |
| `/profile` | show what the assistant has learned about you |
| `/remember <text>` | save a fact to long-term memory |
| `/recall <query>` | search long-term memory |
| `/consolidate` | clean up memory (dedupe + prune) |
| `/forget` | wipe all long-term memory (asks to confirm) |
| `/tasks` | list open tasks and reminders |
| `/new <title>` | start a new named conversation |
| `/sessions` | list saved conversations |
| `/switch <id>` | switch to a saved conversation |
| `/export <path>` | back up all data to a JSON file |
| `/import <path>` | restore data from a JSON file |
| `/history` | show recent messages this session |
| `/verbose` | toggle showing tool calls and results |
| `/quit` | exit |

## Configuration

All settings are environment variables with defaults. Copy `.env.example` to `.env` and adjust. Highlights:

- `PERSONAL_AI_HOME`: where data is stored (default `~/.personal_ai`).
- `PERSONAL_AI_BACKEND`: `auto`, `local`, or `online`.
- `PERSONAL_AI_LOCAL_MODEL`: the Ollama model name.
- `PERSONAL_AI_ALLOW_WEB`: turn the web tools on or off.

## Platforms

Runs anywhere Python 3.9+ runs:

- **Windows**: `python run.py` in PowerShell or Command Prompt. The shell tool uses `cmd.exe`.
- **macOS / Linux**: `python run.py`.
- **Android**: install [Termux](https://termux.dev), then `pkg install python`, and run it there. Ollama runs locally on capable devices, or point the assistant at an online model.

Paths and the shell adapt to the OS automatically.

## Image analysis

The `analyze_image` tool sends an image to a vision model and returns a description or answer.

- Offline: `ollama pull llama3.2-vision` (or `llava`).
- Online: set an API key with a vision-capable model (e.g. `gpt-4o`).

Put an image in your workspace and ask it to analyze the file.

## Automatic profile (personal memory)

The assistant watches for personal details in what you say and stores them, then uses them in every reply.

- It learns your name, job, location, mood, likes, dislikes, and goals automatically.
- Example: say "my name is Carla and I work as a designer" and it remembers both.
- See it with `/profile`, add detail with the `remember_about_me` tool, remove an entry with `forget_about_me`.
- Single-value facts (name, location, job, mood) update when they change; likes, dislikes, and goals accumulate.

## Integrations

Run `/integrations` to see the panel. Each service shows `connected`, `needs setup`, or `offline`.

- Work with a token (easy): GitHub (`GITHUB_TOKEN`), Vercel (`VERCEL_TOKEN`), Discord (`DISCORD_WEBHOOK_URL`).
- Local, always on: Notes, and VS Code if the `code` command is installed.
- OAuth services: Google (Gmail, Drive, Docs, Meet) via `GOOGLE_ACCESS_TOKEN`, Microsoft (Outlook, OneDrive, Word, Teams) via `MS_GRAPH_TOKEN`, Twitter/X via `TWITTER_BEARER_TOKEN`, Instagram via `INSTAGRAM_ACCESS_TOKEN`. The real API calls are implemented and activate the moment you provide a valid token from that service's developer console.

The assistant calls integrations through the `use_integration` tool, e.g. integration `github`, action `list_repos`.

## Plugins

Add your own tools without touching the core. Drop a `.py` file in `~/.personal_ai/plugins/`:

```python
from personal_ai.tools.base import Tool

class WeatherTool(Tool):
    name = "weather"
    description = "Get the weather for a city."
    parameters = {"city": "City name"}
    def run(self, args):
        return f"It is sunny in {args.get('city')}."

def get_tools(config, memory):
    return [WeatherTool()]
```

It loads automatically on startup. A broken plugin is skipped, never crashing the app.

## Self-reprogramming

You can ask the assistant to give itself a new ability or remove one, in plain language. It writes a plugin, validates it, and loads it live, no restart needed.

- "Give yourself a tool that converts currencies" leads it to call `create_capability`, which writes and hot-loads the new tool.
- "Remove the currency tool" calls `remove_capability`.
- `/capabilities` lists built-in versus self-added tools.

Safety built in:

- New code is syntax-checked before it is saved.
- It is imported in isolation first; if it errors, the change is rolled back automatically.
- Every overwrite or removal is backed up under `plugins/.backups/`, and `restore_capability` brings one back.
- Only self-added plugins can change. The built-in core is protected and always survives a reload.
- Turn the whole feature off with `PERSONAL_AI_ALLOW_SELF_MODIFY=false`.

## Encryption at rest (optional)

Message and memory text can be encrypted in the local database.

```bash
pip install cryptography
export PERSONAL_AI_PASSPHRASE="your secret passphrase"
python run.py
```

Embeddings are computed from the plaintext before encryption, so semantic recall still works. Without the passphrase or package, data is stored as plaintext. For full database-file encryption, use SQLCipher.

## Voice (optional)

Hands-free use needs extra packages outside the zero-dependency core:

```bash
pip install SpeechRecognition pyttsx3   # plus PyAudio for the microphone
```

The `personal_ai.voice` module exposes `speak()` and `listen()` and degrades gracefully when the packages are missing.

## Backup and conversations

- `/export backup.json` and `/import backup.json` move all your data (memory, profile, tasks, history) in and out.
- `/new <title>`, `/sessions`, and `/switch <id>` manage separate named conversations.

## Where your data lives

Everything is under `PERSONAL_AI_HOME` (default `~/.personal_ai`):

- `personal_ai.db`: SQLite file with your conversations and memories.
- `workspace/`: the only folder the file tools can read or write.

Nothing is sent anywhere unless you enable an online model or use a web tool.

## Project layout

```
personal_ai/
  config.py         settings and local paths
  connectivity.py   online and Ollama detection
  platform_util.py  Windows/macOS/Linux/Android support
  http_util.py      stdlib HTTP helpers (incl. streaming)
  llm.py            local, online, mock brains + vision + streaming
  embeddings.py     local, online, and fallback embeddings
  memory.py         SQLite storage, recall, tasks, profile,
                    consolidation, conversations, export/import
  profile.py        automatic personal-fact extraction
  crypto.py         optional encryption at rest
  agent.py          the think-act-observe loop + context trimming
  reminders.py      due-task reminders
  plugins.py        load custom tools from a folder
  self_extend.py    write/load/remove the AI's own tools at runtime
  setup_wizard.py   doctor and first-run setup
  voice.py          optional speech in/out
  web_ui.py         local browser chat (stdlib http.server)
  tools/            calculator, code_exec, shell, files, search,
                    tasks, knowledge, vision, profile, memory,
                    datetime, web, integrations, self (reprogramming)
  integrations/     google, microsoft, social, devtools, notes
  cli.py            interactive chat
tests/
  test_core.py      offline tests for the whole pipeline (53 tests)
run.py              entry point (chat / --web / --doctor / --setup)
Dockerfile          one-command container run
.github/workflows/  CI across Python 3.9-3.12
```

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover the calculator, embeddings, memory and semantic recall, JSON parsing, sandboxed file access, and the full agent loop driven by a mock brain. They run fully offline.

## Privacy and safety notes

- File tools are sandboxed to the workspace folder and cannot escape it.
- The calculator parses math with an AST and only allows a whitelist of functions, never arbitrary code.
- The `run_python` and `run_shell` tools execute real code and commands on your machine, in a separate process with a timeout, inside the workspace. They are meant for a machine you control. Turn them off with `PERSONAL_AI_ALLOW_CODE=false` and `PERSONAL_AI_ALLOW_SHELL=false`.
- The agent has a step limit so it cannot loop forever.
- Local mode keeps every byte on your machine.
