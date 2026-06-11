"""Interactive command-line chat for the personal AI.

Run with:  python -m personal_ai   or   python run.py

Slash commands:
  /help              show commands
  /status            show backend, connectivity, memory stats
  /doctor            check environment readiness
  /integrations      show the integrations panel and connection status
  /profile           show what the assistant has learned about you
  /capabilities      list built-in and self-added tools
  /remember <text>   save a fact to long-term memory
  /recall <query>    search long-term memory
  /consolidate       clean up memory (dedupe + prune)
  /forget            wipe all long-term memory (asks to confirm)
  /tasks             list open tasks and reminders
  /new <title>       start a new named conversation
  /sessions          list saved conversations
  /switch <id>       switch to a saved conversation
  /export <path>     back up all data to a JSON file
  /import <path>     restore data from a JSON file
  /history           show recent messages this session
  /verbose           toggle showing tool calls
  /quit              exit
"""

import json
import sys
import uuid

from .agent import Agent
from .config import Config
from .connectivity import is_online, ollama_available
from .crypto import make_cipher
from .embeddings import select_embeddings
from .integrations import build_integrations
from .llm import LLMError, select_backend
from .memory import Memory
from .platform_util import summary as platform_summary
from .plugins import load_plugins
from .reminders import reminder_banner
from .setup_wizard import doctor
from .tools import build_registry

BANNER = r"""
  ____                                 _    _    ___
 |  _ \ ___ _ __ ___  ___  _ __   __ _| |  / \  |_ _|
 | |_) / _ \ '__/ __|/ _ \| '_ \ / _` | | / _ \  | |
 |  __/  __/ |  \__ \ (_) | | | | (_| | |/ ___ \ | |
 |_|   \___|_|  |___/\___/|_| |_|\__,_|_/_/   \_\___|
       local-first - private - offline-capable
"""


def _status_line(config: Config, llm_name: str, embed_name: str, mem: Memory) -> str:
    online = "online" if is_online() else "offline"
    enc = "on" if getattr(mem.cipher, "enabled", False) else "off"
    return (
        f"platform: {platform_summary()} | network: {online} | brain: {llm_name} | "
        f"memory-embed: {embed_name} | facts: {mem.count_memories()} | encryption: {enc}"
    )


def run() -> int:
    config = Config()
    config.ensure_dirs()

    print(BANNER)

    embedder = select_embeddings(config)
    memory = Memory(config.db_path, embedder, cipher=make_cipher(config))

    try:
        llm = select_backend(config)
    except LLMError as exc:
        print("Could not start the AI brain:\n")
        print(str(exc))
        print("\nRun `python run.py --doctor` to check your setup, or install Ollama:")
        print(f"  ollama pull {config.local_model} && ollama pull {config.local_embed_model}")
        memory.close()
        return 1

    registry = build_registry(config, memory, vision_backend=llm)
    plugins = load_plugins(registry, config, memory)
    integrations = build_integrations(config)
    agent = Agent(llm, registry, memory, config)
    session = uuid.uuid4().hex[:8]
    memory.create_conversation(session, "default")
    verbose = True
    msg_count = 0

    print(_status_line(config, llm.name, embedder.name, memory))
    vision = "yes" if getattr(llm, "supports_vision", False) else "no"
    stream = "yes" if getattr(llm, "supports_streaming", False) else "no"
    print(
        f"image analysis: {vision} | streaming: {stream} | "
        f"integrations: {len(integrations.all())} | plugins: {len(plugins)}"
    )
    banner = reminder_banner(memory)
    if banner:
        print("\n" + banner)
    print("\nType your message, or /help for commands.\n")

    def on_step(kind: str, detail: str) -> None:
        if not verbose:
            return
        if kind == "tool":
            print(f"  [tool] {detail}")
        elif kind == "learned":
            print(f"  [learned about you] {detail}")
        else:
            short = detail if len(detail) < 200 else detail[:200] + " ..."
            print(f"  [result] {short}")

    while True:
        try:
            user_input = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd, _, arg = user_input[1:].partition(" ")
            cmd = cmd.lower()
            arg = arg.strip()

            if cmd in ("quit", "exit", "q"):
                print("bye")
                break
            if cmd == "help":
                print(__doc__)
                continue
            if cmd == "status":
                print(_status_line(config, llm.name, embedder.name, memory))
                print(f"ollama reachable: {ollama_available(config.ollama_host)}")
                print(f"tools: {', '.join(t.name for t in registry.all())}")
                continue
            if cmd == "doctor":
                print(doctor(config))
                continue
            if cmd in ("integrations", "panel"):
                print(integrations.panel())
                continue
            if cmd in ("capabilities", "caps"):
                ext = getattr(registry, "self_extender", None)
                if ext is None:
                    print("self-modification is disabled (PERSONAL_AI_ALLOW_SELF_MODIFY=false)")
                else:
                    print(ext.list_capabilities())
                continue
            if cmd == "profile":
                rows = memory.list_profile()
                if not rows:
                    print("(nothing learned about you yet)")
                for r in rows:
                    print(f"  #{r['id']} [{r['category']}] {r['value']}")
                continue
            if cmd == "remember":
                if not arg:
                    print("usage: /remember <text>")
                    continue
                memory.add_memory(arg, kind="fact", source="cli")
                print(f"remembered: {arg}")
                continue
            if cmd == "recall":
                if not arg:
                    print("usage: /recall <query>")
                    continue
                hits = memory.search_memory(arg, top_k=config.memory_top_k)
                if not hits:
                    print("(nothing relevant found)")
                for score, text, kind in hits:
                    print(f"  ({score:.2f}) [{kind}] {text}")
                continue
            if cmd == "consolidate":
                stats = memory.consolidate()
                print(f"removed {stats['removed_duplicates']} duplicates, "
                      f"{stats['removed_old']} old conversation memories")
                continue
            if cmd == "forget":
                confirm = input("Wipe ALL long-term memory? type 'yes': ").strip()
                if confirm.lower() == "yes":
                    memory.forget_all()
                    print("memory cleared")
                else:
                    print("cancelled")
                continue
            if cmd == "tasks":
                banner = reminder_banner(memory)
                print(banner or "(no reminders)")
                for t in memory.list_tasks():
                    print(f"  #{t['id']} {t['text']}" + (f" (due {t['due']})" if t["due"] else ""))
                continue
            if cmd == "new":
                session = uuid.uuid4().hex[:8]
                memory.create_conversation(session, arg or session)
                print(f"started conversation '{arg or session}' ({session})")
                continue
            if cmd == "sessions":
                for c in memory.list_conversations():
                    marker = " *" if c["session"] == session else ""
                    print(f"  {c['session']}  {c['title']}{marker}")
                continue
            if cmd == "switch":
                if not arg:
                    print("usage: /switch <session-id>")
                    continue
                session = arg
                print(f"switched to {session}")
                continue
            if cmd == "export":
                if not arg:
                    print("usage: /export <path>")
                    continue
                with open(arg, "w", encoding="utf-8") as fh:
                    json.dump(memory.export_data(), fh)
                print(f"exported to {arg}")
                continue
            if cmd == "import":
                if not arg:
                    print("usage: /import <path>")
                    continue
                with open(arg, "r", encoding="utf-8") as fh:
                    counts = memory.import_data(json.load(fh))
                print(f"imported: {counts}")
                continue
            if cmd == "history":
                for m in memory.recent_messages(session, config.history_turns):
                    print(f"  {m['role']}: {m['content']}")
                continue
            if cmd == "verbose":
                verbose = not verbose
                print(f"verbose {'on' if verbose else 'off'}")
                continue
            print(f"unknown command: /{cmd} (try /help)")
            continue

        # Stream the answer as it generates.
        state = {"started": False}

        def on_token(tok: str) -> None:
            if not state["started"]:
                print("ai > ", end="", flush=True)
                state["started"] = True
            print(tok, end="", flush=True)

        try:
            agent.chat(session, user_input, on_step=on_step, on_token=on_token)
        except LLMError as exc:
            print(f"ai > [brain error] {exc}")
            continue
        except Exception as exc:  # never crash the REPL
            print(f"\nai > [error] {exc}")
            continue

        if state["started"]:
            print("\n")
        else:
            print("ai > (no response)\n")

        # Periodically consolidate memory to keep recall sharp.
        msg_count += 1
        if config.consolidate_every and msg_count % config.consolidate_every == 0:
            try:
                memory.consolidate()
            except Exception:
                pass

    memory.close()
    return 0


if __name__ == "__main__":
    sys.exit(run())
