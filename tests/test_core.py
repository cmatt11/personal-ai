"""Tests for the parts that run without a network or external model.

These verify the calculator, the hashing embeddings, the SQLite memory store
(including semantic search), JSON extraction, and the full agent loop driven by
a MockBackend standing in for the LLM.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from personal_ai.agent import Agent, extract_json
from personal_ai.config import Config
from personal_ai.embeddings import HashingEmbeddings, cosine_similarity
from personal_ai.integrations import build_integrations
from personal_ai.llm import MockBackend
from personal_ai.memory import Memory
from personal_ai.profile import ProfileExtractor
from personal_ai.tools import build_registry
from personal_ai.tools.calculator import CalculatorTool
from personal_ai.tools.code_exec import RunPythonTool
from personal_ai.tools.files import DeleteFileTool, ReadFileTool, WriteFileTool
from personal_ai.tools.knowledge import IngestFileTool
from personal_ai.tools.search import FindFilesTool, SearchFilesTool
from personal_ai.tools.shell import RunShellTool
from personal_ai.tools.tasks import AddTaskTool, CompleteTaskTool, ListTasksTool
from personal_ai.tools.vision import AnalyzeImageTool


def make_config(tmp: str) -> Config:
    os.environ["PERSONAL_AI_HOME"] = tmp
    os.environ["PERSONAL_AI_WORKSPACE"] = str(Path(tmp) / "ws")
    os.environ["PERSONAL_AI_ALLOW_WEB"] = "false"  # keep tests offline
    cfg = Config()
    cfg.ensure_dirs()
    return cfg


class TestCalculator(unittest.TestCase):
    def test_basic(self):
        tool = CalculatorTool()
        self.assertEqual(tool.run({"expression": "2 + 3 * 4"}), "14")

    def test_percentage(self):
        tool = CalculatorTool()
        self.assertEqual(tool.run({"expression": "0.15 * 240"}), "36.0")

    def test_rejects_code(self):
        tool = CalculatorTool()
        out = tool.run({"expression": "__import__('os').system('echo hi')"})
        self.assertTrue(out.startswith("error"))

    def test_math_functions(self):
        tool = CalculatorTool()
        self.assertEqual(tool.run({"expression": "sqrt(144)"}), "12.0")
        self.assertEqual(tool.run({"expression": "factorial(5)"}), "120")
        self.assertEqual(tool.run({"expression": "log(100, 10)"}), "2.0")
        # constants and nested functions
        self.assertEqual(tool.run({"expression": "round(sin(pi/2))"}), "1")

    def test_rejects_unknown_function(self):
        tool = CalculatorTool()
        out = tool.run({"expression": "eval('2+2')"})
        self.assertTrue(out.startswith("error"))


class TestRunPython(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = make_config(self.tmp)
        self.tool = RunPythonTool(self.cfg.workspace)

    def test_executes_and_prints(self):
        out = self.tool.run({"code": "print(sum(range(101)))"})
        self.assertIn("5050", out)

    def test_real_program_logic(self):
        code = (
            "def is_prime(n):\n"
            "    if n < 2: return False\n"
            "    for i in range(2, int(n**0.5)+1):\n"
            "        if n % i == 0: return False\n"
            "    return True\n"
            "print([n for n in range(20) if is_prime(n)])\n"
        )
        out = self.tool.run({"code": code})
        self.assertIn("[2, 3, 5, 7, 11, 13, 17, 19]", out)

    def test_captures_errors(self):
        out = self.tool.run({"code": "print(1/0)"})
        self.assertIn("ZeroDivisionError", out)

    def test_timeout(self):
        out = self.tool.run({"code": "while True: pass", "timeout": 1})
        self.assertIn("timed out", out)


class TestEmbeddings(unittest.TestCase):
    def test_similar_texts_score_higher(self):
        emb = HashingEmbeddings(dim=512)
        v_cat1 = emb.embed_one("my cat likes to sleep on the couch")
        v_cat2 = emb.embed_one("the cat sleeps on a sofa")
        v_finance = emb.embed_one("quarterly tax filing deadline for business")
        sim_related = cosine_similarity(v_cat1, v_cat2)
        sim_unrelated = cosine_similarity(v_cat1, v_finance)
        self.assertGreater(sim_related, sim_unrelated)


class TestMemory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = make_config(self.tmp)
        self.mem = Memory(self.cfg.db_path, HashingEmbeddings())

    def tearDown(self):
        self.mem.close()

    def test_messages_roundtrip(self):
        self.mem.add_message("s1", "user", "hello")
        self.mem.add_message("s1", "assistant", "hi there")
        recent = self.mem.recent_messages("s1", 10)
        self.assertEqual(recent[0]["content"], "hello")
        self.assertEqual(recent[1]["role"], "assistant")

    def test_semantic_recall(self):
        self.mem.add_memory("The user's dog is named Pixel.")
        self.mem.add_memory("The user works as a graphic designer.")
        self.mem.add_memory("The user lives in Cebu.")
        hits = self.mem.search_memory("what is the pet called", top_k=1, min_score=0.0)
        self.assertTrue(hits)
        self.assertIn("Pixel", hits[0][1])


class TestExtractJson(unittest.TestCase):
    def test_plain(self):
        obj = extract_json('{"answer": "hello"}')
        self.assertEqual(obj["answer"], "hello")

    def test_fenced_with_prose(self):
        text = 'Sure!\n```json\n{"tool": "calculator", "args": {"expression": "1+1"}}\n```'
        obj = extract_json(text)
        self.assertEqual(obj["tool"], "calculator")

    def test_none_when_no_json(self):
        self.assertIsNone(extract_json("just some normal text"))


class TestFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = make_config(self.tmp)

    def test_write_then_read(self):
        writer = WriteFileTool(self.cfg.workspace)
        reader = ReadFileTool(self.cfg.workspace)
        writer.run({"path": "notes/todo.txt", "content": "buy milk"})
        self.assertEqual(reader.run({"path": "notes/todo.txt"}), "buy milk")

    def test_sandbox_escape_blocked(self):
        reader = ReadFileTool(self.cfg.workspace)
        out = reader.run({"path": "../../../../etc/passwd"})
        self.assertTrue(out.startswith("error"))

    def test_delete(self):
        writer = WriteFileTool(self.cfg.workspace)
        deleter = DeleteFileTool(self.cfg.workspace)
        writer.run({"path": "tmp.txt", "content": "x"})
        self.assertIn("Deleted", deleter.run({"path": "tmp.txt"}))
        self.assertTrue(deleter.run({"path": "tmp.txt"}).startswith("error"))


class TestShell(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = make_config(self.tmp)
        self.tool = RunShellTool(self.cfg.workspace)

    def test_runs_command(self):
        out = self.tool.run({"command": "echo hello-shell"})
        self.assertIn("hello-shell", out)

    def test_runs_in_workspace(self):
        WriteFileTool(self.cfg.workspace).run({"path": "a.txt", "content": "hi"})
        out = self.tool.run({"command": "ls"})
        self.assertIn("a.txt", out)


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = make_config(self.tmp)
        w = WriteFileTool(self.cfg.workspace)
        w.run({"path": "notes/alpha.txt", "content": "the secret code is 42\nsecond line"})
        w.run({"path": "notes/beta.md", "content": "no match here"})

    def test_find_files(self):
        out = FindFilesTool(self.cfg.workspace).run({"pattern": "**/*.txt"})
        self.assertIn("alpha.txt", out)
        self.assertNotIn("beta.md", out)

    def test_search_files(self):
        out = SearchFilesTool(self.cfg.workspace).run({"query": "secret code"})
        self.assertIn("alpha.txt", out)
        self.assertIn("42", out)


class TestTasks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = make_config(self.tmp)
        self.mem = Memory(self.cfg.db_path, HashingEmbeddings())

    def tearDown(self):
        self.mem.close()

    def test_add_list_complete(self):
        add = AddTaskTool(self.mem)
        lst = ListTasksTool(self.mem)
        done = CompleteTaskTool(self.mem)
        add.run({"text": "buy milk", "due": "tomorrow"})
        add.run({"text": "call mom"})
        listing = lst.run({})
        self.assertIn("buy milk", listing)
        self.assertIn("call mom", listing)
        self.assertIn("(due tomorrow)", listing)
        # complete the first task
        first_id = self.mem.list_tasks()[0]["id"]
        self.assertIn("Completed", done.run({"id": first_id}))
        # it should drop off the open list
        self.assertNotIn(f"#{first_id} buy milk", lst.run({}))


class TestKnowledge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = make_config(self.tmp)
        self.mem = Memory(self.cfg.db_path, HashingEmbeddings())

    def tearDown(self):
        self.mem.close()

    def test_ingest_then_recall(self):
        doc = (
            "Project Aurora is our internal codename for the new mobile app.\n\n"
            "The launch date for Project Aurora is set for September 2026.\n\n"
            "The backend uses a local SQLite database for offline support.\n"
        )
        WriteFileTool(self.cfg.workspace).run({"path": "spec.txt", "content": doc})
        result = IngestFileTool(self.cfg.workspace, self.mem).run({"path": "spec.txt"})
        self.assertIn("Ingested", result)
        hits = self.mem.search_memory("when does Aurora launch", top_k=3, min_score=0.0)
        self.assertTrue(any("September 2026" in h[1] for h in hits))


class TestAgentLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = make_config(self.tmp)
        self.mem = Memory(self.cfg.db_path, HashingEmbeddings())
        self.registry = build_registry(self.cfg, self.mem)

    def tearDown(self):
        self.mem.close()

    def test_direct_answer(self):
        llm = MockBackend(scripted=['{"answer": "Hello, I am your local AI."}'])
        agent = Agent(llm, self.registry, self.mem, self.cfg)
        out = agent.chat("s1", "hi")
        self.assertIn("local AI", out)

    def test_tool_then_answer(self):
        # First the model calls the calculator, then it answers using the result.
        scripted = [
            '{"tool": "calculator", "args": {"expression": "0.15 * 240"}}',
            '{"answer": "15% of 240 is 36."}',
        ]
        llm = MockBackend(scripted=scripted)
        steps = []
        agent = Agent(llm, self.registry, self.mem, self.cfg)
        out = agent.chat("s1", "what is 15% of 240?", on_step=lambda k, d: steps.append((k, d)))
        self.assertIn("36", out)
        self.assertTrue(any(k == "tool" for k, _ in steps))
        self.assertTrue(any("36" in d for k, d in steps if k == "observation"))

    def test_remember_and_recall_across_turns(self):
        # Turn 1: store a fact via the remember tool.
        llm1 = MockBackend(
            scripted=[
                '{"tool": "remember", "args": {"text": "The user prefers tea over coffee."}}',
                '{"answer": "Noted, you prefer tea."}',
            ]
        )
        Agent(llm1, self.registry, self.mem, self.cfg).chat("s1", "I prefer tea over coffee")
        # Turn 2: the fact should be retrievable from memory.
        hits = self.mem.search_memory("what drink does the user like", top_k=2, min_score=0.0)
        self.assertTrue(any("tea" in h[1].lower() for h in hits))

    def test_unknown_tool_is_handled(self):
        scripted = [
            '{"tool": "nonexistent", "args": {}}',
            '{"answer": "Recovered after a bad tool call."}',
        ]
        llm = MockBackend(scripted=scripted)
        agent = Agent(llm, self.registry, self.mem, self.cfg)
        out = agent.chat("s1", "do something")
        self.assertIn("Recovered", out)

    def test_agent_writes_and_runs_a_program(self):
        # The model writes a real program, runs it, and answers with the output.
        program = (
            "nums=[5,3,8,1,9,2]\n"
            "print('sorted:', sorted(nums))\n"
            "print('max:', max(nums))\n"
        )
        scripted = [
            json.dumps({"tool": "run_python", "args": {"code": program}}),
            '{"answer": "Done. Sorted list and max computed by running the code."}',
        ]
        llm = MockBackend(scripted=scripted)
        steps = []
        agent = Agent(llm, self.registry, self.mem, self.cfg)
        out = agent.chat(
            "s1", "sort [5,3,8,1,9,2] and give the max", on_step=lambda k, d: steps.append((k, d))
        )
        # The observation must contain the ACTUAL program output, not a guess.
        observations = [d for k, d in steps if k == "observation"]
        self.assertTrue(any("sorted: [1, 2, 3, 5, 8, 9]" in o for o in observations))
        self.assertTrue(any("max: 9" in o for o in observations))
        self.assertIn("Done", out)

    def test_agent_does_real_math(self):
        scripted = [
            json.dumps({"tool": "calculator", "args": {"expression": "sqrt(2)**2"}}),
            '{"answer": "The result is about 2."}',
        ]
        llm = MockBackend(scripted=scripted)
        steps = []
        agent = Agent(llm, self.registry, self.mem, self.cfg)
        agent.chat("s1", "what is sqrt(2) squared", on_step=lambda k, d: steps.append((k, d)))
        observations = [d for k, d in steps if k == "observation"]
        self.assertTrue(any(o.startswith("2") or "2.0000" in o for o in observations))

    def test_agent_manages_tasks(self):
        scripted = [
            json.dumps({"tool": "add_task", "args": {"text": "submit report", "due": "Friday"}}),
            '{"answer": "Added it to your tasks for Friday."}',
        ]
        llm = MockBackend(scripted=scripted)
        agent = Agent(llm, self.registry, self.mem, self.cfg)
        agent.chat("s1", "remind me to submit the report on Friday")
        open_tasks = self.mem.list_tasks()
        self.assertTrue(any("submit report" in t["text"] for t in open_tasks))


class TestProfile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = make_config(self.tmp)
        self.mem = Memory(self.cfg.db_path, HashingEmbeddings())
        self.ex = ProfileExtractor(self.mem)

    def tearDown(self):
        self.mem.close()

    def test_extracts_personal_facts(self):
        self.ex.observe("my name is Carla and I live in Cebu.")
        self.ex.observe("I work as a graphic designer.")
        self.ex.observe("I love hiking.")
        self.ex.observe("I hate slow wifi.")
        self.ex.observe("My goal is to launch my own studio.")
        self.ex.observe("I'm feeling motivated today.")
        prof = self.mem.get_profile()
        self.assertEqual(prof.get("name"), ["Carla"])
        self.assertEqual(prof.get("location"), ["Cebu"])
        self.assertEqual(prof.get("job"), ["graphic designer"])
        self.assertIn("hiking", prof.get("like", []))
        self.assertIn("slow wifi", prof.get("dislike", []))
        self.assertEqual(prof.get("mood"), ["motivated"])

    def test_single_value_categories_replace(self):
        self.ex.observe("I'm feeling great.")
        self.ex.observe("I'm feeling tired.")
        self.assertEqual(self.mem.get_profile().get("mood"), ["tired"])

    def test_manual_add_and_delete(self):
        self.mem.set_profile("like", "tea")
        rows = self.mem.list_profile()
        self.assertTrue(any(r["value"] == "tea" for r in rows))
        pid = [r["id"] for r in rows if r["value"] == "tea"][0]
        self.assertTrue(self.mem.delete_profile(pid))

    def test_profile_injected_via_agent(self):
        registry = build_registry(self.cfg, self.mem)
        llm = MockBackend(scripted=['{"answer":"Hello Carla!"}'])
        agent = Agent(llm, registry, self.mem, self.cfg)
        agent.chat("s1", "hey, my name is Carla")
        self.assertEqual(self.mem.get_profile().get("name"), ["Carla"])


class TestIntegrations(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = make_config(self.tmp)

    def test_panel_lists_all_categories(self):
        panel = build_integrations(self.cfg).panel()
        for svc in ["GitHub", "Gmail", "Outlook", "Discord", "Notes (local)", "Vercel"]:
            self.assertIn(svc, panel)

    def test_unconfigured_reports_setup(self):
        reg = build_integrations(self.cfg)
        gh = reg.get("github")
        # No token in env -> needs setup, guarded call refuses gracefully.
        out = gh.guarded_call("list_repos", {})
        self.assertIn("not connected", out)

    def test_local_notes_works_offline(self):
        reg = build_integrations(self.cfg)
        notes = reg.get("notes")
        self.assertEqual(notes.status(), "connected")
        self.assertIn("Saved", notes.guarded_call("add", {"title": "Ideas", "body": "build stuff"}))
        self.assertIn("Ideas", notes.guarded_call("list", {}))
        self.assertIn("build stuff", notes.guarded_call("read", {"title": "Ideas"}))


class TestVision(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = make_config(self.tmp)

    def test_no_backend_message(self):
        tool = AnalyzeImageTool(self.cfg.workspace, None)
        out = tool.run({"path": "x.png"})
        self.assertIn("No vision model", out)

    def test_analyzes_with_mock_vision(self):
        # write a fake image file
        img = self.cfg.workspace / "pic.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\nfakeimagedata")
        backend = MockBackend()
        backend.supports_vision = True
        tool = AnalyzeImageTool(self.cfg.workspace, backend)
        out = tool.run({"path": "pic.png", "question": "what is this?"})
        self.assertIn("mock vision", out)
        self.assertIn("1 image", out)


class TestPlatform(unittest.TestCase):
    def test_os_detection(self):
        from personal_ai import platform_util

        self.assertIn(platform_util.os_name(), {"windows", "macos", "linux", "android", "unknown"})

    def test_shell_command_shape(self):
        from personal_ai import platform_util

        argv = platform_util.shell_command("echo hi")
        self.assertIsInstance(argv, list)
        self.assertIn("echo hi", argv)


class TestUpgrades(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = make_config(self.tmp)
        self.mem = Memory(self.cfg.db_path, HashingEmbeddings())
        self.registry = build_registry(self.cfg, self.mem)

    def tearDown(self):
        self.mem.close()

    def test_streaming_delivers_full_answer(self):
        from personal_ai.agent import Agent

        llm = MockBackend(scripted=["Hello there, this is a streamed reply."])
        agent = Agent(llm, self.registry, self.mem, self.cfg)
        tokens = []
        answer = agent.chat("s1", "hi", on_token=lambda t: tokens.append(t))
        self.assertEqual("".join(tokens).strip(), answer)
        self.assertIn("streamed reply", answer)

    def test_streaming_hides_tool_json(self):
        from personal_ai.agent import Agent

        # First a tool call (JSON, should NOT stream), then a prose answer.
        scripted = [
            json.dumps({"tool": "calculator", "args": {"expression": "2+2"}}),
            "The answer is 4.",
        ]
        llm = MockBackend(scripted=scripted)
        agent = Agent(llm, self.registry, self.mem, self.cfg)
        tokens = []
        agent.chat("s1", "what is 2+2", on_token=lambda t: tokens.append(t))
        streamed = "".join(tokens)
        self.assertNotIn("calculator", streamed)
        self.assertIn("answer is 4", streamed)

    def test_context_trim(self):
        from personal_ai.agent import trim_to_budget

        msgs = [{"role": "system", "content": "S" * 100}]
        msgs += [{"role": "user", "content": "U" * 100} for _ in range(50)]
        trimmed = trim_to_budget(msgs, char_budget=400)
        total = sum(len(m["content"]) for m in trimmed)
        self.assertLessEqual(total, 700)  # system + budget + note
        self.assertEqual(trimmed[0]["role"], "system")
        self.assertTrue(any("trimmed" in m["content"] for m in trimmed))

    def test_memory_consolidation_dedupes(self):
        for _ in range(5):
            self.mem.add_memory("The user loves hiking on weekends.", kind="conversation")
        self.mem.add_memory("Completely different unrelated fact about taxes.", kind="conversation")
        before = self.mem.count_memories()
        stats = self.mem.consolidate(similarity_threshold=0.9)
        after = self.mem.count_memories()
        self.assertLess(after, before)
        self.assertGreater(stats["removed_duplicates"], 0)

    def test_conversations(self):
        self.mem.create_conversation("abc", "Work chat")
        self.mem.create_conversation("xyz", "Personal")
        titles = [c["title"] for c in self.mem.list_conversations()]
        self.assertIn("Work chat", titles)
        self.mem.rename_conversation("abc", "Renamed")
        self.assertIn("Renamed", [c["title"] for c in self.mem.list_conversations()])

    def test_export_import_roundtrip(self):
        self.mem.add_memory("Backup me please.", kind="fact")
        self.mem.set_profile("name", "Carla")
        self.mem.add_task("test task")
        data = self.mem.export_data()

        other = Memory(Path(self.tmp) / "restore.db", HashingEmbeddings())
        counts = other.import_data(data)
        self.assertGreaterEqual(counts["memories"], 1)
        self.assertTrue(any("Backup me" in h[1] for h in other.search_memory("backup", min_score=0.0)))
        self.assertEqual(other.get_profile().get("name"), ["Carla"])
        other.close()

    def test_reminders_flags_overdue(self):
        from personal_ai.reminders import due_reminders

        self.mem.add_task("file taxes", due="2000-01-01")
        self.mem.add_task("no due date task")
        lines = due_reminders(self.mem)
        self.assertTrue(any("OVERDUE" in l for l in lines))
        self.assertTrue(all("no due date" not in l for l in lines))

    def test_plugin_loading(self):
        from personal_ai.plugins import load_plugins, plugins_dir

        plugin_code = (
            "from personal_ai.tools.base import Tool\n"
            "class HelloTool(Tool):\n"
            "    name = 'hello_plugin'\n"
            "    description = 'a test plugin tool'\n"
            "    parameters = {}\n"
            "    def run(self, args):\n"
            "        return 'hi from plugin'\n"
            "def get_tools(config, memory):\n"
            "    return [HelloTool()]\n"
        )
        (plugins_dir(self.cfg) / "hello.py").write_text(plugin_code, encoding="utf-8")
        loaded = load_plugins(self.registry, self.cfg, self.mem)
        self.assertIn("hello", loaded)
        self.assertTrue(self.registry.has("hello_plugin"))
        self.assertEqual(self.registry.get("hello_plugin").run({}), "hi from plugin")


class TestEncryption(unittest.TestCase):
    class ReversibleCipher:
        enabled = True

        def encrypt(self, text):
            return "E:" + text if text is not None else text

        def decrypt(self, text):
            return text[2:] if text and text.startswith("E:") else text

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = make_config(self.tmp)
        self.mem = Memory(self.cfg.db_path, HashingEmbeddings(), cipher=self.ReversibleCipher())

    def tearDown(self):
        self.mem.close()

    def test_messages_encrypted_at_rest_decrypted_on_read(self):
        self.mem.add_message("s1", "user", "secret diary entry")
        # raw stored value is ciphertext
        raw = self.mem.conn.execute("SELECT content FROM messages").fetchone()[0]
        self.assertTrue(raw.startswith("E:"))
        # read path decrypts transparently
        msgs = self.mem.recent_messages("s1", 10)
        self.assertEqual(msgs[0]["content"], "secret diary entry")

    def test_memory_encrypted_but_searchable(self):
        self.mem.add_memory("my bank pin is hidden here", kind="fact")
        raw = self.mem.conn.execute("SELECT text FROM memories").fetchone()[0]
        self.assertTrue(raw.startswith("E:"))
        hits = self.mem.search_memory("bank pin", min_score=0.0)
        self.assertTrue(any("bank pin" in h[1] for h in hits))


class TestWebUI(unittest.TestCase):
    def test_page_and_module_import(self):
        from personal_ai import web_ui

        self.assertIn("Personal AI", web_ui.PAGE)
        self.assertIn("/api/chat", web_ui.PAGE)
        self.assertTrue(hasattr(web_ui, "serve"))


class TestSelfExtend(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cfg = make_config(self.tmp)
        self.mem = Memory(self.cfg.db_path, HashingEmbeddings())
        self.registry = build_registry(self.cfg, self.mem)
        self.extender = self.registry.self_extender

    def tearDown(self):
        self.mem.close()

    PLUGIN = (
        "from personal_ai.tools.base import Tool\n"
        "class DoublerTool(Tool):\n"
        "    name = 'doubler'\n"
        "    description = 'double a number'\n"
        "    parameters = {'n': 'a number'}\n"
        "    def run(self, args):\n"
        "        return str(int(args.get('n', 0)) * 2)\n"
        "def get_tools(config, memory):\n"
        "    return [DoublerTool()]\n"
    )

    def test_self_tools_present(self):
        for name in ["create_capability", "remove_capability", "list_capabilities"]:
            self.assertTrue(self.registry.has(name))

    def test_create_then_use_then_remove(self):
        # Create a brand-new ability at runtime.
        result = self.extender.create_capability("doubler_plugin", self.PLUGIN)
        self.assertIn("doubler", result)
        self.assertTrue(self.registry.has("doubler"))
        # Use the freshly added tool.
        self.assertEqual(self.registry.get("doubler").run({"n": 21}), "42")
        # Remove it.
        removed = self.extender.remove_capability("doubler_plugin")
        self.assertIn("doubler", removed)
        self.assertFalse(self.registry.has("doubler"))

    def test_rejects_syntax_error(self):
        out = self.extender.create_capability("broken", "def oops(:\n  pass\n")
        self.assertIn("syntax error", out)
        self.assertFalse(self.registry.has("broken"))

    def test_rollback_on_import_error(self):
        bad = (
            "from personal_ai.tools.base import Tool\n"
            "raise RuntimeError('boom')\n"
            "def get_tools(config, memory):\n"
            "    return []\n"
        )
        out = self.extender.create_capability("explode", bad)
        self.assertIn("failed to import", out)
        # The broken file must not remain active.
        self.assertNotIn("explode", [p.stem for p in self.extender.dir().glob("*.py")])

    def test_core_tools_survive_reload(self):
        self.extender.create_capability("doubler_plugin", self.PLUGIN)
        # Core tools must still be present after a self-modification reload.
        for name in ["calculator", "create_capability", "remember"]:
            self.assertTrue(self.registry.has(name))

    def test_agent_can_extend_itself(self):
        from personal_ai.agent import Agent

        code_json = json.dumps(self.PLUGIN)
        scripted = [
            '{"tool":"create_capability","args":{"name":"doubler_plugin","code":'
            + code_json
            + "}}",
            "I added a new doubler ability.",
        ]
        llm = MockBackend(scripted=scripted)
        agent = Agent(llm, self.registry, self.mem, self.cfg)
        agent.chat("s1", "give yourself a tool that doubles numbers")
        self.assertTrue(self.registry.has("doubler"))


class TestHybridBackend(unittest.TestCase):
    def _mk(self, online_up):
        from personal_ai.llm import HybridBackend, MockBackend

        online = MockBackend(scripted=["ONLINE reply"])
        local = MockBackend(scripted=["LOCAL reply"])
        online.name = "online:test"
        local.name = "local:test"
        return HybridBackend(online, local, online_check=lambda: online_up)

    def test_uses_online_when_connected(self):
        h = self._mk(online_up=True)
        self.assertEqual(h.chat([{"role": "user", "content": "hi"}]), "ONLINE reply")

    def test_falls_back_to_local_when_offline(self):
        h = self._mk(online_up=False)
        self.assertEqual(h.chat([{"role": "user", "content": "hi"}]), "LOCAL reply")

    def test_local_only_works_offline(self):
        from personal_ai.llm import HybridBackend, MockBackend

        local = MockBackend(scripted=["LOCAL only"])
        h = HybridBackend(None, local, online_check=lambda: False)
        self.assertEqual(h.chat([{"role": "user", "content": "hi"}]), "LOCAL only")

    def test_streaming_switches_too(self):
        h = self._mk(online_up=False)
        tokens = list(h.chat_stream([{"role": "user", "content": "hi"}]))
        self.assertIn("LOCAL", "".join(tokens))

    def test_select_backend_hybrid_mode(self):
        from personal_ai.config import Config
        from personal_ai.llm import HybridBackend, select_backend

        os.environ["PERSONAL_AI_BACKEND"] = "hybrid"
        os.environ["PERSONAL_AI_ONLINE_API_KEY"] = "sk-test"
        try:
            backend = select_backend(Config())
            self.assertIsInstance(backend, HybridBackend)
        finally:
            os.environ.pop("PERSONAL_AI_BACKEND", None)
            os.environ.pop("PERSONAL_AI_ONLINE_API_KEY", None)


if __name__ == "__main__":
    unittest.main(verbosity=2)
