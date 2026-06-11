"""Offline smoke test of the full pipeline using a mock brain.

Run: python demo_smoke.py
Demonstrates tool use, calculator, persistent SQLite memory, and recall.
"""

import os
import tempfile

os.environ["PERSONAL_AI_HOME"] = tempfile.mkdtemp()
os.environ["PERSONAL_AI_ALLOW_WEB"] = "false"

from personal_ai.agent import Agent
from personal_ai.config import Config
from personal_ai.embeddings import HashingEmbeddings
from personal_ai.llm import MockBackend
from personal_ai.memory import Memory
from personal_ai.tools import build_registry

cfg = Config()
cfg.ensure_dirs()
mem = Memory(cfg.db_path, HashingEmbeddings())
reg = build_registry(cfg, mem)

llm = MockBackend(
    scripted=[
        '{"tool":"remember","args":{"text":"The user is building a personal AI and lives in the Philippines."}}',
        '{"answer":"Got it. I will remember that."}',
        '{"tool":"calculator","args":{"expression":"(1200*0.12)+500"}}',
        '{"answer":"Your total is 1640."}',
    ]
)
agent = Agent(llm, reg, mem, cfg)

print("== turn 1 ==")
print(agent.chat("sess1", "Remember I live in the Philippines and am building a personal AI."))
print("== turn 2 ==")
print(agent.chat("sess1", "compute (1200*0.12)+500"))

print("\n== facts stored:", mem.count_memories())
print("== recall 'where does the user live':")
for s, t, k in mem.search_memory("where does the user live", top_k=3, min_score=0.0):
    print(f"  ({s:.2f}) {t}")
print("\n== DB file exists:", cfg.db_path.exists(), "->", cfg.db_path)
mem.close()
