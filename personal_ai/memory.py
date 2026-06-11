"""Local data storage: conversation history and long-term vector memory.

Everything lives in a single local SQLite file. Nothing leaves the machine.

- messages: full chat transcript, per session.
- memories: durable facts and past exchanges, each with an embedding for
  semantic recall (the RAG store).
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .embeddings import EmbeddingBackend, Vector, cosine_similarity


class _NullCipher:
    """Fallback no-op cipher so Memory works without the crypto module."""

    def encrypt(self, text):
        return text

    def decrypt(self, text):
        return text

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    session   TEXT NOT NULL,
    role      TEXT NOT NULL,
    content   TEXT NOT NULL,
    ts        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session);

CREATE TABLE IF NOT EXISTS memories (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    text       TEXT NOT NULL,
    kind       TEXT NOT NULL,
    source     TEXT,
    embedding  TEXT NOT NULL,
    ts         REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    text     TEXT NOT NULL,
    due      TEXT,
    done     INTEGER NOT NULL DEFAULT 0,
    created  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS profile (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    category  TEXT NOT NULL,
    value     TEXT NOT NULL,
    ts        REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_profile_cv ON profile(category, value);

CREATE TABLE IF NOT EXISTS conversations (
    session  TEXT PRIMARY KEY,
    title    TEXT NOT NULL,
    created  REAL NOT NULL
);
"""


class Memory:
    """SQLite-backed store for messages and semantic memories."""

    def __init__(self, db_path: Path, embedder: EmbeddingBackend, same_thread: bool = True, cipher=None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.cipher = cipher or _NullCipher()
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=same_thread)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ----- conversation history -----

    def add_message(self, session: str, role: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO messages (session, role, content, ts) VALUES (?, ?, ?, ?)",
            (session, role, self.cipher.encrypt(content), time.time()),
        )
        self.conn.commit()

    def recent_messages(self, session: str, limit: int) -> List[Dict[str, str]]:
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE session = ? "
            "ORDER BY id DESC LIMIT ?",
            (session, limit),
        ).fetchall()
        return [
            {"role": r["role"], "content": self.cipher.decrypt(r["content"])}
            for r in reversed(rows)
        ]

    def all_sessions(self) -> List[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT session FROM messages ORDER BY session"
        ).fetchall()
        return [r["session"] for r in rows]

    # ----- long-term semantic memory (RAG) -----

    def add_memory(self, text: str, kind: str = "note", source: Optional[str] = None) -> int:
        text = text.strip()
        if not text:
            return -1
        vec = self.embedder.embed_one(text)  # embed plaintext before encrypting
        cur = self.conn.execute(
            "INSERT INTO memories (text, kind, source, embedding, ts) "
            "VALUES (?, ?, ?, ?, ?)",
            (self.cipher.encrypt(text), kind, source, json.dumps(vec), time.time()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def search_memory(
        self, query: str, top_k: int = 4, min_score: float = 0.2
    ) -> List[Tuple[float, str, str]]:
        """Return [(score, text, kind)] for the closest memories to the query."""
        rows = self.conn.execute(
            "SELECT text, kind, embedding FROM memories"
        ).fetchall()
        if not rows:
            return []
        q_vec: Vector = self.embedder.embed_one(query)
        scored: List[Tuple[float, str, str]] = []
        for r in rows:
            try:
                vec = json.loads(r["embedding"])
            except (json.JSONDecodeError, TypeError):
                continue
            score = cosine_similarity(q_vec, vec)
            if score >= min_score:
                scored.append((score, self.cipher.decrypt(r["text"]), r["kind"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def count_memories(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0])

    def forget_all(self) -> None:
        self.conn.execute("DELETE FROM memories")
        self.conn.commit()

    def consolidate(
        self, similarity_threshold: float = 0.93, conversation_cap: int = 300
    ) -> Dict[str, int]:
        """Clean up long-term memory so recall stays sharp.

        - Removes near-duplicate memories (keeping the newest of each cluster).
        - Caps stored 'conversation' memories to the most recent N.

        Returns counts of what was removed. Runs fully offline.
        """
        removed_dupes = 0
        removed_old = 0

        rows = self.conn.execute(
            "SELECT id, text, kind, embedding, ts FROM memories ORDER BY ts DESC"
        ).fetchall()

        # De-duplicate by embedding similarity, keeping the newest (first seen).
        kept: List[Tuple[int, list]] = []
        to_delete: List[int] = []
        for r in rows:
            try:
                vec = json.loads(r["embedding"])
            except (json.JSONDecodeError, TypeError):
                continue
            is_dupe = False
            for _kid, kvec in kept:
                if cosine_similarity(vec, kvec) >= similarity_threshold:
                    is_dupe = True
                    break
            if is_dupe:
                to_delete.append(r["id"])
            else:
                kept.append((r["id"], vec))
        for mid in to_delete:
            self.conn.execute("DELETE FROM memories WHERE id = ?", (mid,))
            removed_dupes += 1

        # Cap conversation memories to the most recent N.
        conv_ids = [
            row["id"]
            for row in self.conn.execute(
                "SELECT id FROM memories WHERE kind = 'conversation' ORDER BY ts DESC"
            ).fetchall()
        ]
        for mid in conv_ids[conversation_cap:]:
            self.conn.execute("DELETE FROM memories WHERE id = ?", (mid,))
            removed_old += 1

        self.conn.commit()
        return {"removed_duplicates": removed_dupes, "removed_old": removed_old}

    # ----- named conversations -----

    def create_conversation(self, session: str, title: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO conversations (session, title, created) VALUES (?, ?, ?)",
            (session, title.strip() or session, time.time()),
        )
        self.conn.commit()

    def list_conversations(self) -> List[Dict[str, object]]:
        rows = self.conn.execute(
            "SELECT session, title, created FROM conversations ORDER BY created DESC"
        ).fetchall()
        return [{"session": r["session"], "title": r["title"]} for r in rows]

    def rename_conversation(self, session: str, title: str) -> None:
        self.conn.execute(
            "UPDATE conversations SET title = ? WHERE session = ?", (title.strip(), session)
        )
        self.conn.commit()

    # ----- backup: export / import -----

    def export_data(self) -> Dict[str, object]:
        """Dump all data to a JSON-serializable dict."""
        def dump(table: str) -> List[dict]:
            return [dict(r) for r in self.conn.execute(f"SELECT * FROM {table}").fetchall()]

        return {
            "version": 1,
            "messages": dump("messages"),
            "memories": dump("memories"),
            "tasks": dump("tasks"),
            "profile": dump("profile"),
            "conversations": dump("conversations"),
        }

    def import_data(self, data: Dict[str, object], replace: bool = False) -> Dict[str, int]:
        """Load data produced by export_data. If replace, wipe existing first."""
        counts: Dict[str, int] = {}
        tables = {
            "memories": ("text", "kind", "source", "embedding", "ts"),
            "tasks": ("text", "due", "done", "created"),
            "profile": ("category", "value", "ts"),
            "messages": ("session", "role", "content", "ts"),
            "conversations": ("session", "title", "created"),
        }
        for table, cols in tables.items():
            rows = data.get(table) or []
            if replace:
                self.conn.execute(f"DELETE FROM {table}")
            n = 0
            placeholders = ", ".join("?" for _ in cols)
            collist = ", ".join(cols)
            for row in rows:
                values = [row.get(c) for c in cols]
                try:
                    self.conn.execute(
                        f"INSERT OR IGNORE INTO {table} ({collist}) VALUES ({placeholders})",
                        values,
                    )
                    n += 1
                except sqlite3.Error:
                    continue
            counts[table] = n
        self.conn.commit()
        return counts

    # ----- tasks and reminders -----

    def add_task(self, text: str, due: Optional[str] = None) -> int:
        text = text.strip()
        if not text:
            return -1
        cur = self.conn.execute(
            "INSERT INTO tasks (text, due, done, created) VALUES (?, ?, 0, ?)",
            (text, due, time.time()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def list_tasks(self, include_done: bool = False) -> List[Dict[str, object]]:
        query = "SELECT id, text, due, done FROM tasks"
        if not include_done:
            query += " WHERE done = 0"
        query += " ORDER BY done, COALESCE(due, '9999'), id"
        rows = self.conn.execute(query).fetchall()
        return [
            {"id": r["id"], "text": r["text"], "due": r["due"], "done": bool(r["done"])}
            for r in rows
        ]

    def complete_task(self, task_id: int) -> bool:
        cur = self.conn.execute("UPDATE tasks SET done = 1 WHERE id = ?", (task_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def delete_task(self, task_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        self.conn.commit()
        return cur.rowcount > 0

    # ----- user profile (auto-learned facts about the user) -----

    # Categories that hold a single current value (replaced on update).
    SINGLE_CATEGORIES = {"name", "location", "job", "mood"}

    def set_profile(self, category: str, value: str) -> None:
        """Add a profile attribute. Single-value categories replace the old one."""
        category = category.strip().lower()
        value = value.strip()
        if not category or not value:
            return
        if category in self.SINGLE_CATEGORIES:
            self.conn.execute("DELETE FROM profile WHERE category = ?", (category,))
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO profile (category, value, ts) VALUES (?, ?, ?)",
                (category, value, time.time()),
            )
        except sqlite3.IntegrityError:
            pass
        self.conn.commit()

    def get_profile(self) -> Dict[str, List[str]]:
        rows = self.conn.execute(
            "SELECT category, value FROM profile ORDER BY category, ts"
        ).fetchall()
        out: Dict[str, List[str]] = {}
        for r in rows:
            out.setdefault(r["category"], []).append(r["value"])
        return out

    def list_profile(self) -> List[Dict[str, object]]:
        rows = self.conn.execute(
            "SELECT id, category, value FROM profile ORDER BY category, ts"
        ).fetchall()
        return [{"id": r["id"], "category": r["category"], "value": r["value"]} for r in rows]

    def delete_profile(self, profile_id: int) -> bool:
        cur = self.conn.execute("DELETE FROM profile WHERE id = ?", (profile_id,))
        self.conn.commit()
        return cur.rowcount > 0

    def profile_summary(self) -> str:
        """A compact human-readable profile string for prompt injection."""
        prof = self.get_profile()
        if not prof:
            return ""
        order = ["name", "job", "location", "mood", "like", "dislike", "goal"]
        labels = {
            "name": "Name",
            "job": "Job",
            "location": "Location",
            "mood": "Recent mood",
            "like": "Likes",
            "dislike": "Dislikes",
            "goal": "Goals",
        }
        parts = []
        for cat in order:
            if cat in prof:
                parts.append(f"{labels.get(cat, cat)}: {', '.join(prof[cat])}")
        for cat, vals in prof.items():
            if cat not in order:
                parts.append(f"{cat}: {', '.join(vals)}")
        return "\n".join(parts)

    def close(self) -> None:
        self.conn.close()
