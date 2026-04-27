"""
SQLite-backed lineage store for the stem agent evolution system.

Design principles:
- Single writer thread with a queue to prevent "database is locked" errors
  under high async concurrency.
- Full audit trail: every spec, every evaluation, every mutation recorded.
- Resumable: on startup, load existing population and scores from DB.
"""
import sqlite3
import threading
import queue
import json
import time
import os
from typing import Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    generation  INTEGER NOT NULL,
    best_id     TEXT,
    best_score  REAL,
    timestamp   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS specs (
    id            TEXT PRIMARY KEY,
    generation    INTEGER NOT NULL DEFAULT 0,
    parent_id     TEXT,
    mutation_type TEXT,
    spec_json     TEXT NOT NULL,
    eliminated    INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluations (
    rowid       INTEGER PRIMARY KEY AUTOINCREMENT,
    spec_id     TEXT NOT NULL,
    task_id     TEXT NOT NULL,
    score       REAL NOT NULL,
    matched     INTEGER NOT NULL,
    extracted   TEXT,
    cause_code  TEXT,
    timestamp   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_traces (
    rowid       INTEGER PRIMARY KEY AUTOINCREMENT,
    spec_id     TEXT NOT NULL,
    task_id     TEXT NOT NULL,
    messages    TEXT,
    token_count INTEGER NOT NULL DEFAULT 0,
    status      TEXT,
    timestamp   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mutations (
    rowid         INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id     TEXT NOT NULL,
    child_id      TEXT NOT NULL,
    mutation_type TEXT NOT NULL,
    prompt_used   TEXT,
    timestamp     TEXT NOT NULL
);
"""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class LineageDB:
    """
    Thread-safe SQLite lineage store.
    All writes go through an internal queue processed by a single writer thread,
    which eliminates "database is locked" errors under concurrent async loads.
    """

    def __init__(self, db_path: str = "logs/lineage.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()

        # Init schema
        conn = sqlite3.connect(db_path)
        conn.executescript(_SCHEMA)
        conn.commit()
        conn.close()

        # Start background writer thread
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="lineage-db-writer"
        )
        self._writer_thread.start()

        self._register_shutdown_handlers()

    def _register_shutdown_handlers(self):
        """Registers handlers to gracefully flush and close the DB on exit."""
        import atexit
        import signal
        
        def graceful_shutdown(*args):
            self.close()

        atexit.register(graceful_shutdown)
        
        try:
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, graceful_shutdown)
                signal.signal(signal.SIGTERM, graceful_shutdown)
        except (ValueError, AttributeError):
            pass

    # ------------------------------------------------------------------
    # Public write API  (enqueues; returns immediately)
    # ------------------------------------------------------------------

    def log_spec(self, spec_id: str, generation: int, spec_json: dict,
                 parent_id: Optional[str] = None, mutation_type: Optional[str] = None):
        self._enqueue(
            "INSERT OR IGNORE INTO specs (id, generation, parent_id, mutation_type, spec_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (spec_id, generation, parent_id, mutation_type, json.dumps(spec_json), _now())
        )

    def log_evaluation(self, spec_id: str, task_id: str, score: float,
                       matched: bool, extracted: str, cause_code: str):
        self._enqueue(
            "INSERT INTO evaluations (spec_id, task_id, score, matched, extracted, cause_code, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (spec_id, task_id, score, int(matched), extracted, cause_code, _now())
        )

    def log_run_trace(self, spec_id: str, task_id: str, messages: list,
                      token_count: int, status: str):
        self._enqueue(
            "INSERT INTO run_traces (spec_id, task_id, messages, token_count, status, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (spec_id, task_id, json.dumps(messages), token_count, status, _now())
        )

    def log_mutation(self, parent_id: str, child_id: str,
                     mutation_type: str, prompt_used: str = ""):
        self._enqueue(
            "INSERT INTO mutations (parent_id, child_id, mutation_type, prompt_used, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (parent_id, child_id, mutation_type, prompt_used, _now())
        )

    def log_generation(self, generation: int, best_id: str, best_score: float):
        self._enqueue(
            "INSERT INTO generations (generation, best_id, best_score, timestamp) "
            "VALUES (?, ?, ?, ?)",
            (generation, best_id, best_score, _now())
        )

    def mark_eliminated(self, spec_id: str):
        self._enqueue(
            "UPDATE specs SET eliminated = 1 WHERE id = ?",
            (spec_id,)
        )

    # ------------------------------------------------------------------
    # Public read API  (direct, synchronous — reads are safe from any thread)
    # ------------------------------------------------------------------

    def load_population(self) -> list:
        """Return all non-eliminated specs as dicts (for resuming evolution)."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT spec_json FROM specs WHERE eliminated = 0 ORDER BY created_at ASC"
        ).fetchall()
        conn.close()
        return [json.loads(r["spec_json"]) for r in rows]

    def load_scores(self) -> dict:
        """Return {spec_id: avg_score} for all evaluated specs."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT spec_id, AVG(score) as avg_score FROM evaluations GROUP BY spec_id"
        ).fetchall()
        conn.close()
        return {r["spec_id"]: r["avg_score"] for r in rows}

    def get_last_generation(self) -> int:
        """Return the last completed generation number (0 if none)."""
        conn = sqlite3.connect(self._db_path)
        row = conn.execute("SELECT MAX(generation) FROM generations").fetchone()
        conn.close()
        return row[0] or 0

    def get_lineage(self, spec_id: str) -> list:
        """Return the ancestry chain for a spec as a list of spec_ids."""
        chain = [spec_id]
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        current = spec_id
        while True:
            row = conn.execute(
                "SELECT parent_id FROM specs WHERE id = ?", (current,)
            ).fetchone()
            if not row or not row["parent_id"]:
                break
            chain.append(row["parent_id"])
            current = row["parent_id"]
        conn.close()
        return list(reversed(chain))

    def get_failure_examples(self, spec_id: str, limit: int = 3) -> list:
        """Return recent low-score run traces for a spec (for rich mutation prompts)."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT task_id, messages FROM run_traces "
            "WHERE spec_id = ? ORDER BY timestamp DESC LIMIT ?",
            (spec_id, limit)
        ).fetchall()
        conn.close()
        results = []
        for r in rows:
            try:
                msgs = json.loads(r["messages"])
                last_content = next(
                    (m.get("content", "") for m in reversed(msgs)
                     if m.get("role") == "assistant" and m.get("content")),
                    ""
                )
                results.append({
                    "task_id":      r["task_id"],
                    "output_snippet": last_content[:300]
                })
            except Exception:
                pass
        return results

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def flush(self):
        """Block until all queued writes are committed."""
        self._queue.join()

    def close(self):
        self._stop_event.set()
        self._queue.put(None)  # sentinel
        self._writer_thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Internal writer loop
    # ------------------------------------------------------------------

    def _enqueue(self, sql: str, params: tuple):
        self._queue.put((sql, params))

    def _writer_loop(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")  # better concurrency
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.5)
                if item is None:
                    self._queue.task_done()
                    break
                sql, params = item
                try:
                    conn.execute(sql, params)
                    conn.commit()
                except sqlite3.Error as e:
                    print(f"[LineageDB] Write error: {e} | SQL: {sql[:80]}")
                finally:
                    self._queue.task_done()
            except queue.Empty:
                continue
        conn.close()
