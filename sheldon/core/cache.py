"""Content-addressed disk cache for model calls.

This is the single most important thing for iteration speed. Change a scorer,
re-run the whole benchmark, pay nothing: every LLM call that has not changed is
served from disk. Keys are a hash of the full request, so any prompt edit is
automatically a cache miss and nothing goes stale silently.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

from sheldon.core.io import REPO_ROOT


def cache_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class DiskCache:
    """SQLite-backed KV store. Safe for many asyncio tasks in one process."""

    def __init__(self, path: str | Path | None = None, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = Path(path or os.environ.get("SHELDON_CACHE", REPO_ROOT / ".cache" / "llm.db"))
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self.hits = 0
        self.misses = 0

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(self.path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS kv ("
                "  k TEXT PRIMARY KEY,"
                "  v TEXT NOT NULL,"
                "  tag TEXT,"
                "  created REAL DEFAULT (julianday('now'))"
                ")"
            )
            self._conn.commit()
        return self._conn

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        with self._lock:
            row = self._db().execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def put(self, key: str, value: dict[str, Any], tag: str = "") -> None:
        if not self.enabled:
            return
        blob = json.dumps(value, ensure_ascii=False, default=str)
        with self._lock:
            db = self._db()
            db.execute("INSERT OR REPLACE INTO kv (k, v, tag) VALUES (?, ?, ?)", (key, blob, tag))
            db.commit()

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
