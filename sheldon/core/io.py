"""JSONL + atomic-write helpers. Every run artifact goes through here."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Where benchmark payloads live. Override with SHELDON_DATA."""
    return Path(os.environ.get("SHELDON_DATA", REPO_ROOT / "data"))


def runs_root() -> Path:
    return Path(os.environ.get("SHELDON_RUNS", REPO_ROOT / "runs"))


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def rewrite_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Replace a jsonl file's contents atomically.

    The append-only files are append-only in the normal case; this is for the
    one case that is not, which is dropping a task's rows so it can be run
    again. Same temp-file-then-rename as `write_json`, because a run artifact
    truncated by a crash halfway through a rewrite is worse than a stale one.
    """
    path = Path(path)
    if not path.exists():
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def write_json(path: str | Path, obj: Any) -> None:
    """Atomic: write to a sibling temp file, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(path)


def git_sha() -> str | None:
    """Recorded in every run manifest so a number can be traced back to code."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(out.stdout.strip())
    except Exception:
        return False


def load_dotenv(path: str | Path | None = None, override: bool = True) -> list[str]:
    """Load `.env` into the environment. **`.env` wins by default.**

    This is the opposite of the usual dotenv convention, on purpose. A stale
    `OPENAI_API_KEY` exported in a shell months ago silently shadows the correct
    one in `.env`, and the failure it produces — a 429 naming a key you did not
    choose — reads like a billing problem rather than a config problem. Losing
    an hour to that once is enough; the file you edited is the file that wins.

    Pass `override=False` for the conventional behaviour (CI exports winning).

    Returns the names it set, never the values.
    """
    path = Path(path) if path else REPO_ROOT / ".env"
    if not path.exists():
        return []
    loaded = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("\"'")
        if not key or not value:
            continue
        if override or key not in os.environ:
            shadowed = key in os.environ and os.environ[key] != value
            os.environ[key] = value
            loaded.append(f"{key}*" if shadowed else key)
    return loaded
