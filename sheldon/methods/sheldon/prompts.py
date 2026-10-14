"""Rendering this method's templates.

Each template has two sections, `[system]` and `[user]`, split in the source
before substitution. The submit and ask plumbing lives in `methods/stage.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template

from sheldon.methods.stage import Prompt, split_source

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"

_env = Environment(
    loader=FileSystemLoader(str(PROMPT_DIR)),
    undefined=StrictUndefined,
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)
_compiled: dict[str, dict[str, Template]] = {}


def render(name: str, **context: Any) -> Prompt:
    if name not in _compiled:
        source = (PROMPT_DIR / f"{name}.prompt").read_text(encoding="utf-8")
        _compiled[name] = {k: _env.from_string(v) for k, v in split_source(source).items()}
    sections = _compiled[name]
    system = sections["system"].render(**context).strip() if "system" in sections else ""
    user = sections["user"].render(**context).strip()
    if not user:
        raise ValueError(f"{name}.prompt rendered an empty [user] section")
    return Prompt(system=system, user=user)
