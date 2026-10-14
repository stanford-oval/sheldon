"""Importing this package is what registers the engines.

Each engine is a folder: `engine.py` holds the engine, decorated with
`@engines.register("name")`, beside whatever files only it uses. Add one by
adding a folder and one import line below. Nothing else needs to know it exists.
"""

from sheldon.core.registry import engines
from sheldon.engines import codex, react, rlm, simple, toolloop  # noqa: F401
from sheldon.engines.base import Engine, EngineResult

DEFAULT = "toolloop"


def get(name: str | None = None, **params) -> Engine:
    return engines.get(name or DEFAULT)(**params)


__all__ = ["Engine", "EngineResult", "engines", "get", "DEFAULT"]
