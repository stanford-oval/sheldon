"""Importing this package is what registers the methods.

Add a method: drop a module in here, decorate with `@methods.register("name")`,
and add one import line below.
"""

from sheldon.core.registry import methods
from sheldon.methods import direct, sheldon  # noqa: F401
from sheldon.methods.base import Method


def get(name: str, **params) -> Method:
    return methods.get(name)(**params)


__all__ = ["Method", "methods", "get"]
